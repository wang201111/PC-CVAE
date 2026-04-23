"""
================================================================================
Small Sample Sensitivity Experiment - PC-CVAE (Viscosity System)
================================================================================

PC-CVAE replaces DNN; no internal split, no early stopping, fixed epoch count.
cvae.predict(X) for direct evaluation; ViscosityPhysicsEvaluator + _CVAEWrapper
for physics evaluation.

File Location: experiments/viscosity/small_sample/cvae.py
================================================================================
"""

import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / 'models'))

import numpy as np
import pandas as pd
import torch
from sklearn import metrics as sk_metrics

from pc_cvae_viscosity import CVAEConfig, CVAEPhysicsModel, LowDimInfo
from low_dim_model import LowDimEnsemble
from utils_viscosity import ViscosityPhysicsEvaluator, get_logger

warnings.filterwarnings('ignore')

logger = get_logger(__name__)


def load_viscosity_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    data = pd.read_excel(filepath, engine='openpyxl')
    return data.iloc[:, :4].values.astype(np.float32), data.iloc[:, 4].values.astype(np.float32)


def move_to_device(m, device):
    m.to(device); m.device = device; return m


def sample_from_pool(train_pool, fraction, sampling_idx, base_seed=10000):
    X_pool, y_pool = train_pool
    seed = base_seed + int(fraction * 1000) + sampling_idx * 100
    np.random.seed(seed)
    idx = np.random.choice(len(X_pool), size=int(len(X_pool) * fraction), replace=False)
    return X_pool[idx].copy(), y_pool[idx].copy(), seed


class _CVAEWrapper:
    """Adapts cvae.predict(X) to trainer.predict(X, return_original_scale=True)."""
    def __init__(self, cvae): self._cvae = cvae
    def predict(self, X, return_original_scale=True): return self._cvae.predict(X)


@dataclass
class SmallSampleConfig:
    data_dir: Path = PROJECT_ROOT / 'data' / 'viscosity' / 'fixed_splits'
    train_pool_file:  str = 'train_set.xlsx'
    validation_file:  str = 'val_set.xlsx'
    near_extrap_file: str = 'near-range extrapolation.xlsx'
    far_extrap_file:  str = 'far-range extrapolation.xlsx'

    models_dir: Path = PROJECT_ROOT / 'models' / 'Low_dim_model' / 'viscosity'
    mch_hmn_model_file: str = 'MCH_HMN.pth'
    mch_dec_model_file: str = 'MCH_cis_Decalin.pth'
    dec_hmn_model_file: str = 'cis_Decalin_HMN.pth'

    cvae_config: CVAEConfig = field(default_factory=lambda: CVAEConfig(
        LATENT_DIM=2,
        HIDDEN_DIMS=[128, 256, 256, 128], DROPOUT=0.1,
        LEARNING_RATE=1e-3, BATCH_SIZE=32, N_EPOCHS=500, WEIGHT_DECAY=1e-5,
        LAMBDA_KL=0.001,
        LAMBDA_COLLOCATION_MCH=1.0, LAMBDA_COLLOCATION_DEC=1.0, LAMBDA_COLLOCATION_HMN=1.0,
        N_COLLOCATION_POINTS=64,
        COLLOCATION_T_RANGE=(20.0, 80.0), COLLOCATION_P_RANGE=(1e5, 1e8),
        Z_LOW=-2.0, Z_HIGH=2.0, Z_COLLOC_WIDTH=0.5,
        PHI_HIDDEN_DIMS=[64, 64], LAMBDA_CYCLE=1.0, N_CYCLE_POINTS=64,
        CYCLE_T_RANGE=(20.0, 80.0), CYCLE_P_RANGE=(1e5, 1e8),
        USE_EARLY_STOPPING=False, USE_LR_SCHEDULER=True,
        LR_SCHEDULER_TYPE='cosine', LR_MIN=1e-5, DEVICE='auto', VERBOSE=False,
    ))

    sample_fractions: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 1.00])
    n_sampling_repeats: int = 10
    sampling_base_seed: int = 10000

    t_min: float = 20.0; t_max: float = 80.0
    p_min: float = 1e5;  p_max: float = 1e8
    boundary_decay_lambda: float = 5.0; smoothness_decay_lambda: float = 15.0

    output_dir: Path = PROJECT_ROOT / 'results' / 'viscosity' / 'small_sample' / 'cvae_results'
    save_predictions: bool = True; save_metrics: bool = True; save_cvae_history: bool = True
    excel_prefix: str = 'cvae_'; device: str = 'auto'; log_level: int = logging.INFO

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cvae_config.DEVICE = self.device


class SingleExperimentRunner:
    def __init__(self, config, output_dir, model_mch_hmn=None, model_mch_dec=None, model_dec_hmn=None):
        self.config = config; self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        self.results = {}
        self.model_mch_hmn = model_mch_hmn
        self.model_mch_dec = model_mch_dec
        self.model_dec_hmn = model_dec_hmn

    def train_cvae(self, X_train, y_train):
        low_dim_list = None
        if all(m is not None for m in (self.model_mch_hmn, self.model_mch_dec, self.model_dec_hmn)):
            low_dim_list = [
                LowDimInfo(model=self.model_dec_hmn, name='cis_Decalin_HMN', boundary_type='mch_zero'),
                LowDimInfo(model=self.model_mch_hmn, name='MCH_HMN',         boundary_type='dec_zero'),
                LowDimInfo(model=self.model_mch_dec, name='MCH_cis_Decalin', boundary_type='hmn_zero'),
            ]
        cvae = CVAEPhysicsModel(config=self.config.cvae_config)
        history = cvae.fit(X=X_train,
                           y=y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train,
                           low_dim_list=low_dim_list, X_val=None, y_val=None)
        tr = history.get('train_loss', [])
        if tr:
            self.logger.info(f"  CVAE: total={tr[-1]:.4f}  "
                             f"cycle={history['train_cycle'][-1]:.4f}")
        return cvae, history

    def evaluate(self, cvae, X_train, y_train, X_gval, y_gval, X_near, y_near, X_far, y_far):
        preds = {'train': cvae.predict(X_train).flatten(),
                 'val':   cvae.predict(X_gval).flatten(),
                 'near':  cvae.predict(X_near).flatten(),
                 'far':   cvae.predict(X_far).flatten()}
        trues = {'train': y_train, 'val': y_gval, 'near': y_near, 'far': y_far}
        m = {}
        for s in ('train', 'val', 'near', 'far'):
            m[f'{s}_r2']   = float(sk_metrics.r2_score(trues[s], preds[s]))
            m[f'{s}_rmse'] = float(np.sqrt(sk_metrics.mean_squared_error(trues[s], preds[s])))
            m[f'{s}_mae']  = float(sk_metrics.mean_absolute_error(trues[s], preds[s]))
        self.logger.info(f"  train_r²={m['train_r2']:.4f}  val_r²={m['val_r2']:.4f}  "
                         f"near_r²={m['near_r2']:.4f}  far_r²={m['far_r2']:.4f}")
        return {'metrics': m, 'predictions': preds, 'true_values': trues}

    def run_physics_eval(self, cvae):
        if any(m is None for m in (self.model_mch_hmn, self.model_mch_dec, self.model_dec_hmn)): return None
        try:
            ev = ViscosityPhysicsEvaluator(
                teacher_models=(self.model_mch_hmn, self.model_dec_hmn, self.model_mch_dec),
                temp_range=(self.config.t_min, self.config.t_max),
                pressure_range=(self.config.p_min, self.config.p_max),
            )
            score, results = ev.evaluate_full(_CVAEWrapper(cvae))
            self.results['physics_score'] = score; self.results['physical_evaluation'] = results
            self.logger.info(f"  physics={score:.4f}  boundary={results.get('boundary_score', float('nan')):.4f}")
            return results
        except Exception as e:
            self.logger.error(f"Physics eval failed: {e}"); return None

    def _save(self, result, history):
        m  = result['metrics']
        pe = self.results.get('physical_evaluation')

        physics_score    = self.results.get('physics_score', float('nan'))
        boundary_score   = pe.get('boundary_score',   float('nan')) if pe else float('nan')
        smoothness_score = pe.get('smoothness_score', float('nan')) if pe else float('nan')

        rows = [
            ['Train R²',                 m['train_r2']],
            ['Train RMSE',               m['train_rmse']],
            ['Train MAE',                m['train_mae']],
            ['Val R²',                   m['val_r2']],
            ['Val RMSE',                 m['val_rmse']],
            ['Val MAE',                  m['val_mae']],
            ['Near-Range R²',            m['near_r2']],
            ['Near-Range RMSE',          m['near_rmse']],
            ['Near-Range MAE',           m['near_mae']],
            ['Far-Range R²',             m['far_r2']],
            ['Far-Range RMSE',           m['far_rmse']],
            ['Far-Range MAE',            m['far_mae']],
            ['Physics Score',            physics_score],
            ['Boundary Consistency',     boundary_score],
            ['Thermodynamic Smoothness', smoothness_score],
        ]
        excel_dir = self.output_dir / 'excel'; excel_dir.mkdir(exist_ok=True)
        pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
            excel_dir / f'{self.config.excel_prefix}metrics.xlsx', index=False, engine='openpyxl')
        if self.config.save_cvae_history and history.get('train_loss'):
            pd.DataFrame([{
                'epoch': ep, 'train_total': tl,
                'train_cycle': history['train_cycle'][ep] if ep < len(history.get('train_cycle', [])) else float('nan'),
                'train_colloc_mch': history['train_colloc_mch'][ep] if ep < len(history.get('train_colloc_mch', [])) else float('nan'),
                'train_colloc_dec': history['train_colloc_dec'][ep] if ep < len(history.get('train_colloc_dec', [])) else float('nan'),
                'train_colloc_hmn': history['train_colloc_hmn'][ep] if ep < len(history.get('train_colloc_hmn', [])) else float('nan'),
            } for ep, tl in enumerate(history['train_loss'])]).to_excel(
                excel_dir / f'{self.config.excel_prefix}cvae_history.xlsx', index=False, engine='openpyxl')

    def run(self, train_pool, global_val_set, near_set, far_set, fraction, sampling_idx):
        start = time.time()
        X_gval, y_gval = global_val_set
        X_near, y_near = near_set
        X_far,  y_far  = far_set
        X_sampled, y_sampled, seed = sample_from_pool(
            train_pool, fraction, sampling_idx, self.config.sampling_base_seed)
        self.logger.info(f"  {int(fraction*100)}%  idx={sampling_idx}  n={len(X_sampled)}")

        cvae, history = self.train_cvae(X_sampled, y_sampled)
        result = self.evaluate(cvae, X_sampled, y_sampled, X_gval, y_gval, X_near, y_near, X_far, y_far)
        self.run_physics_eval(cvae)

        # Save model weights (for best model evaluation loading)
        model_dir = self.output_dir / 'model'
        model_dir.mkdir(exist_ok=True)
        cvae.save(str(model_dir / 'cvae.pth'))

        if self.config.save_metrics: self._save(result, history)

        return {'fraction': fraction, 'sampling_idx': sampling_idx, 'sampling_seed': seed,
                'n_sampled_full': len(X_sampled), 'n_internal_train': len(X_sampled), 'n_internal_val': 0,
                'metrics': result['metrics'],
                'model_path': str(model_dir / 'cvae.pth'),
                'physics_score':      self.results.get('physics_score'),
                'physics_boundary':   self.results.get('physical_evaluation', {}).get('boundary_score'),
                'physics_smoothness': self.results.get('physical_evaluation', {}).get('smoothness_score'),
                'elapsed_time': time.time() - start}


class SmallSampleExperimentManager:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_pool = load_viscosity_data(config.data_dir / config.train_pool_file)
        self.global_val = load_viscosity_data(config.data_dir / config.validation_file)
        self.near_set   = load_viscosity_data(config.data_dir / config.near_extrap_file)
        self.far_set    = load_viscosity_data(config.data_dir / config.far_extrap_file)
        self.logger.info(f"Data: train={len(self.train_pool[0])} val={len(self.global_val[0])} "
                         f"near={len(self.near_set[0])} far={len(self.far_set[0])}")
        self.model_mch_hmn = self.model_mch_dec = self.model_dec_hmn = None
        self._load_models()
        self.all_results = {f: [] for f in config.sample_fractions}

    def _load_models(self):
        try:
            device = torch.device(self.config.device)
            p = {k: self.config.models_dir / getattr(self.config, f'{k}_model_file')
                 for k in ('mch_hmn', 'mch_dec', 'dec_hmn')}
            if all(v.exists() for v in p.values()):
                self.model_mch_hmn = move_to_device(LowDimEnsemble.load(str(p['mch_hmn'])), device)
                self.model_mch_dec = move_to_device(LowDimEnsemble.load(str(p['mch_dec'])), device)
                self.model_dec_hmn = move_to_device(LowDimEnsemble.load(str(p['dec_hmn'])), device)
                self.logger.info(f"Boundary models loaded → {device}")
            else: self.logger.warning("Boundary model files missing")
        except Exception as e: self.logger.error(f"Failed to load models: {e}")

    def run_all_experiments(self):
        total = len(self.config.sample_fractions) * self.config.n_sampling_repeats
        self.logger.info(f"Viscosity PC-CVAE small sample experiment: {total} runs")
        count = 0
        for frac in self.config.sample_fractions:
            self.logger.info(f"\n{'='*60}\n{int(frac*100)}%\n{'='*60}")
            for idx in range(self.config.n_sampling_repeats):
                count += 1
                out_dir = self.config.output_dir / f'fraction_{int(frac*100):03d}' / f'sampling_{idx:02d}'
                runner = SingleExperimentRunner(self.config, out_dir,
                    self.model_mch_hmn, self.model_mch_dec, self.model_dec_hmn)
                self.all_results[frac].append(
                    runner.run(self.train_pool, self.global_val, self.near_set, self.far_set, frac, idx))
                self.logger.info(f"Progress: {count}/{total}")
        self._save_summary()

    def _save_summary(self):
        summary_dir = self.config.output_dir / 'summary'; summary_dir.mkdir(exist_ok=True)

        STAT_KEYS  = ['train_r2',  'train_rmse', 'train_mae',
                      'val_r2',    'val_rmse',   'val_mae',
                      'near_r2',   'near_rmse',  'near_mae',
                      'far_r2',    'far_rmse',   'far_mae']
        STAT_NAMES = ['Train R²',  'Train RMSE', 'Train MAE',
                      'Val R²',    'Val RMSE',   'Val MAE',
                      'Near-Range R²',   'Near-Range RMSE', 'Near-Range MAE',
                      'Far-Range R²',    'Far-Range RMSE',  'Far-Range MAE']
        PHYS_ATTRS = [
            ('physics_score',       'Physics Score',            'Physics_Score'),
            ('physics_boundary',    'Boundary Consistency',     'Boundary_Consistency'),
            ('physics_smoothness',  'Thermodynamic Smoothness', 'Thermodynamic_Smoothness'),
        ]
        STAT_COLS  = ['Train_R2', 'Train_RMSE', 'Train_MAE',
                      'Val_R2',   'Val_RMSE',   'Val_MAE',
                      'Near_R2',  'Near_RMSE',  'Near_MAE',
                      'Far_R2',   'Far_RMSE',   'Far_MAE']

        wide_rows = []

        for frac in self.config.sample_fractions:
            res = self.all_results[frac]
            if not res:
                continue

            # ── Per-fraction summary_metrics.xlsx (12 rows) ──────────────────
            sm_rows = []
            for key, name in zip(STAT_KEYS, STAT_NAMES):
                vals = [r['metrics'][key] for r in res]
                sm_rows.append({
                    'Metric':     name,
                    'Mean Value': float(np.mean(vals)),
                    'Std':        float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                })
            for attr, label, _ in PHYS_ATTRS:
                vals = [r.get(attr) for r in res
                        if r.get(attr) is not None
                        and not np.isnan(float(r.get(attr, float('nan'))))]
                if vals:
                    sm_rows.append({'Metric': label,
                                    'Mean Value': float(np.mean(vals)),
                                    'Std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0})
                else:
                    sm_rows.append({'Metric': label, 'Mean Value': float('nan'), 'Std': float('nan')})

            frac_label = f'fraction_{int(frac*100):03d}'
            out_path = summary_dir / frac_label / 'summary_metrics.xlsx'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(sm_rows, columns=['Metric', 'Mean Value', 'Std']).to_excel(
                out_path, index=False, engine='openpyxl')
            self.logger.info(f"summary_metrics saved: {out_path.relative_to(self.config.output_dir)}")

            # ── Wide-format row (25 columns) ────────────────────────────────────────────
            wide_row = {'Sample_Fraction': f'{int(frac*100)}%'}
            for key, col in zip(STAT_KEYS, STAT_COLS):
                vals = [r['metrics'][key] for r in res]
                wide_row[f'{col}_Mean'] = float(np.mean(vals))
                wide_row[f'{col}_Std']  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            for attr, _, col in PHYS_ATTRS:
                vals       = [r.get(attr, float('nan')) for r in res]
                vals_clean = [v for v in vals if v is not None and not np.isnan(float(v))]
                if vals_clean:
                    wide_row[f'{col}_Mean'] = float(np.mean(vals_clean))
                    wide_row[f'{col}_Std']  = float(np.std(vals_clean, ddof=1)) if len(vals_clean) > 1 else 0.0
                else:
                    wide_row[f'{col}_Mean'] = float('nan')
                    wide_row[f'{col}_Std']  = float('nan')
            wide_rows.append(wide_row)

        pd.DataFrame(wide_rows).to_excel(
            summary_dir / 'overall_metrics.xlsx', index=False, engine='openpyxl')

        lines = ['='*70, 'Viscosity PC-CVAE Small Sample - Summary', '='*70,
                 f"Generated:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 f"Sample fractions: {[f'{int(f*100)}%' for f in self.config.sample_fractions]}",
                 f"Repeats/fraction: {self.config.n_sampling_repeats}",
                 f"CVAE epochs:      {self.config.cvae_config.N_EPOCHS} (fixed, no early stopping)", '']
        for frac, row in zip(self.config.sample_fractions, wide_rows):
            res = self.all_results[frac]
            n = res[0]['n_sampled_full'] if res else 0
            lines.append(f"{row['Sample_Fraction']:>5} ({n:3d} pts): "
                         f"Near R² = {row['Near_R2_Mean']:.6f} ± {row['Near_R2_Std']:.6f}  "
                         f"Far R² = {row['Far_R2_Mean']:.6f} ± {row['Far_R2_Std']:.6f}")
        (summary_dir / 'summary_report.txt').write_text('\n'.join(lines), encoding='utf-8')
        self.logger.info('\n' + '\n'.join(lines))
        self._save_best_model_per_fraction(summary_dir)

    def _save_best_model_per_fraction(self, summary_dir: Path) -> None:
        """For each sample fraction, select the best repeat by Val R², load the CVAE,
        and independently evaluate on the near/far domains.

        Output written to summary/fraction_NNN/best_model/:
          - best_near_predictions.xlsx
          - best_far_predictions.xlsx
          - best_model_metrics.xlsx
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error as mae_fn
        X_near, y_near = self.near_set
        X_far,  y_far  = self.far_set

        for frac in self.config.sample_fractions:
            res = self.all_results[frac]
            if not res:
                continue
            val_r2_list = [r['metrics']['val_r2'] for r in res]
            best_idx    = int(np.argmax(val_r2_list))
            best_run    = res[best_idx]
            model_path  = Path(best_run.get('model_path', ''))
            if not model_path.exists():
                self.logger.warning(f"  [{int(frac*100)}%] Best model file not found, skipping: {model_path}")
                continue

            cvae     = CVAEPhysicsModel.load(str(model_path))
            best_dir = summary_dir / f'fraction_{int(frac*100):03d}' / 'best_model'
            best_dir.mkdir(parents=True, exist_ok=True)

            summary_rows = [
                ['Best Sampling Idx', best_run['sampling_idx']],
                ['Val R²',            val_r2_list[best_idx]],
            ]
            for tag, X, y_true_arr in [('near', X_near, y_near), ('far', X_far, y_far)]:
                y_pred = cvae.predict(X).flatten()
                y_true = y_true_arr.flatten()
                r2   = float(r2_score(y_true, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mae  = float(mae_fn(y_true, y_pred))
                label = 'Near-Range' if tag == 'near' else 'Far-Range'
                self.logger.info(
                    f"  [{int(frac*100)}%] best_model {label}: r²={r2:.4f}  rmse={rmse:.4f}  mae={mae:.4f}"
                )
                summary_rows += [
                    [f'{label} R²',   r2],
                    [f'{label} RMSE', rmse],
                    [f'{label} MAE',  mae],
                ]
                pd.DataFrame({
                    'y_true': y_true, 'y_pred': y_pred, 'residual': y_true - y_pred,
                }).to_excel(
                    best_dir / f'best_{tag}_predictions.xlsx', index=False, engine='openpyxl')

            pd.DataFrame(summary_rows, columns=['Metric', 'Value']).to_excel(
                best_dir / 'best_model_metrics.xlsx', index=False, engine='openpyxl')


def main():
    config = SmallSampleConfig()
    config.sample_fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    config.n_sampling_repeats = 10; config.sampling_base_seed = 10000
    config.save_predictions = True; config.save_metrics = True; config.save_cvae_history = True
    config.log_level = logging.INFO
    config.cvae_config.LATENT_DIM = 2; config.cvae_config.N_EPOCHS = 500
    config.cvae_config.BATCH_SIZE = 64; config.cvae_config.LAMBDA_CYCLE = 1.0
    config.cvae_config.USE_EARLY_STOPPING = False; config.cvae_config.DEVICE = config.device
    SmallSampleExperimentManager(config).run_all_experiments()

if __name__ == '__main__':
    main()