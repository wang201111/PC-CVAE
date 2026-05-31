"""
================================================================================
Noise Robustness Experiment - PC-CVAE (Viscosity System)
================================================================================

Viscosity PC-CVAE noise robustness experiment, symmetric with the solubility
CVAE version: 4D input, three boundary models, physics evaluation via
ViscosityPhysicsEvaluator + _CVAEWrapper.

Datasets:
    - train_set / val_set      : in-domain training and validation
    - near-range extrapolation : near-range extrapolation evaluation (held-out)
    - far-range extrapolation  : far-range extrapolation evaluation (held-out)

Best model evaluation:
    After all repeats for each noise level, the best repeat is selected by val R²;
    the saved model is loaded via cvae.load() and a dedicated report is produced
    for near / far domains.

File Location: experiments/viscosity/noise/cvae.py
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

from low_dim_model import LowDimEnsemble
from pc_cvae_viscosity import CVAEConfig, CVAEPhysicsModel, LowDimInfo
from utils_viscosity import ViscosityPhysicsEvaluator, get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


# ==============================================================================
# Utility functions
# ==============================================================================

def load_viscosity_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    data = pd.read_excel(filepath, engine='openpyxl')
    if data.shape[1] < 5:
        raise ValueError(f"Expected at least 5 columns, got: {data.shape[1]}")
    return data.iloc[:, :4].values.astype(np.float32), data.iloc[:, 4].values.astype(np.float32)


def move_to_device(m, device):
    m.to(device); m.device = device; return m


def add_gaussian_noise(
    train_set: Tuple[np.ndarray, np.ndarray],
    noise_level: float, noise_idx: int, base_seed: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    X_train, y_train = train_set
    noise_seed = base_seed + int(noise_level * 10000) + noise_idx * 100
    if noise_level == 0.0:
        return X_train.copy(), y_train.copy(), noise_seed
    np.random.seed(noise_seed)
    y_range = y_train.max() - y_train.min()
    return X_train.copy(), y_train + np.random.normal(0, noise_level * y_range, size=y_train.shape), noise_seed


class _CVAEWrapper:
    """Adapts cvae.predict(X) to trainer.predict(X, return_original_scale=True)."""
    def __init__(self, cvae): self._cvae = cvae
    def predict(self, X, return_original_scale=True): return self._cvae.predict(X)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class NoiseRobustnessConfig:
    # ── Data paths ────────────────────────────────────────────────────
    data_dir: Path = PROJECT_ROOT / 'data' / 'viscosity' / 'fixed_splits'
    train_file:       str = 'train_set.xlsx'
    validation_file:  str = 'val_set.xlsx'
    near_extrap_file: str = 'near-range extrapolation.xlsx'
    far_extrap_file:  str = 'far-range extrapolation.xlsx'

    # ── Boundary models ───────────────────────────────────────────────
    models_dir: Path = PROJECT_ROOT / 'models' / 'Low_dim_model' / 'viscosity'
    mch_hmn_model_file: str = 'MCH_HMN.pth'
    mch_dec_model_file: str = 'MCH_cis_Decalin.pth'
    dec_hmn_model_file: str = 'cis_Decalin_HMN.pth'

    # ── CVAE hyperparameters ──────────────────────────────────────────
    cvae_config: CVAEConfig = field(default_factory=lambda: CVAEConfig(
        LATENT_DIM=2, HIDDEN_DIMS=[128, 256, 256, 128], DROPOUT=0.1,
        LEARNING_RATE=1e-3, BATCH_SIZE=64, N_EPOCHS=200, WEIGHT_DECAY=1e-5,
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

    # ── Noise experiment parameters ───────────────────────────────────
    noise_levels:    List[float] = field(default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20])
    n_noise_repeats: int = 10

    training_seed:   int = 42
    noise_base_seed: int = 20000

    # ── Physics evaluation parameters ─────────────────────────────────
    t_min: float = 20.0; t_max: float = 80.0
    p_min: float = 1e5;  p_max: float = 1e8
    boundary_decay_lambda:   float = 5.0
    smoothness_decay_lambda: float = 15.0

    # ── Output ────────────────────────────────────────────────────────
    output_dir: Path = PROJECT_ROOT / 'results' / 'viscosity' / 'noise' / 'cvae_results'
    save_predictions:  bool = True
    save_metrics:      bool = True
    save_cvae_history: bool = False
    excel_prefix: str = 'noisy_cvae_'
    device:    str = 'auto'
    log_level: int = logging.INFO

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cvae_config.DEVICE = self.device


# ==============================================================================
# Single experiment runner
# ==============================================================================

class SingleNoiseExperimentRunner:
    def __init__(self, config, output_dir,
                 model_mch_hmn=None, model_mch_dec=None, model_dec_hmn=None):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger  = get_logger(self.__class__.__name__, config.log_level)
        self.results = {}
        self.model_mch_hmn = model_mch_hmn
        self.model_mch_dec = model_mch_dec
        self.model_dec_hmn = model_dec_hmn

    # ------------------------------------------------------------------
    # Train PC-CVAE
    # ------------------------------------------------------------------

    def train_cvae(self, X_train, y_noisy, training_seed: int = None) -> CVAEPhysicsModel:
        if training_seed is None:
            training_seed = self.config.training_seed
        torch.manual_seed(training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(training_seed)

        low_dim_list = None
        if all(m is not None for m in (self.model_mch_hmn, self.model_mch_dec, self.model_dec_hmn)):
            low_dim_list = [
                LowDimInfo(model=self.model_dec_hmn, name='cis_Decalin_HMN', boundary_type='mch_zero'),
                LowDimInfo(model=self.model_mch_hmn, name='MCH_HMN',         boundary_type='dec_zero'),
                LowDimInfo(model=self.model_mch_dec, name='MCH_cis_Decalin', boundary_type='hmn_zero'),
            ]

        cvae = CVAEPhysicsModel(config=self.config.cvae_config)
        history = cvae.fit(
            X=X_train,
            y=y_noisy.reshape(-1, 1) if y_noisy.ndim == 1 else y_noisy,
            low_dim_list=low_dim_list, X_val=None, y_val=None,
        )
        tr = history.get('train_loss', [])
        if tr:
            self.logger.info(
                f"  CVAE done — total={tr[-1]:.4f}  "
                f"cycle={history['train_cycle'][-1]:.4f}"
            )
        return cvae

    # ------------------------------------------------------------------
    # Evaluation (val/near/far are all clean data)
    # ------------------------------------------------------------------

    def evaluate(self, cvae, X_train, y_noisy,
                 X_val, y_val, X_near, y_near, X_far, y_far):
        preds = {
            'train': cvae.predict(X_train).flatten(),
            'val':   cvae.predict(X_val).flatten(),
            'near':  cvae.predict(X_near).flatten(),
            'far':   cvae.predict(X_far).flatten(),
        }
        trues = {
            'train': y_noisy, 'val': y_val, 'near': y_near, 'far': y_far,
        }
        m = {}
        for s in ('train', 'val', 'near', 'far'):
            m[f'{s}_r2']   = float(sk_metrics.r2_score(trues[s], preds[s]))
            m[f'{s}_rmse'] = float(np.sqrt(sk_metrics.mean_squared_error(trues[s], preds[s])))
            m[f'{s}_mae']  = float(sk_metrics.mean_absolute_error(trues[s], preds[s]))

        self.logger.info(
            f"  train_r²={m['train_r2']:.4f}  val_r²={m['val_r2']:.4f}  "
            f"near_r²={m['near_r2']:.4f}  far_r²={m['far_r2']:.4f}"
        )
        return {'metrics': m, 'predictions': preds, 'true_values': trues}

    # ------------------------------------------------------------------
    # Physics evaluation
    # ------------------------------------------------------------------

    def run_physics_eval(self, cvae: CVAEPhysicsModel) -> Optional[Dict]:
        if any(m is None for m in (self.model_mch_hmn, self.model_mch_dec, self.model_dec_hmn)):
            return None
        try:
            ev = ViscosityPhysicsEvaluator(
                teacher_models=(self.model_mch_hmn, self.model_dec_hmn, self.model_mch_dec),
                temp_range=(self.config.t_min, self.config.t_max),
                pressure_range=(self.config.p_min, self.config.p_max),
            )
            score, results = ev.evaluate_full(_CVAEWrapper(cvae))
            self.results['physics_score']       = score
            self.results['physical_evaluation'] = results
            self.logger.info(
                f"  physics={score:.4f}  "
                f"boundary={results.get('boundary_score', float('nan')):.4f}  "
                f"smoothness={results.get('smoothness_score', float('nan')):.4f}"
            )
            return results
        except Exception as e:
            self.logger.error(f"Physics eval failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Save (15 rows: train/val/near/far × 3 + physics × 3)
    # ------------------------------------------------------------------

    def _save(self, result):
        m  = result['metrics']
        pe = self.results.get('physical_evaluation')

        rows = [
            ['Train R²',                m['train_r2']],
            ['Train RMSE',              m['train_rmse']],
            ['Train MAE',               m['train_mae']],
            ['Val R²',                  m['val_r2']],
            ['Val RMSE',                m['val_rmse']],
            ['Val MAE',                 m['val_mae']],
            ['Near-Range R²',           m['near_r2']],
            ['Near-Range RMSE',         m['near_rmse']],
            ['Near-Range MAE',          m['near_mae']],
            ['Far-Range R²',            m['far_r2']],
            ['Far-Range RMSE',          m['far_rmse']],
            ['Far-Range MAE',           m['far_mae']],
            ['Physics Score',           self.results.get('physics_score', float('nan')) if pe else float('nan')],
            ['Boundary Consistency',    pe.get('boundary_score',   float('nan')) if pe else float('nan')],
            ['Thermodynamic Smoothness', pe.get('smoothness_score', float('nan')) if pe else float('nan')],
        ]
        excel_dir = self.output_dir / 'excel'
        excel_dir.mkdir(exist_ok=True)
        pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
            excel_dir / f'{self.config.excel_prefix}metrics.xlsx',
            index=False, engine='openpyxl')

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------

    def run(self, train_set, val_set, near_set, far_set, noise_level, noise_idx):
        start = time.time()
        self.logger.info(f"  Noise={int(noise_level*100)}%  Repeat={noise_idx}")

        X_train, y_noisy, noise_seed = add_gaussian_noise(
            train_set, noise_level, noise_idx, self.config.noise_base_seed)
        X_val,  y_val  = val_set
        X_near, y_near = near_set
        X_far,  y_far  = far_set

        self.logger.info(
            f"  Noise injection: {noise_level*100:.0f}%  seed={noise_seed}  "
            f"train={len(X_train)} (noisy)  val={len(X_val)}  "
            f"near={len(X_near)}  far={len(X_far)} (clean)"
        )

        effective_seed = self.config.training_seed + noise_idx
        cvae = self.train_cvae(X_train, y_noisy, training_seed=effective_seed)

        result = self.evaluate(cvae, X_train, y_noisy,
                               X_val, y_val, X_near, y_near, X_far, y_far)

        self.run_physics_eval(cvae)

        # Save model (for subsequent best model loading)
        model_dir = self.output_dir / 'model'
        model_dir.mkdir(exist_ok=True)
        cvae.save(str(model_dir / 'cvae.pth'))

        if self.config.save_metrics:
            self._save(result)

        return {
            'noise_level': noise_level, 'noise_idx': noise_idx,
            'noise_seed':  noise_seed,  'n_train':   len(X_train),
            'metrics':     result['metrics'],
            'physics_score':      self.results.get('physics_score'),
            'physics_boundary':   self.results.get('physical_evaluation', {}).get('boundary_score'),
            'physics_smoothness': self.results.get('physical_evaluation', {}).get('smoothness_score'),
            'elapsed_time': time.time() - start,
            'model_dir':    str(model_dir),
        }


# ==============================================================================
# Experiment manager
# ==============================================================================

class NoiseRobustnessExperimentManager:
    def __init__(self, config: NoiseRobustnessConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_set = load_viscosity_data(config.data_dir / config.train_file)
        self.val_set   = load_viscosity_data(config.data_dir / config.validation_file)
        self.near_set  = load_viscosity_data(config.data_dir / config.near_extrap_file)
        self.far_set   = load_viscosity_data(config.data_dir / config.far_extrap_file)
        self.logger.info(
            f"Data: train={len(self.train_set[0])}  val={len(self.val_set[0])}  "
            f"near={len(self.near_set[0])}  far={len(self.far_set[0])}"
        )

        self.model_mch_hmn = self.model_mch_dec = self.model_dec_hmn = None
        self._load_models()
        self.all_results = {n: [] for n in config.noise_levels}

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
            else:
                self.logger.warning("Boundary model files missing — physics evaluation will be disabled")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")

    def run_all_experiments(self):
        total = len(self.config.noise_levels) * self.config.n_noise_repeats
        self.logger.info(f"Viscosity PC-CVAE noise robustness experiment: {total} runs")
        count = 0
        for nl in self.config.noise_levels:
            self.logger.info(f"\n{'='*60}\nNoise level {int(nl*100)}%\n{'='*60}")
            for idx in range(self.config.n_noise_repeats):
                count += 1
                out_dir = self.config.output_dir / f'noise_{int(nl*100):03d}' / f'repeat_{idx:02d}'
                runner = SingleNoiseExperimentRunner(
                    self.config, out_dir,
                    self.model_mch_hmn, self.model_mch_dec, self.model_dec_hmn)
                self.all_results[nl].append(
                    runner.run(self.train_set, self.val_set,
                               self.near_set, self.far_set, nl, idx))
                self.logger.info(f"Progress: {count}/{total}")

            # Load best model for evaluation after each noise level
            self._best_model_evaluation(nl)

        self._save_summary()

    # ------------------------------------------------------------------
    # Best model evaluation
    # ------------------------------------------------------------------

    def _best_model_evaluation(self, noise_level: float) -> None:
        """Select the best repeat by val R², load the CVAE, and produce a dedicated
        near / far evaluation report."""
        res = self.all_results[noise_level]
        if not res:
            return

        best_idx = int(np.argmax([r['metrics']['val_r2'] for r in res]))
        best_res = res[best_idx]
        model_dir = Path(best_res['model_dir'])

        self.logger.info(
            f"\n[noise={int(noise_level*100)}%] Best repeat={best_idx}  "
            f"val_r²={best_res['metrics']['val_r2']:.4f}"
        )

        cvae = CVAEPhysicsModel.load(str(model_dir / 'cvae.pth'))

        best_dir = self.config.output_dir / 'best_model' / f'noise_{int(noise_level*100):03d}'
        best_dir.mkdir(parents=True, exist_ok=True)

        for tag, (X_ext, y_ext) in [
            ('near_extrap', self.near_set),
            ('far_extrap',  self.far_set),
        ]:
            y_pred = cvae.predict(X_ext).flatten()
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2   = float(r2_score(y_ext, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_ext, y_pred)))
            mae  = float(mean_absolute_error(y_ext, y_pred))
            self.logger.info(f"  Best {tag}: R²={r2:.4f}  RMSE={rmse:.6f}  MAE={mae:.6f}")
            pd.DataFrame({
                'y_true': y_ext, 'y_pred': y_pred,
                'residual': y_ext - y_pred,
            }).to_excel(
                best_dir / f'{self.config.excel_prefix}{tag}_predictions.xlsx',
                index=False, engine='openpyxl')

        import shutil
        shutil.copy(model_dir / 'cvae.pth', best_dir / 'cvae_best.pth')
        self.logger.info(f"Best model artefacts → {best_dir}")

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def _save_summary(self):
        summary_dir = self.config.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)

        STAT_KEYS  = ['train_r2',  'train_rmse', 'train_mae',
                      'val_r2',    'val_rmse',   'val_mae',
                      'near_r2',   'near_rmse',  'near_mae',
                      'far_r2',    'far_rmse',   'far_mae']
        STAT_NAMES = ['Train R²',      'Train RMSE',      'Train MAE',
                      'Val R²',        'Val RMSE',         'Val MAE',
                      'Near-Range R²', 'Near-Range RMSE', 'Near-Range MAE',
                      'Far-Range R²',  'Far-Range RMSE',  'Far-Range MAE']
        STAT_COLS  = ['Train_R2',      'Train_RMSE',      'Train_MAE',
                      'Val_R2',        'Val_RMSE',         'Val_MAE',
                      'Near_R2',       'Near_RMSE',       'Near_MAE',
                      'Far_R2',        'Far_RMSE',         'Far_MAE']
        PHYS_ATTRS = [
            ('physics_score',      'Physics Score',            'Physics_Score'),
            ('physics_boundary',   'Boundary Consistency',     'Boundary_Consistency'),
            ('physics_smoothness', 'Thermodynamic Smoothness', 'Thermodynamic_Smoothness'),
        ]

        wide_rows = []

        for nl in self.config.noise_levels:
            res = self.all_results[nl]
            if not res:
                continue

            # Per-noise-level summary_metrics.xlsx (15 rows)
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
                sm_rows.append({
                    'Metric':     label,
                    'Mean Value': float(np.mean(vals)) if vals else float('nan'),
                    'Std':        float(np.std(vals, ddof=1)) if len(vals) > 1 else float('nan'),
                })

            noise_label = f'noise_{int(nl*100):03d}'
            out_path = summary_dir / noise_label / 'summary_metrics.xlsx'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(sm_rows, columns=['Metric', 'Mean Value', 'Std']).to_excel(
                out_path, index=False, engine='openpyxl')
            self.logger.info(f"summary_metrics saved: {out_path.relative_to(self.config.output_dir)}")

            # Wide-format row
            wide_row = {'Noise_Level': f'{int(nl*100)}%'}
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

        lines = [
            '='*70,
            'Viscosity PC-CVAE Noise Robustness - Summary Report',
            '='*70,
            f"Generated:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Noise levels:   {[f'{int(n*100)}%' for n in self.config.noise_levels]}",
            f"Repeats/level:  {self.config.n_noise_repeats}",
            f"Training seed:  {self.config.training_seed} (base, +noise_idx per repeat)",
            f"CVAE epochs:    {self.config.cvae_config.N_EPOCHS} (fixed, no early stopping)",
            '', '-'*70, 'Results (Near-Range R² / Far-Range R²)', '-'*70,
        ]
        for row in wide_rows:
            lines.append(
                f"{row['Noise_Level']:>5} noise: "
                f"Near R² = {row['Near_R2_Mean']:.6f} ± {row['Near_R2_Std']:.6f}  "
                f"Far R²  = {row['Far_R2_Mean']:.6f} ± {row['Far_R2_Std']:.6f}"
            )
        lines.append('='*70)
        report = '\n'.join(lines)
        (summary_dir / 'summary_report.txt').write_text(report, encoding='utf-8')
        self.logger.info(f'\n{report}')


# ==============================================================================
# Main
# ==============================================================================

def main():
    config = NoiseRobustnessConfig()
    config.noise_levels     = [0.0, 0.05, 0.10, 0.15, 0.20]
    config.n_noise_repeats  = 10
    config.training_seed    = 42
    config.noise_base_seed  = 20000
    config.boundary_decay_lambda    = 5.0
    config.smoothness_decay_lambda  = 15.0
    config.save_predictions = True
    config.save_metrics     = True
    config.log_level        = logging.INFO

    config.cvae_config.LATENT_DIM              = 2
    config.cvae_config.N_EPOCHS                = 200
    config.cvae_config.LAMBDA_KL               = 0.001
    config.cvae_config.LAMBDA_COLLOCATION_MCH  = 1.0
    config.cvae_config.LAMBDA_COLLOCATION_DEC  = 1.0
    config.cvae_config.LAMBDA_COLLOCATION_HMN  = 1.0
    config.cvae_config.LAMBDA_CYCLE            = 1.0
    config.cvae_config.CYCLE_T_RANGE           = (20.0, 80.0)
    config.cvae_config.CYCLE_P_RANGE           = (1e5, 1e8)
    config.cvae_config.USE_EARLY_STOPPING      = False
    config.cvae_config.DEVICE                  = config.device

    NoiseRobustnessExperimentManager(config).run_all_experiments()


if __name__ == '__main__':
    main()