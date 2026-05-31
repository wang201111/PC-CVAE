"""
================================================================================
Noise Robustness Experiment - PC-CVAE (Solubility System)
================================================================================

Key differences from the DNN solubility noise baseline:
    - PC-CVAE (with phi head + L_cycle) replaces DNN
    - No early stopping; fixed epochs; torch.manual_seed is fixed before each
      training run to ensure that injected noise is the only variable
    - Physics evaluation uses evaluate_with_predictor(cvae.predict, ...)

Dataset split:
    - Train / Val : from fixed_splits (interpolation domain pre-split)
    - Near-range extrapolation: 50 < T < 100°C  (held-out evaluation)
    - Far-range extrapolation : T >= 100°C        (held-out evaluation)

After all repeats per noise level, the best-val-R2 model is loaded and
evaluated separately on near / far domains.

File Location: experiments/solubility/noise/cvae.py
================================================================================
"""

import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
from pc_cvae_solubility import CVAEConfig, CVAEPhysicsModel, LowDimInfo
from utils_solubility import PhysicalConsistencyEvaluator

warnings.filterwarnings('ignore')


# ==============================================================================
# Utilities
# ==============================================================================

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger(__name__)


def load_ternary_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Column order: [T, W_Na2SO4, W_MgSO4] -> X:(N,2), y:(N,)"""
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    data      = pd.read_excel(filepath, engine='openpyxl')
    col_names = data.columns.tolist()
    temp_col  = next((c for c in col_names if 'T' in c or 'temp' in c.lower()), col_names[0])
    comp_cols = [c for c in col_names if c != temp_col]
    if len(comp_cols) < 2:
        raise ValueError(f"Expected >=2 composition columns, found {len(comp_cols)}")
    X = data[[temp_col, comp_cols[0]]].values.astype(np.float32)
    y = data[comp_cols[1]].values.astype(np.float32)
    return X, y


def move_to_device(m: LowDimEnsemble, device: torch.device) -> LowDimEnsemble:
    m.to(device); m.device = device; return m


def add_gaussian_noise(
    train_set: Tuple[np.ndarray, np.ndarray],
    noise_level: float,
    noise_idx: int,
    base_seed: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Inject Gaussian noise into training labels (identical to the baseline experiment)."""
    X_train, y_train = train_set
    noise_seed = base_seed + int(noise_level * 10000) + noise_idx * 100
    if noise_level == 0.0:
        return X_train.copy(), y_train.copy(), noise_seed
    np.random.seed(noise_seed)
    y_range = y_train.max() - y_train.min()
    y_noisy = y_train + np.random.normal(0, noise_level * y_range, size=y_train.shape)
    return X_train.copy(), y_noisy, noise_seed


def _compute_metrics(y_true, y_pred, prefix: str) -> Dict[str, float]:
    return {
        f'{prefix}_r2':   float(sk_metrics.r2_score(y_true, y_pred)),
        f'{prefix}_rmse': float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred))),
        f'{prefix}_mae':  float(sk_metrics.mean_absolute_error(y_true, y_pred)),
    }


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class NoiseRobustnessConfig:
    # Train / Val (fixed split from interpolation domain)
    data_dir: Path = PROJECT_ROOT / 'data' / 'solubility' / 'fixed_splits'
    train_file:      str = 'train_set.xlsx'
    validation_file: str = 'val_set.xlsx'

    # Extrapolation evaluation datasets
    extrap_dir: Path = PROJECT_ROOT / 'data' / 'solubility' / 'split_by_temperature'
    near_extrap_file: str = 'near-range extrapolation.xlsx'
    far_extrap_file:  str = 'far-range extrapolation.xlsx'

    # Boundary models
    models_dir: Path = PROJECT_ROOT / 'models' / 'Low_dim_model' / 'solubility'
    input_boundary_model_file:  str = 'MgSO4-H2O.pth'
    output_boundary_model_file: str = 'Na2SO4-H2O.pth'

    # PC-CVAE hyperparameters
    cvae_config: CVAEConfig = field(default_factory=lambda: CVAEConfig(
        LATENT_DIM=1,
        HIDDEN_DIMS=[128, 256, 256, 128], DROPOUT=0.1,
        LEARNING_RATE=1e-3, BATCH_SIZE=64, N_EPOCHS=200, WEIGHT_DECAY=1e-5,
        LAMBDA_KL=0.001,
        LAMBDA_COLLOCATION_Na2SO4=1, LAMBDA_COLLOCATION_MgSO4=1,
        N_COLLOCATION_POINTS=64, COLLOCATION_T_RANGE=(-10.0, 200.0),
        Z_LOW=-2.0, Z_HIGH=2.0, Z_COLLOC_WIDTH=0.5,
        PHI_HIDDEN_DIMS=[64, 64], LAMBDA_CYCLE=1.0, N_CYCLE_POINTS=64,
        CYCLE_T_RANGE=(-10.0, 200.0),
        USE_EARLY_STOPPING=False, USE_LR_SCHEDULER=True,
        LR_SCHEDULER_TYPE='cosine', LR_MIN=1e-5, DEVICE='auto', VERBOSE=False,
    ))

    # Physics evaluation
    t_min: float = -10.0
    t_max: float = 200.0
    boundary_decay_lambda:   float = 5.0
    smoothness_decay_lambda: float = 15.0

    # Noise configuration
    noise_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20]
    )
    n_noise_repeats: int = 10

    # Seeds
    training_seed:   int = 42
    noise_base_seed: int = 20000

    # Output
    output_dir: Path = (
        PROJECT_ROOT / 'results' / 'solubility' / 'noise' / 'cvae_results'
    )
    save_predictions: bool = True
    save_metrics:     bool = True
    excel_prefix:     str  = 'noisy_cvae_'
    device:    str = 'auto'
    log_level: int = logging.INFO

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cvae_config.DEVICE = self.device


# ==============================================================================
# Single Experiment Runner
# ==============================================================================

class SingleNoiseExperimentRunner:
    def __init__(self, config, output_dir,
                 input_boundary_model=None, output_boundary_model=None):
        self.config   = config
        self.out_dir  = output_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger   = get_logger(self.__class__.__name__, config.log_level)
        self.results: Dict = {}
        self.input_boundary_model  = input_boundary_model
        self.output_boundary_model = output_boundary_model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_cvae(self, X_train: np.ndarray, y_noisy: np.ndarray) -> CVAEPhysicsModel:
        torch.manual_seed(self.config.training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.training_seed)

        low_dim_list = None
        if self.input_boundary_model is not None and self.output_boundary_model is not None:
            low_dim_list = [
                LowDimInfo(model=self.output_boundary_model,
                           name='MgSO4_H2O', constraint_type='Na2SO4'),
                LowDimInfo(model=self.input_boundary_model,
                           name='Na2SO4_H2O', constraint_type='MgSO4'),
            ]

        cvae    = CVAEPhysicsModel(input_dim=3, condition_dim=1, config=self.config.cvae_config)
        history = cvae.fit(X=X_train, y=y_noisy,
                           low_dim_list=low_dim_list, X_val=None, y_val=None)

        last = history['train'][-1] if history['train'] else {}
        self.logger.info(
            f"  CVAE done — total={last.get('total', float('nan')):.4f}  "
            f"recon={last.get('recon', float('nan')):.4f}  "
            f"cycle={last.get('cycle', float('nan')):.4f}"
        )
        return cvae

    # ------------------------------------------------------------------
    # Evaluation (train / val / near / far)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        cvae:    CVAEPhysicsModel,
        X_train: np.ndarray, y_noisy: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        X_near:  np.ndarray, y_near:  np.ndarray,
        X_far:   np.ndarray, y_far:   np.ndarray,
    ) -> Dict[str, Any]:
        preds = {
            'train': cvae.predict(X_train),
            'val':   cvae.predict(X_val),
            'near':  cvae.predict(X_near),
            'far':   cvae.predict(X_far),
        }
        trues = {'train': y_noisy, 'val': y_val, 'near': y_near, 'far': y_far}

        m = {}
        for s in ('train', 'val', 'near', 'far'):
            m.update(_compute_metrics(trues[s], preds[s], s))

        self.logger.info(
            f"  train_r2={m['train_r2']:.4f}  val_r2={m['val_r2']:.4f}  "
            f"near_r2={m['near_r2']:.4f}  far_r2={m['far_r2']:.4f}"
        )
        return {'metrics': m, 'predictions': preds, 'true_values': trues}

    # ------------------------------------------------------------------
    # Physics evaluation
    # ------------------------------------------------------------------

    def run_physics_eval(self, cvae: CVAEPhysicsModel) -> Optional[Dict]:
        if self.input_boundary_model is None or self.output_boundary_model is None:
            return None
        try:
            T  = np.linspace(self.config.t_min, self.config.t_max, 100)
            W  = np.linspace(0, 55, 100)
            Tm, Wm = np.meshgrid(T, W)
            X_ph   = np.column_stack([Tm.ravel(), Wm.ravel()]).astype(np.float32)
            yp     = cvae.predict(X_ph)
            pred_d = np.column_stack([X_ph[:, 0], X_ph[:, 1], yp])

            evaluator = PhysicalConsistencyEvaluator(
                input_boundary_model=self.input_boundary_model,
                output_boundary_model=self.output_boundary_model,
                boundary_decay_lambda=self.config.boundary_decay_lambda,
                smoothness_decay_lambda=self.config.smoothness_decay_lambda,
            )
            score, results = evaluator.evaluate_with_predictor(
                predict_fn=cvae.predict, predicted_data=pred_d)
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
    # Saving
    # ------------------------------------------------------------------

    def _save(self, result: Dict, X_train, X_val, X_near, X_far,
              cvae: CVAEPhysicsModel) -> None:
        m  = result['metrics']
        pe = self.results.get('physical_evaluation')

        rows = [
            ['Train R2',   m['train_r2']],  ['Train RMSE', m['train_rmse']],
            ['Train MAE',  m['train_mae']],
            ['Val R2',     m['val_r2']],     ['Val RMSE',   m['val_rmse']],
            ['Val MAE',    m['val_mae']],
            ['Near-Range R2',   m['near_r2']],  ['Near-Range RMSE', m['near_rmse']],
            ['Near-Range MAE',  m['near_mae']],
            ['Far-Range R2',    m['far_r2']],   ['Far-Range RMSE',  m['far_rmse']],
            ['Far-Range MAE',   m['far_mae']],
            ['Physics Score',
             self.results.get('physics_score', float('nan'))],
            ['Boundary Consistency',
             pe.get('boundary_score',  float('nan')) if pe else float('nan')],
            ['Thermodynamic Smoothness',
             pe.get('smoothness_score', float('nan')) if pe else float('nan')],
        ]
        excel_dir = self.out_dir / 'excel'
        excel_dir.mkdir(exist_ok=True)
        pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
            excel_dir / f'{self.config.excel_prefix}metrics.xlsx',
            index=False, engine='openpyxl')

        if self.config.save_predictions:
            for split, X in [('train', X_train), ('val', X_val),
                              ('near', X_near),   ('far', X_far)]:
                pd.DataFrame({
                    'Temperature':       X[:, 0],
                    'Input_Composition': X[:, 1],
                    'Output_True':       result['true_values'][split],
                    'Output_Pred':       result['predictions'][split],
                    'Error':             result['true_values'][split] - result['predictions'][split],
                }).to_excel(
                    excel_dir / f'{self.config.excel_prefix}{split}_predictions.xlsx',
                    index=False, engine='openpyxl')

        # Save CVAE model for best-model evaluation
        model_dir = self.out_dir / 'model'
        model_dir.mkdir(exist_ok=True)
        cvae.save(str(model_dir / 'cvae.pth'))

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, train_set, val_set, near_set, far_set,
            noise_level, noise_idx) -> Dict[str, Any]:
        start = time.time()
        self.logger.info(f"  Noise={int(noise_level*100)}%  Repeat={noise_idx}")

        X_train, y_noisy, noise_seed = add_gaussian_noise(
            train_set, noise_level, noise_idx, self.config.noise_base_seed)
        X_val,  y_val  = val_set
        X_near, y_near = near_set
        X_far,  y_far  = far_set

        self.logger.info(
            f"  Noise injection: level={noise_level*100:.0f}%  seed={noise_seed}  "
            f"train={len(X_train)}  val={len(X_val)}  "
            f"near={len(X_near)}  far={len(X_far)}"
        )

        cvae   = self.train_cvae(X_train, y_noisy)
        result = self.evaluate(cvae, X_train, y_noisy, X_val, y_val,
                               X_near, y_near, X_far, y_far)
        self.run_physics_eval(cvae)

        if self.config.save_metrics:
            self._save(result, X_train, X_val, X_near, X_far, cvae)

        return {
            'noise_level':  noise_level, 'noise_idx':  noise_idx,
            'noise_seed':   noise_seed,  'n_train':    len(X_train),
            'metrics':      result['metrics'],
            'physics_score':      self.results.get('physics_score'),
            'physics_boundary':   (self.results.get('physical_evaluation') or {}).get('boundary_score'),
            'physics_smoothness': (self.results.get('physical_evaluation') or {}).get('smoothness_score'),
            'elapsed_time': time.time() - start,
        }


# ==============================================================================
# Experiment Manager
# ==============================================================================

class NoiseRobustnessExperimentManager:
    def __init__(self, config: NoiseRobustnessConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_set = load_ternary_data(config.data_dir / config.train_file)
        self.val_set   = load_ternary_data(config.data_dir / config.validation_file)
        self.near_set  = load_ternary_data(config.extrap_dir / config.near_extrap_file)
        self.far_set   = load_ternary_data(config.extrap_dir / config.far_extrap_file)
        self.logger.info(
            f"Datasets loaded: train={len(self.train_set[0])}  val={len(self.val_set[0])}  "
            f"near={len(self.near_set[0])}  far={len(self.far_set[0])}"
        )

        self.input_boundary_model  = None
        self.output_boundary_model = None
        self._load_boundary_models()
        self.all_results: Dict[float, List] = {n: [] for n in config.noise_levels}

    def _load_boundary_models(self) -> None:
        try:
            device = torch.device(self.config.device)
            in_p   = self.config.models_dir / self.config.input_boundary_model_file
            out_p  = self.config.models_dir / self.config.output_boundary_model_file
            if in_p.exists() and out_p.exists():
                self.input_boundary_model  = move_to_device(LowDimEnsemble.load(str(in_p)),  device)
                self.output_boundary_model = move_to_device(LowDimEnsemble.load(str(out_p)), device)
                self.logger.info(f"Boundary models loaded successfully -> {device}")
            else:
                self.logger.warning("Boundary model files not found — physics evaluation disabled")
        except Exception as e:
            self.logger.error(f"Failed to load boundary models: {e}")

    # ------------------------------------------------------------------

    def run_all_experiments(self) -> None:
        total = len(self.config.noise_levels) * self.config.n_noise_repeats
        self.logger.info(f"Solubility PC-CVAE noise robustness experiment: {total} runs")
        count = 0
        for noise_level in self.config.noise_levels:
            self.logger.info(f"\n{'='*60}\nNoise level {int(noise_level*100)}%\n{'='*60}")
            for noise_idx in range(self.config.n_noise_repeats):
                count += 1
                out_dir = (self.config.output_dir
                           / f'noise_{int(noise_level*100):03d}'
                           / f'repeat_{noise_idx:02d}')
                runner = SingleNoiseExperimentRunner(
                    self.config, out_dir,
                    self.input_boundary_model, self.output_boundary_model)
                self.all_results[noise_level].append(
                    runner.run(self.train_set, self.val_set,
                               self.near_set, self.far_set, noise_level, noise_idx))
                self.logger.info(f"Progress: {count}/{total}")

            # After all repeats for this noise level, load best-val-R2 model for dedicated evaluation
            self._best_model_evaluation(noise_level)

        self._save_summary()

    # ------------------------------------------------------------------

    def _best_model_evaluation(self, noise_level: float) -> None:
        """Select the best repeat by val R2, load its model, and evaluate on near/far."""
        res      = self.all_results[noise_level]
        val_r2s  = [r['metrics']['val_r2'] for r in res]
        best_idx = int(np.argmax(val_r2s))
        self.logger.info(
            f"\n  [noise={int(noise_level*100)}%] Best repeat: #{best_idx}  "
            f"val_r2={val_r2s[best_idx]:.4f}"
        )

        model_path = (self.config.output_dir
                      / f'noise_{int(noise_level*100):03d}'
                      / f'repeat_{best_idx:02d}' / 'model' / 'cvae.pth')
        cvae = CVAEPhysicsModel.load(str(model_path))

        best_dir = self.config.output_dir / f'noise_{int(noise_level*100):03d}' / 'best_model'
        best_dir.mkdir(exist_ok=True)

        for tag, (X_ext, y_ext) in [
            ('near_extrap', self.near_set),
            ('far_extrap',  self.far_set),
        ]:
            y_pred = cvae.predict(X_ext)
            m = _compute_metrics(y_ext, y_pred, tag.split('_')[0])
            self.logger.info(
                f"  Best model {tag}: "
                f"R2={list(m.values())[0]:.4f}  "
                f"RMSE={list(m.values())[1]:.4f}  "
                f"MAE={list(m.values())[2]:.4f}"
            )
            pd.DataFrame({
                'y_true': y_ext, 'y_pred': y_pred,
                'residual': y_ext - y_pred,
            }).to_excel(
                best_dir / f'{self.config.excel_prefix}{tag}_predictions.xlsx',
                index=False, engine='openpyxl')

        import shutil
        shutil.copy(model_path, best_dir / 'cvae_best.pth')
        self.logger.info(f"  Best model artefacts -> {best_dir}")

    # ------------------------------------------------------------------

    def _save_summary(self) -> None:
        summary_dir = self.config.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)

        STAT_KEYS  = ['train_r2', 'train_rmse', 'train_mae',
                      'val_r2',   'val_rmse',   'val_mae',
                      'near_r2',  'near_rmse',  'near_mae',
                      'far_r2',   'far_rmse',   'far_mae']
        STAT_NAMES = ['Train R2',       'Train RMSE',      'Train MAE',
                      'Val R2',         'Val RMSE',        'Val MAE',
                      'Near-Range R2',  'Near-Range RMSE', 'Near-Range MAE',
                      'Far-Range R2',   'Far-Range RMSE',  'Far-Range MAE']
        STAT_COLS  = ['Train_R2', 'Train_RMSE', 'Train_MAE',
                      'Val_R2',   'Val_RMSE',   'Val_MAE',
                      'Near_R2',  'Near_RMSE',  'Near_MAE',
                      'Far_R2',   'Far_RMSE',   'Far_MAE']
        PHYS_RKEYS = ['physics_score', 'physics_boundary', 'physics_smoothness']
        PHYS_NAMES = ['Physics Score', 'Boundary Consistency', 'Thermodynamic Smoothness']
        PHYS_COLS  = ['Physics_Score', 'Boundary_Consistency', 'Thermodynamic_Smoothness']

        wide_rows = []

        for nl in self.config.noise_levels:
            res = self.all_results[nl]
            if not res:
                continue

            noise_tag = f'noise_{int(nl * 100):03d}'
            nl_dir    = summary_dir / noise_tag
            nl_dir.mkdir(exist_ok=True)

            sm_rows = []
            for key, name in zip(STAT_KEYS, STAT_NAMES):
                vals = [r['metrics'][key] for r in res]
                sm_rows.append({
                    'Metric':     name,
                    'Mean Value': float(np.mean(vals)),
                    'Std':        float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                })
            for pk, pn in zip(PHYS_RKEYS, PHYS_NAMES):
                vals       = [r.get(pk, float('nan')) for r in res]
                vals_clean = [v for v in vals if not np.isnan(float(v))]
                sm_rows.append({
                    'Metric':     pn,
                    'Mean Value': float(np.mean(vals_clean)) if vals_clean else float('nan'),
                    'Std':        float(np.std(vals_clean, ddof=1)) if len(vals_clean) > 1 else float('nan'),
                })
            pd.DataFrame(sm_rows).to_excel(
                nl_dir / 'summary_metrics.xlsx', index=False, engine='openpyxl')

            wide_row = {'Noise_Level': f'{int(nl * 100)}%'}
            for key, col in zip(STAT_KEYS, STAT_COLS):
                vals = [r['metrics'][key] for r in res]
                wide_row[f'{col}_Mean'] = float(np.mean(vals))
                wide_row[f'{col}_Std']  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            for pk, col in zip(PHYS_RKEYS, PHYS_COLS):
                vals       = [r.get(pk, float('nan')) for r in res]
                vals_clean = [v for v in vals if not np.isnan(float(v))]
                wide_row[f'{col}_Mean'] = float(np.mean(vals_clean)) if vals_clean else float('nan')
                wide_row[f'{col}_Std']  = (float(np.std(vals_clean, ddof=1))
                                           if len(vals_clean) > 1 else float('nan'))
            wide_rows.append(wide_row)

        pd.DataFrame(wide_rows).to_excel(
            summary_dir / 'overall_metrics.xlsx', index=False, engine='openpyxl')

        lines = [
            '=' * 70,
            'Solubility PC-CVAE Noise Robustness - Summary Report',
            '=' * 70,
            f"Generated:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Noise levels:   {[f'{int(n*100)}%' for n in self.config.noise_levels]}",
            f"Repeats/level:  {self.config.n_noise_repeats}",
            f"Training seed:  {self.config.training_seed} (FIXED — noise is the only variable)",
            f"CVAE epochs:    {self.config.cvae_config.N_EPOCHS} (fixed, no early stopping)",
            '',
            '-' * 70,
            'Results (Near-Range R2  /  Far-Range R2)',
            '-' * 70,
        ]
        for row in wide_rows:
            lines.append(
                f"{row['Noise_Level']:>5} noise: "
                f"Near R2 = {row['Near_R2_Mean']:.6f} +/- {row['Near_R2_Std']:.6f}  "
                f"Far R2 = {row['Far_R2_Mean']:.6f} +/- {row['Far_R2_Std']:.6f}"
            )
        lines.append('=' * 70)
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
    config.boundary_decay_lambda   = 5.0
    config.smoothness_decay_lambda = 15.0
    config.save_predictions = True
    config.save_metrics     = True
    config.log_level        = logging.INFO

    config.cvae_config.LATENT_DIM                = 1
    config.cvae_config.N_EPOCHS                  = 500
    config.cvae_config.LAMBDA_KL                 = 0.001
    config.cvae_config.LAMBDA_COLLOCATION_Na2SO4 = 1
    config.cvae_config.LAMBDA_COLLOCATION_MgSO4  = 1
    config.cvae_config.LAMBDA_CYCLE              = 1.0
    config.cvae_config.CYCLE_T_RANGE             = (-10.0, 200.0)
    config.cvae_config.USE_EARLY_STOPPING        = False
    config.cvae_config.DEVICE                    = config.device

    manager = NoiseRobustnessExperimentManager(config)
    manager.run_all_experiments()


if __name__ == '__main__':
    main()