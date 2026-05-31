"""
================================================================================
Small Sample Sensitivity Experiment - PC-CVAE (Solubility System)
================================================================================

Key differences from the DNN baseline:
    - PC-CVAE (with phi head + L_cycle) replaces DNN
    - Fixed val set (from fixed_splits, excluded from sampling), no early stopping,
      fixed number of epochs
    - Evaluated directly via cvae.predict(X); no additional DNN required
    - Physics evaluation uses evaluate_with_predictor(cvae.predict, ...)
    - After all repeats for each fraction, the best-val-R² model is loaded
      for dedicated extrapolation evaluation

Datasets:
    Training pool : interpolation domain (T <= 50°C)  — sampled proportionally
    Validation set: fixed_splits/val_set.xlsx          — fixed, not sampled
    Near-range extrapolation: 50 < T < 100°C
    Far-range extrapolation : T >= 100°C

File Location: experiments/solubility/small_sample/cvae.py
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

from pc_cvae_solubility import CVAEConfig, CVAEPhysicsModel, LowDimInfo
from low_dim_model import LowDimEnsemble
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


def move_to_device(model: LowDimEnsemble, device: torch.device) -> LowDimEnsemble:
    model.to(device)
    model.device = device
    return model


def sample_from_pool(
    train_pool: Tuple[np.ndarray, np.ndarray],
    fraction: float,
    sampling_idx: int,
    base_seed: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Randomly sample from the training pool at the given fraction (val set is fixed and excluded)."""
    X_pool, y_pool = train_pool
    seed = base_seed + int(fraction * 1000) + sampling_idx * 100
    np.random.seed(seed)
    n_sample = int(len(X_pool) * fraction)
    idx      = np.random.choice(len(X_pool), size=n_sample, replace=False)
    return X_pool[idx].copy(), y_pool[idx].copy(), seed


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
class SmallSampleConfig:
    # ── Training pool (sampled proportionally) ────────────────────────────────
    train_pool_dir:  Path = PROJECT_ROOT / 'data' / 'solubility' / 'split_by_temperature'
    train_pool_file: str  = 'interpolation domain.xlsx'

    # ── Fixed validation set (excluded from sampling; consistent with noise experiment) ──
    val_dir:  Path = PROJECT_ROOT / 'data' / 'solubility' / 'fixed_splits'
    val_file: str  = 'val_set.xlsx'

    # ── Extrapolation evaluation datasets ────────────────────────────────────
    extrap_dir:       Path = PROJECT_ROOT / 'data' / 'solubility' / 'split_by_temperature'
    near_extrap_file: str  = 'near-range extrapolation.xlsx'
    far_extrap_file:  str  = 'far-range extrapolation.xlsx'

    # ── LowDimEnsemble boundary model paths ──────────────────────────────────
    models_dir: Path = PROJECT_ROOT / 'models' / 'Low_dim_model' / 'solubility'
    input_boundary_model_file:  str = 'MgSO4-H2O.pth'
    output_boundary_model_file: str = 'Na2SO4-H2O.pth'

    # ── PC-CVAE hyperparameters ───────────────────────────────────────────────
    cvae_config: CVAEConfig = field(default_factory=lambda: CVAEConfig(
        LATENT_DIM=1,
        HIDDEN_DIMS=[128, 256, 256, 128],
        DROPOUT=0.1,
        LEARNING_RATE=1e-3,
        BATCH_SIZE=64,
        N_EPOCHS=500,
        WEIGHT_DECAY=1e-5,
        LAMBDA_KL=0.001,
        LAMBDA_COLLOCATION_Na2SO4=1,
        LAMBDA_COLLOCATION_MgSO4=1,
        N_COLLOCATION_POINTS=64,
        COLLOCATION_T_RANGE=(-10.0, 200.0),
        Z_LOW=-2.0,
        Z_HIGH=2.0,
        Z_COLLOC_WIDTH=0.5,
        PHI_HIDDEN_DIMS=[64, 64],
        LAMBDA_CYCLE=1.0,
        N_CYCLE_POINTS=64,
        CYCLE_T_RANGE=(-10.0, 200.0),
        USE_EARLY_STOPPING=False,
        USE_LR_SCHEDULER=True,
        LR_SCHEDULER_TYPE='cosine',
        LR_MIN=1e-5,
        DEVICE='auto',
        VERBOSE=False,
    ))

    # ── Physics evaluation range ──────────────────────────────────────────────
    t_min: float = -10.0
    t_max: float = 200.0
    boundary_decay_lambda:   float = 5.0
    smoothness_decay_lambda: float = 15.0

    # ── Sampling configuration ────────────────────────────────────────────────
    sample_fractions: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    )
    n_sampling_repeats: int = 10
    sampling_base_seed: int = 10000

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: Path = (
        PROJECT_ROOT / 'results' / 'solubility' / 'small_sample' / 'cvae_results'
    )
    save_predictions:  bool = True
    save_metrics:      bool = True
    save_cvae_history: bool = True
    excel_prefix:      str  = 'cvae_'

    # ── Device ────────────────────────────────────────────────────────────────
    device:    str = 'auto'
    log_level: int = logging.INFO

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cvae_config.DEVICE = self.device


# ==============================================================================
# Single Experiment Runner
# ==============================================================================

class SingleExperimentRunner:
    """Single sampling run: train PC-CVAE -> phi evaluation (train/val/near/far) -> physics evaluation."""

    def __init__(
        self,
        config: SmallSampleConfig,
        output_dir: Path,
        input_boundary_model:  Optional[LowDimEnsemble] = None,
        output_boundary_model: Optional[LowDimEnsemble] = None,
    ) -> None:
        self.config                = config
        self.output_dir            = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger                = get_logger(self.__class__.__name__, config.log_level)
        self.results: Dict         = {}
        self.input_boundary_model  = input_boundary_model
        self.output_boundary_model = output_boundary_model

    # ------------------------------------------------------------------
    # Train PC-CVAE
    # ------------------------------------------------------------------

    def train_cvae(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[CVAEPhysicsModel, dict]:
        """Train PC-CVAE on sampled data (fixed epochs, no early stopping, val not passed)."""
        low_dim_list = None
        if self.input_boundary_model is not None and self.output_boundary_model is not None:
            low_dim_list = [
                LowDimInfo(model=self.output_boundary_model,
                           name='MgSO4_H2O',  constraint_type='Na2SO4'),
                LowDimInfo(model=self.input_boundary_model,
                           name='Na2SO4_H2O', constraint_type='MgSO4'),
            ]
        cvae    = CVAEPhysicsModel(input_dim=3, condition_dim=1, config=self.config.cvae_config)
        history = cvae.fit(X=X_train, y=y_train,
                           low_dim_list=low_dim_list, X_val=None, y_val=None)
        last = history['train'][-1] if history['train'] else {}
        self.logger.info(
            f"  CVAE: total={last.get('total', float('nan')):.4f}  "
            f"recon={last.get('recon', float('nan')):.4f}  "
            f"cycle={last.get('cycle', float('nan')):.4f}"
        )
        return cvae, history

    # ------------------------------------------------------------------
    # Evaluation (train / val / near / far)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        cvae:    CVAEPhysicsModel,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        X_near:  np.ndarray, y_near:  np.ndarray,
        X_far:   np.ndarray, y_far:   np.ndarray,
    ) -> Dict[str, Any]:
        """cvae.predict(X) accepts (N, 2) [T, W1] and returns (N,) W2."""
        splits = {
            'train': (X_train, y_train),
            'val':   (X_val,   y_val),
            'near':  (X_near,  y_near),
            'far':   (X_far,   y_far),
        }
        preds = {s: cvae.predict(X) for s, (X, _) in splits.items()}
        trues = {s: y              for s, (_, y) in splits.items()}

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
                predict_fn=cvae.predict, predicted_data=pred_d
            )
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

    def _save_metrics(self, result: Dict) -> None:
        m  = result['metrics']
        pe = self.results.get('physical_evaluation')

        physics_score    = self.results.get('physics_score', float('nan'))
        boundary_score   = pe.get('boundary_score',   float('nan')) if pe else float('nan')
        smoothness_score = pe.get('smoothness_score', float('nan')) if pe else float('nan')

        rows = [
            ['Train R2',                m['train_r2']],
            ['Train RMSE',              m['train_rmse']],
            ['Train MAE',               m['train_mae']],
            ['Val R2',                  m['val_r2']],
            ['Val RMSE',                m['val_rmse']],
            ['Val MAE',                 m['val_mae']],
            ['Near-Range R2',           m['near_r2']],
            ['Near-Range RMSE',         m['near_rmse']],
            ['Near-Range MAE',          m['near_mae']],
            ['Far-Range R2',            m['far_r2']],
            ['Far-Range RMSE',          m['far_rmse']],
            ['Far-Range MAE',           m['far_mae']],
            ['Physics Score',           physics_score],
            ['Boundary Consistency',    boundary_score],
            ['Thermodynamic Smoothness', smoothness_score],
        ]
        excel_dir = self.output_dir / 'excel'
        excel_dir.mkdir(exist_ok=True)
        pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
            excel_dir / f'{self.config.excel_prefix}metrics.xlsx',
            index=False, engine='openpyxl',
        )

    def _save_predictions(
        self, result: Dict,
        X_tr: np.ndarray, X_val: np.ndarray,
        X_near: np.ndarray, X_far: np.ndarray,
    ) -> None:
        excel_dir = self.output_dir / 'excel'
        excel_dir.mkdir(exist_ok=True)
        for split, X in [('train', X_tr), ('val', X_val),
                          ('near', X_near), ('far', X_far)]:
            pd.DataFrame({
                'T/°C':              X[:, 0],
                'W(MgSO4)/%':        X[:, 1],
                'W(Na2SO4)/%_true':  result['true_values'][split],
                'W(Na2SO4)/%_pred':  result['predictions'][split],
                'residual':          result['true_values'][split] - result['predictions'][split],
            }).to_excel(
                excel_dir / f'{self.config.excel_prefix}{split}_predictions.xlsx',
                index=False, engine='openpyxl',
            )

    def _save_cvae_history(self, history: dict) -> None:
        train_eps = history.get('train', [])
        if not train_eps:
            return
        rows = [
            {
                'epoch':        ep,
                'train_total':  e.get('total',         float('nan')),
                'train_recon':  e.get('recon',         float('nan')),
                'train_kl':     e.get('kl',            float('nan')),
                'train_cycle':  e.get('cycle',         float('nan')),
                'train_Na2SO4': e.get('colloc_Na2SO4', float('nan')),
                'train_MgSO4':  e.get('colloc_MgSO4',  float('nan')),
            }
            for ep, e in enumerate(train_eps)
        ]
        lr_list = history.get('lr', [])
        df = pd.DataFrame(rows)
        if lr_list:
            df['lr'] = lr_list[:len(df)]
        excel_dir = self.output_dir / 'excel'
        excel_dir.mkdir(exist_ok=True)
        df.to_excel(
            excel_dir / f'{self.config.excel_prefix}cvae_history.xlsx',
            index=False, engine='openpyxl',
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        train_pool: Tuple,
        val_set:    Tuple,
        near_set:   Tuple,
        far_set:    Tuple,
        fraction:   float,
        sampling_idx: int,
    ) -> Dict[str, Any]:
        start = time.time()
        self.logger.info(f"  Fraction={int(fraction*100)}%  Sampling={sampling_idx}")

        X_val,  y_val  = val_set
        X_near, y_near = near_set
        X_far,  y_far  = far_set

        # Step 1: sample from pool (val set is independent and not included)
        X_sampled, y_sampled, seed = sample_from_pool(
            train_pool, fraction, sampling_idx, self.config.sampling_base_seed
        )
        self.logger.info(
            f"  Sampled: {len(X_sampled)} points  val={len(X_val)}  "
            f"near={len(X_near)}  far={len(X_far)}"
        )

        # Step 2: train PC-CVAE (X_val not passed; fixed epochs)
        cvae, history = self.train_cvae(X_sampled, y_sampled)

        # Step 3: evaluate (train / val / near / far)
        result = self.evaluate(
            cvae,
            X_sampled, y_sampled,
            X_val,  y_val,
            X_near, y_near,
            X_far,  y_far,
        )

        # Step 4: physics evaluation
        self.run_physics_eval(cvae)

        # Step 5: save
        if self.config.save_metrics:
            self._save_metrics(result)
        if self.config.save_predictions:
            self._save_predictions(result, X_sampled, X_val, X_near, X_far)
        if self.config.save_cvae_history:
            self._save_cvae_history(history)

        # Save model for best-model evaluation
        model_dir = self.output_dir / 'model'
        model_dir.mkdir(exist_ok=True)
        cvae.save(str(model_dir / 'cvae.pth'))

        return {
            'fraction':           fraction,
            'sampling_idx':       sampling_idx,
            'sampling_seed':      seed,
            'n_sampled':          len(X_sampled),
            'metrics':            result['metrics'],
            'physics_score':      self.results.get('physics_score'),
            'physics_boundary':   (self.results.get('physical_evaluation') or {}).get('boundary_score'),
            'physics_smoothness': (self.results.get('physical_evaluation') or {}).get('smoothness_score'),
            'elapsed_time':       time.time() - start,
        }


# ==============================================================================
# Experiment Manager
# ==============================================================================

class SmallSampleExperimentManager:
    def __init__(self, config: SmallSampleConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Load datasets
        self.train_pool = load_ternary_data(config.train_pool_dir / config.train_pool_file)
        self.val_set    = load_ternary_data(config.val_dir        / config.val_file)
        self.near_set   = load_ternary_data(config.extrap_dir     / config.near_extrap_file)
        self.far_set    = load_ternary_data(config.extrap_dir     / config.far_extrap_file)
        self.logger.info(
            f"Datasets loaded: train_pool={len(self.train_pool[0])}  "
            f"val={len(self.val_set[0])}  "
            f"near={len(self.near_set[0])}  far={len(self.far_set[0])}"
        )

        # Load boundary models (shared globally; not reloaded per repeat)
        self.input_boundary_model  = None
        self.output_boundary_model = None
        self._load_boundary_models()

        self.all_results: Dict[float, List] = {f: [] for f in config.sample_fractions}

    def _load_boundary_models(self) -> None:
        try:
            device = torch.device(self.config.device)
            in_p   = self.config.models_dir / self.config.input_boundary_model_file
            out_p  = self.config.models_dir / self.config.output_boundary_model_file
            if in_p.exists() and out_p.exists():
                self.input_boundary_model = move_to_device(
                    LowDimEnsemble.load(str(in_p)), device
                )
                self.output_boundary_model = move_to_device(
                    LowDimEnsemble.load(str(out_p)), device
                )
                self.logger.info(f"Boundary models loaded successfully -> {device}")
            else:
                self.logger.warning("Boundary model files not found — collocation constraints and physics evaluation disabled")
        except Exception as e:
            self.logger.error(f"Failed to load boundary models: {e}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_all_experiments(self) -> None:
        total = len(self.config.sample_fractions) * self.config.n_sampling_repeats
        self.logger.info(f"PC-CVAE small sample experiment: {total} runs")
        start = time.time()
        count = 0

        for frac in self.config.sample_fractions:
            self.logger.info(f"\n{'='*70}\nSample fraction {int(frac*100)}%\n{'='*70}")

            for idx in range(self.config.n_sampling_repeats):
                count += 1
                out_dir = (self.config.output_dir
                           / f'fraction_{int(frac*100):03d}'
                           / f'sampling_{idx:02d}')
                runner = SingleExperimentRunner(
                    self.config, out_dir,
                    self.input_boundary_model, self.output_boundary_model,
                )
                result = runner.run(
                    self.train_pool, self.val_set,
                    self.near_set, self.far_set,
                    frac, idx,
                )
                self.all_results[frac].append(result)
                self.logger.info(f"Progress: {count}/{total}")

            # After all repeats for this fraction, load best-val-R2 model for dedicated evaluation
            self._best_model_evaluation(frac)

        self._save_summary()
        self._generate_report()
        self.logger.info(
            f"\nAll experiments completed  elapsed {timedelta(seconds=int(time.time()-start))}"
            f"  Results: {self.config.output_dir}"
        )

    # ------------------------------------------------------------------
    # Best-model evaluation (select best repeat by val R2)
    # ------------------------------------------------------------------

    def _best_model_evaluation(self, fraction: float) -> None:
        """Select the best repeat by val R2, load its model, and evaluate on near/far."""
        res      = self.all_results[fraction]
        val_r2s  = [r['metrics']['val_r2'] for r in res]
        best_idx = int(np.argmax(val_r2s))
        self.logger.info(
            f"\n  [fraction={int(fraction*100)}%] Best repeat: #{best_idx}  "
            f"val_r2={val_r2s[best_idx]:.4f}"
        )

        model_path = (self.config.output_dir
                      / f'fraction_{int(fraction*100):03d}'
                      / f'sampling_{best_idx:02d}' / 'model' / 'cvae.pth')
        cvae = CVAEPhysicsModel.load(str(model_path))

        best_dir = self.config.output_dir / f'fraction_{int(fraction*100):03d}' / 'best_model'
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
                'T/°C':             X_ext[:, 0],
                'W(MgSO4)/%':       X_ext[:, 1],
                'W(Na2SO4)/%_true': y_ext,
                'W(Na2SO4)/%_pred': y_pred,
                'residual':         y_ext - y_pred,
            }).to_excel(
                best_dir / f'{self.config.excel_prefix}{tag}_predictions.xlsx',
                index=False, engine='openpyxl',
            )

        import shutil
        shutil.copy(model_path, best_dir / 'cvae_best.pth')
        self.logger.info(f"  Best model artefacts -> {best_dir}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _save_summary(self) -> None:
        summary_dir = self.config.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)

        STAT_KEYS  = ['train_r2',  'train_rmse', 'train_mae',
                      'val_r2',    'val_rmse',   'val_mae',
                      'near_r2',   'near_rmse',  'near_mae',
                      'far_r2',    'far_rmse',   'far_mae']
        STAT_NAMES = ['Train R2',       'Train RMSE',      'Train MAE',
                      'Val R2',         'Val RMSE',        'Val MAE',
                      'Near-Range R2',  'Near-Range RMSE', 'Near-Range MAE',
                      'Far-Range R2',   'Far-Range RMSE',  'Far-Range MAE']
        STAT_COLS  = ['Train_R2',  'Train_RMSE',  'Train_MAE',
                      'Val_R2',    'Val_RMSE',    'Val_MAE',
                      'Near_R2',   'Near_RMSE',   'Near_MAE',
                      'Far_R2',    'Far_RMSE',    'Far_MAE']
        PHYS_ATTRS = [
            ('physics_score',      'Physics Score',             'Physics_Score'),
            ('physics_boundary',   'Boundary Consistency',      'Boundary_Consistency'),
            ('physics_smoothness', 'Thermodynamic Smoothness',  'Thermodynamic_Smoothness'),
        ]

        wide_rows = []

        for frac in self.config.sample_fractions:
            res = self.all_results[frac]
            if not res:
                continue

            # ── Per-fraction summary_metrics.xlsx ────────────────────────────
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

            frac_label = f'fraction_{int(frac*100):03d}'
            out_path = summary_dir / frac_label / 'summary_metrics.xlsx'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(sm_rows, columns=['Metric', 'Mean Value', 'Std']).to_excel(
                out_path, index=False, engine='openpyxl',
            )
            self.logger.info(
                f"summary_metrics saved: {out_path.relative_to(self.config.output_dir)}"
            )

            # ── Wide-format row ───────────────────────────────────────────────
            wide_row = {'Sample_Fraction': f'{int(frac*100)}%'}
            for key, col in zip(STAT_KEYS, STAT_COLS):
                vals = [r['metrics'][key] for r in res]
                wide_row[f'{col}_Mean'] = float(np.mean(vals))
                wide_row[f'{col}_Std']  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            for attr, _, col in PHYS_ATTRS:
                vals_clean = [r.get(attr) for r in res
                              if r.get(attr) is not None
                              and not np.isnan(float(r.get(attr, float('nan'))))]
                wide_row[f'{col}_Mean'] = float(np.mean(vals_clean)) if vals_clean else float('nan')
                wide_row[f'{col}_Std']  = (float(np.std(vals_clean, ddof=1))
                                           if len(vals_clean) > 1 else float('nan'))
            wide_rows.append(wide_row)

        pd.DataFrame(wide_rows).to_excel(
            summary_dir / 'overall_metrics.xlsx', index=False, engine='openpyxl',
        )

    def _generate_report(self) -> None:
        summary_dir = self.config.output_dir / 'summary'
        c = self.config.cvae_config
        lines = [
            '=' * 70,
            'PC-CVAE Small Sample Sensitivity Study - Summary Report',
            '=' * 70,
            f"Generated:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Sample fractions: {[f'{int(f*100)}%' for f in self.config.sample_fractions]}",
            f"Repeats/fraction: {self.config.n_sampling_repeats}",
            f"CVAE epochs:      {c.N_EPOCHS} (fixed, no early stopping)",
            f"lambda_KL:        {c.LAMBDA_KL}",
            f"lambda_Na2SO4:    {c.LAMBDA_COLLOCATION_Na2SO4}",
            f"lambda_MgSO4:     {c.LAMBDA_COLLOCATION_MgSO4}",
            f"Collocation T:    {c.COLLOCATION_T_RANGE}",
            f"Train pool:       interpolation domain (T <= 50°C)",
            f"Val set:          fixed_splits/val_set.xlsx (fixed, not sampled)",
            f"Near-range:       50 < T < 100°C",
            f"Far-range:        T >= 100°C",
            '', '-' * 70, 'Results by fraction (mean +/- std over repeats)', '-' * 70,
            f"{'Fraction':>10}  {'Val R2':>24}  {'Near R2':>24}  {'Far R2':>24}",
            '-' * 70,
        ]
        for frac in self.config.sample_fractions:
            res = self.all_results[frac]
            if not res:
                continue
            vr = [r['metrics']['val_r2']  for r in res]
            nr = [r['metrics']['near_r2'] for r in res]
            fr = [r['metrics']['far_r2']  for r in res]
            lines.append(
                f"{int(frac*100):3d}% ({res[0]['n_sampled']:3d} pts):  "
                f"Val R2  = {np.mean(vr):.6f} +/- {np.std(vr,  ddof=1):.6f}  "
                f"Near R2 = {np.mean(nr):.6f} +/- {np.std(nr, ddof=1):.6f}  "
                f"Far R2  = {np.mean(fr):.6f} +/- {np.std(fr, ddof=1):.6f}"
            )
        lines.append('=' * 70)
        report = '\n'.join(lines)
        (summary_dir / 'summary_report.txt').write_text(report, encoding='utf-8')
        self.logger.info(f'\n{report}')


# ==============================================================================
# Main
# ==============================================================================

def main():
    config = SmallSampleConfig()
    config.sample_fractions        = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    config.n_sampling_repeats      = 10
    config.sampling_base_seed      = 10000
    config.boundary_decay_lambda   = 5.0
    config.smoothness_decay_lambda = 15.0
    config.save_predictions        = True
    config.save_metrics            = True
    config.save_cvae_history       = True
    config.log_level               = logging.INFO

    config.cvae_config.LATENT_DIM                = 1
    config.cvae_config.N_EPOCHS                  = 500
    config.cvae_config.BATCH_SIZE                = 64
    config.cvae_config.LEARNING_RATE             = 1e-3
    config.cvae_config.LAMBDA_KL                 = 0.001
    config.cvae_config.LAMBDA_COLLOCATION_Na2SO4 = 1
    config.cvae_config.LAMBDA_COLLOCATION_MgSO4  = 1
    config.cvae_config.N_COLLOCATION_POINTS      = 64
    config.cvae_config.COLLOCATION_T_RANGE        = (-10.0, 200.0)
    config.cvae_config.Z_LOW                     = -2.0
    config.cvae_config.Z_HIGH                    = 2.0
    config.cvae_config.Z_COLLOC_WIDTH            = 0.5
    config.cvae_config.PHI_HIDDEN_DIMS           = [64, 64]
    config.cvae_config.LAMBDA_CYCLE              = 1.0
    config.cvae_config.N_CYCLE_POINTS            = 64
    config.cvae_config.CYCLE_T_RANGE             = (-10.0, 200.0)
    config.cvae_config.USE_EARLY_STOPPING        = False
    config.cvae_config.DEVICE                    = config.device

    manager = SmallSampleExperimentManager(config)
    manager.run_all_experiments()


if __name__ == '__main__':
    main()