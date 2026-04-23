"""
================================================================================
PC-CVAE Ablation Study - K-Fold Cross-Validation Framework (Viscosity Ternary System)
================================================================================

Fully symmetric with the solubility version; the three core differences are:
  1. Three boundary models (MCH-HMN / cis-Decalin-HMN / MCH-cis-Decalin).
  2. Input dimension 4 (T, P, MCH, Dec); history contains colloc_mch/dec/hmn/cycle.
  3. Physics evaluation is called via ViscosityPhysicsEvaluator.evaluate_full(trainer),
     where trainer must implement the predict(X, return_original_scale=True) interface;
     _CVAEWrapper lightly adapts CVAEPhysicsModel.predict(X) to this interface with
     zero intrusion.

Pipeline:
    1. PC-CVAE Training  → learns the viscosity manifold, with triangular collocation
                           constraints and φ + L_cycle
    2. Direct Evaluation → deterministic prediction via the φ head, cvae.predict(X)
    3. Physics Eval      → ViscosityPhysicsEvaluator dual-pillar evaluation

Design principles:
  - X_val is not passed to fit(); N_EPOCHS is fixed, eliminating fold-to-fold
    variance introduced by early stopping.
  - Three LowDimEnsemble models are shared across all folds and loaded only once.
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pc_cvae_viscosity import CVAEConfig, CVAEPhysicsModel, LowDimInfo
from low_dim_model import LowDimEnsemble
from utils_viscosity import ViscosityPhysicsEvaluator

warnings.filterwarnings('ignore')


# ==============================================================================
# Utility functions
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


def load_viscosity_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load viscosity data file.

    Column order convention: [T, P, MCH, Dec, Visc]
    X: (N, 4) [T, P, MCH, Dec]
    y: (N, 1) Visc
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    data = pd.read_excel(filepath, engine='openpyxl')
    if data.shape[1] < 5:
        raise ValueError(f"Expected at least 5 columns [T, P, MCH, Dec, Visc], got: {data.shape[1]}")
    X = data.iloc[:, :4].values.astype(np.float32)
    y = data.iloc[:, 4].values.astype(np.float32)
    return X, y


def move_to_device(model: LowDimEnsemble, device: torch.device) -> LowDimEnsemble:
    """Move LowDimEnsemble to the target device and sync the .device attribute."""
    model.to(device)
    model.device = device
    return model


def create_k_folds(
    X: np.ndarray, y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dict[str, np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [
        {
            'fold_idx': fi,
            'X_train': X[tr], 'y_train': y[tr],
            'X_val':   X[va], 'y_val':   y[va],
        }
        for fi, (tr, va) in enumerate(kf.split(X))
    ]


# ==============================================================================
# ViscosityPhysicsEvaluator adapter wrapper
# ==============================================================================

class _CVAEWrapper:
    """Lightweight wrapper: adapts CVAEPhysicsModel.predict(X) to the
    trainer.predict(X, return_original_scale=True) interface required by
    ViscosityPhysicsEvaluator.evaluate_full(). Zero modification to CVAEPhysicsModel.
    """

    def __init__(self, cvae: CVAEPhysicsModel) -> None:
        self._cvae = cvae

    def predict(self, X: np.ndarray, return_original_scale: bool = True) -> np.ndarray:
        """(N, 4) [T, P, MCH, Dec] → (N, 1) Visc; always returns original scale."""
        return self._cvae.predict(X)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class CVAEExperimentConfig:
    """PC-CVAE viscosity ablation experiment configuration."""

    # Data paths
    data_dir: Path = PROJECT_ROOT / 'data' / 'viscosity' / 'split_by_temperature'
    train_data_file:  str = 'interpolation domain.xlsx'
    near_extrap_file: str = 'near-range extrapolation.xlsx'
    far_extrap_file:  str = 'far-range extrapolation.xlsx'

    # Boundary model paths
    models_dir: Path = PROJECT_ROOT / 'models' / 'Low_dim_model' / 'viscosity'
    mch_hmn_model_file: str = 'MCH_HMN.pth'
    mch_dec_model_file: str = 'MCH_cis_Decalin.pth'
    dec_hmn_model_file: str = 'cis_Decalin_HMN.pth'

    # PC-CVAE hyperparameters (fixed epochs, early stopping disabled for fold comparability)
    cvae_config: CVAEConfig = field(default_factory=lambda: CVAEConfig(
        LATENT_DIM=2,
        HIDDEN_DIMS=[128, 256, 256, 128],
        DROPOUT=0.1,
        LEARNING_RATE=1e-3,
        BATCH_SIZE=64,
        N_EPOCHS=500,
        WEIGHT_DECAY=1e-5,
        LAMBDA_KL=0.001,
        LAMBDA_COLLOCATION_MCH=1.0,
        LAMBDA_COLLOCATION_DEC=1.0,
        LAMBDA_COLLOCATION_HMN=1.0,
        N_COLLOCATION_POINTS=64,
        COLLOCATION_T_RANGE=(20.0, 80.0),
        COLLOCATION_P_RANGE=(1e5, 1e8),
        Z_LOW=-2.0,
        Z_HIGH=2.0,
        Z_COLLOC_WIDTH=0.5,
        PHI_HIDDEN_DIMS=[64, 64],
        LAMBDA_CYCLE=1.0,
        N_CYCLE_POINTS=64,
        CYCLE_T_RANGE=(20.0, 80.0),    # must explicitly cover the high-temperature extrapolation region
        CYCLE_P_RANGE=(1e5, 1e8),      # must explicitly cover the high-pressure extrapolation region
        USE_EARLY_STOPPING=False,
        USE_LR_SCHEDULER=True,
        LR_SCHEDULER_TYPE='cosine',
        LR_MIN=1e-5,
        DEVICE='auto',
        VERBOSE=False,
    ))

    # Physics evaluation range (consistent with training/test data)
    t_min: float = 20.0
    t_max: float = 80.0
    p_min: float = 1e5
    p_max: float = 1e8

    # Cross-validation
    k_folds: int = 5
    kfold_random_state: int = 42

    # Output
    output_dir: Path = (
        PROJECT_ROOT / 'results' / 'viscosity' / 'ablation' / 'CVAE_results'
    )
    save_predictions:  bool = True
    save_metrics:      bool = True
    save_cvae_history: bool = True
    excel_prefix:      str  = 'cvae_'

    # Device and logging
    device:    str = 'auto'
    log_level: int = logging.INFO

    def __post_init__(self) -> None:
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cvae_config.DEVICE = self.device
        # Assemble full paths
        self.train_data_path  = self.data_dir   / self.train_data_file
        self.near_extrap_path = self.data_dir   / self.near_extrap_file
        self.far_extrap_path  = self.data_dir   / self.far_extrap_file
        self.mch_hmn_path    = self.models_dir / self.mch_hmn_model_file
        self.mch_dec_path    = self.models_dir / self.mch_dec_model_file
        self.dec_hmn_path    = self.models_dir / self.dec_hmn_model_file


# ==============================================================================
# Single-fold runner
# ==============================================================================

class SingleFoldRunner:
    """Executes the full PC-CVAE pipeline for a single K-fold split."""

    def __init__(
        self,
        config: CVAEExperimentConfig,
        model_mch_hmn: Optional[LowDimEnsemble],
        model_mch_dec: Optional[LowDimEnsemble],
        model_dec_hmn: Optional[LowDimEnsemble],
    ) -> None:
        self.config        = config
        self.logger        = get_logger(self.__class__.__name__, config.log_level)
        self.device        = torch.device(config.device)
        self.model_mch_hmn = model_mch_hmn
        self.model_mch_dec = model_mch_dec
        self.model_dec_hmn = model_dec_hmn

    def run(
        self,
        fold_idx: int,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        X_near:  np.ndarray, y_near:  np.ndarray,
        X_far:   np.ndarray, y_far:   np.ndarray,
        fold_dir: Path,
    ) -> Dict[str, Any]:
        """Three-step pipeline: training → evaluation → physical consistency."""
        self.logger.info("=" * 70)
        self.logger.info(f"Fold {fold_idx}")
        self.logger.info("=" * 70)

        self.logger.info("[Step 1] Training PC-CVAE (viscosity version, fixed epochs)")
        cvae, cvae_history = self._train_cvae(X_train, y_train)

        self.logger.info("[Step 2] Direct evaluation via φ head")
        metrics = self._evaluate_cvae(
            cvae, X_train, y_train, X_val, y_val, X_near, y_near, X_far, y_far,
        )

        self.logger.info("[Step 3] Physics consistency evaluation (dual-pillar framework)")
        physics_results = self._compute_physics_metrics(cvae)

        self._save_fold_results(fold_idx, fold_dir, metrics, physics_results, cvae_history, cvae)

        return {
            'fold_idx':           fold_idx,
            'metrics':            metrics,
            'physics_score':      physics_results.get('physics_score',      None),
            'physics_boundary':   physics_results.get('physics_boundary',   None),
            'physics_smoothness': physics_results.get('physics_smoothness', None),
        }

    # ------------------------------------------------------------------
    # Step 1: Training
    # ------------------------------------------------------------------

    def _train_cvae(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[CVAEPhysicsModel, dict]:
        """Build the LowDimInfo list and train CVAEPhysicsModel.

        Boundary type mapping:
          mch_zero — x_MCH=0 boundary, using dec_hmn model (cis-Decalin-HMN system)
          dec_zero — x_Dec=0 boundary, using mch_hmn model (MCH-HMN system)
          hmn_zero — x_HMN=0 boundary, using mch_dec model (MCH-cis-Decalin system)
        """
        low_dim_list: Optional[List[LowDimInfo]] = None

        if all(m is not None for m in (
            self.model_mch_hmn, self.model_mch_dec, self.model_dec_hmn
        )):
            low_dim_list = [
                LowDimInfo(
                    model=self.model_dec_hmn,
                    name='cis_Decalin_HMN',
                    boundary_type='mch_zero',
                ),
                LowDimInfo(
                    model=self.model_mch_hmn,
                    name='MCH_HMN',
                    boundary_type='dec_zero',
                ),
                LowDimInfo(
                    model=self.model_mch_dec,
                    name='MCH_cis_Decalin',
                    boundary_type='hmn_zero',
                ),
            ]
        else:
            self.logger.warning("Boundary models missing — collocation constraint disabled")

        cvae = CVAEPhysicsModel(config=self.config.cvae_config)

        history = cvae.fit(
            X=X_train,
            y=y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train,
            low_dim_list=low_dim_list,
            X_val=None,
            y_val=None,
        )

        tr = history.get('train_loss', [])
        if tr:
            self.logger.info(
                f"  Training done — total={tr[-1]:.5f}  "
                f"recon={history['train_recon'][-1]:.5f}  "
                f"kl={history['train_kl'][-1]:.5f}  "
                f"cycle={history['train_cycle'][-1]:.5f}  "
                f"colloc_mch={history['train_colloc_mch'][-1]:.5f}  "
                f"colloc_dec={history['train_colloc_dec'][-1]:.5f}  "
                f"colloc_hmn={history['train_colloc_hmn'][-1]:.5f}"
            )
        return cvae, history

    # ------------------------------------------------------------------
    # Step 2: Evaluation
    # ------------------------------------------------------------------

    def _evaluate_cvae(
        self,
        cvae: CVAEPhysicsModel,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        X_near:  np.ndarray, y_near:  np.ndarray,
        X_far:   np.ndarray, y_far:   np.ndarray,
    ) -> Dict[str, Any]:
        """Compute R²/RMSE/MAE on train/val/near/far splits using cvae.predict(X).

        cvae.predict(X) accepts (N, 4) [T, P, MCH, Dec] and returns (N, 1) Visc.
        """
        splits = {
            'train': (X_train, y_train),
            'val':   (X_val,   y_val),
            'near':  (X_near,  y_near),
            'far':   (X_far,   y_far),
        }

        preds = {s: cvae.predict(X).flatten() for s, (X, _) in splits.items()}
        trues = {s: y.flatten() for s, (_, y) in splits.items()}

        metrics_result: Dict[str, float] = {}
        for s in ('train', 'val', 'near', 'far'):
            metrics_result[f'{s}_r2']   = float(r2_score(trues[s], preds[s]))
            metrics_result[f'{s}_rmse'] = float(np.sqrt(mean_squared_error(trues[s], preds[s])))
            metrics_result[f'{s}_mae']  = float(mean_absolute_error(trues[s], preds[s]))

        self.logger.info(
            f"  train_r²={metrics_result['train_r2']:.4f}  "
            f"val_r²={metrics_result['val_r2']:.4f}  "
            f"near_r²={metrics_result['near_r2']:.4f}  "
            f"far_r²={metrics_result['far_r2']:.4f}"
        )

        return {
            'metrics':     metrics_result,
            'predictions': preds,
            'true_values': trues,
        }

    # ------------------------------------------------------------------
    # Step 3: Physical consistency
    # ------------------------------------------------------------------

    def _compute_physics_metrics(self, cvae: CVAEPhysicsModel) -> Dict[str, float]:
        """Evaluate physical consistency using the ViscosityPhysicsEvaluator dual-pillar framework.

        ViscosityPhysicsEvaluator.evaluate_full() requires trainer to implement
        the predict(X, return_original_scale=True) interface.
        _CVAEWrapper adapts cvae.predict(X) to this interface with zero intrusion.
        """
        if any(m is None for m in (
            self.model_mch_hmn, self.model_mch_dec, self.model_dec_hmn
        )):
            self.logger.warning("Boundary models missing — skipping physics evaluation")
            return {}

        try:
            evaluator = ViscosityPhysicsEvaluator(
                teacher_models=(
                    self.model_mch_hmn,
                    self.model_dec_hmn,
                    self.model_mch_dec,
                ),
                temp_range=(self.config.t_min, self.config.t_max),
                pressure_range=(self.config.p_min, self.config.p_max),
            )

            wrapper = _CVAEWrapper(cvae)
            overall_score, results = evaluator.evaluate_full(wrapper)

            boundary_score   = results.get('boundary_score',   float('nan'))
            smoothness_score = results.get('smoothness_score', float('nan'))

            self.logger.info(
                f"  physics={overall_score:.4f}  "
                f"boundary={boundary_score:.4f}  "
                f"smoothness={smoothness_score:.4f}"
            )
            return {
                'physics_score':      float(overall_score),
                'physics_boundary':   float(boundary_score),
                'physics_smoothness': float(smoothness_score),
            }

        except Exception as e:
            import traceback
            self.logger.error(f"Physics evaluation failed: {e}\n{traceback.format_exc()}")
            return {}

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_fold_results(
        self,
        fold_idx:        int,
        fold_dir:        Path,
        metrics:         Dict,
        physics_results: Dict,
        cvae_history:    dict,
        cvae:            CVAEPhysicsModel,
    ) -> None:
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights (for best-fold evaluation loading)
        model_dir = fold_dir / 'model'
        model_dir.mkdir(exist_ok=True)
        cvae.save(str(model_dir / 'cvae.pth'))

        if self.config.save_metrics:
            self._save_fold_metrics(metrics, physics_results, fold_dir)
        if self.config.save_predictions:
            self._save_predictions(metrics, fold_dir)
        if self.config.save_cvae_history:
            self._save_cvae_history(cvae_history, fold_dir)

    def _save_fold_metrics(
        self, metrics: Dict, physics_results: Dict, fold_dir: Path
    ) -> None:
        inner = metrics.get('metrics', metrics)
        pe    = physics_results

        physics_score    = pe.get('physics_score',      float('nan')) if pe else float('nan')
        boundary_score   = pe.get('physics_boundary',   float('nan')) if pe else float('nan')
        smoothness_score = pe.get('physics_smoothness', float('nan')) if pe else float('nan')

        rows = [
            ['Train R²',                 inner.get('train_r2',   float('nan'))],
            ['Train RMSE',               inner.get('train_rmse', float('nan'))],
            ['Train MAE',                inner.get('train_mae',  float('nan'))],
            ['Val R²',                   inner.get('val_r2',     float('nan'))],
            ['Val RMSE',                 inner.get('val_rmse',   float('nan'))],
            ['Val MAE',                  inner.get('val_mae',    float('nan'))],
            ['Near-Range R²',            inner.get('near_r2',    float('nan'))],
            ['Near-Range RMSE',          inner.get('near_rmse',  float('nan'))],
            ['Near-Range MAE',           inner.get('near_mae',   float('nan'))],
            ['Far-Range R²',             inner.get('far_r2',     float('nan'))],
            ['Far-Range RMSE',           inner.get('far_rmse',   float('nan'))],
            ['Far-Range MAE',            inner.get('far_mae',    float('nan'))],
            ['Physics Score',            physics_score],
            ['Boundary Consistency',     boundary_score],
            ['Thermodynamic Smoothness', smoothness_score],
        ]
        pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
            fold_dir / f'{self.config.excel_prefix}metrics.xlsx',
            index=False, engine='openpyxl',
        )

    def _save_predictions(self, metrics: Dict, fold_dir: Path) -> None:
        predictions = metrics.get('predictions', {})
        true_values = metrics.get('true_values', {})
        for split in ('train', 'val', 'near', 'far'):
            if split not in predictions:
                continue
            y_true = true_values[split].flatten()
            y_pred = predictions[split].flatten()
            pd.DataFrame({
                'y_true': y_true, 'y_pred': y_pred, 'residual': y_true - y_pred,
            }).to_excel(
                fold_dir / f'{self.config.excel_prefix}{split}_predictions.xlsx',
                index=False, engine='openpyxl',
            )

    def _save_cvae_history(self, history: dict, fold_dir: Path) -> None:
        """Save per-epoch loss details.

        Columns: epoch, total, recon, kl, colloc_mch, colloc_dec, colloc_hmn, cycle, lr
        """
        train_loss = history.get('train_loss', [])
        if not train_loss:
            return

        rows = []
        for ep, total in enumerate(train_loss):
            rows.append({
                'epoch':           ep,
                'train_total':     total,
                'train_recon':     history['train_recon'][ep]     if ep < len(history.get('train_recon', [])) else float('nan'),
                'train_kl':        history['train_kl'][ep]        if ep < len(history.get('train_kl', [])) else float('nan'),
                'train_cycle':     history['train_cycle'][ep]     if ep < len(history.get('train_cycle', [])) else float('nan'),
                'train_colloc_mch': history['train_colloc_mch'][ep] if ep < len(history.get('train_colloc_mch', [])) else float('nan'),
                'train_colloc_dec': history['train_colloc_dec'][ep] if ep < len(history.get('train_colloc_dec', [])) else float('nan'),
                'train_colloc_hmn': history['train_colloc_hmn'][ep] if ep < len(history.get('train_colloc_hmn', [])) else float('nan'),
            })

        df = pd.DataFrame(rows)

        val_loss = history.get('val_loss', [])
        if val_loss:
            df['val_total'] = pd.Series(val_loss)

        df.to_excel(
            fold_dir / f'{self.config.excel_prefix}cvae_history.xlsx',
            index=False, engine='openpyxl',
        )


# ==============================================================================
# K-Fold experiment manager
# ==============================================================================

class KFoldExperimentManager:
    """Orchestrates training, evaluation, and summary report generation across all folds."""

    def __init__(self, config: CVAEExperimentConfig, n_folds: int = 5) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got: {n_folds}")
        self.config      = config
        self.n_folds     = n_folds
        self.logger      = get_logger(self.__class__.__name__, level=config.log_level)
        self.output_dir  = config.output_dir
        self.all_results: List[Dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_mch_hmn: Optional[LowDimEnsemble] = None
        self.model_mch_dec: Optional[LowDimEnsemble] = None
        self.model_dec_hmn: Optional[LowDimEnsemble] = None
        self._load_boundary_models()

    def _load_boundary_models(self) -> None:
        """Load three low-dimensional binary system models (shared across all folds)."""
        try:
            device = torch.device(self.config.device)
            paths = {
                'mch_hmn': self.config.mch_hmn_path,
                'mch_dec': self.config.mch_dec_path,
                'dec_hmn': self.config.dec_hmn_path,
            }
            missing = [k for k, p in paths.items() if not p.exists()]
            if missing:
                self.logger.warning(
                    f"Boundary model files missing: {missing} — collocation constraint and physics evaluation will be disabled"
                )
                return

            self.model_mch_hmn = move_to_device(LowDimEnsemble.load(str(paths['mch_hmn'])), device)
            self.model_mch_dec = move_to_device(LowDimEnsemble.load(str(paths['mch_dec'])), device)
            self.model_dec_hmn = move_to_device(LowDimEnsemble.load(str(paths['dec_hmn'])), device)
            self.logger.info(f"Three boundary models loaded → {device}")

        except Exception as e:
            self.logger.error(f"Failed to load boundary models: {e}")

    def run_all_folds(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info("PC-CVAE Ablation Study — K-Fold CV (Viscosity Ternary System, direct φ prediction)")
        self.logger.info("=" * 70)
        self.logger.info(f"K-Folds: {self.n_folds}  Device: {self.config.device}")

        cfg = self.config.cvae_config
        self.logger.info(
            f"CVAE: latent_dim={cfg.LATENT_DIM}  epochs={cfg.N_EPOCHS}  "
            f"λ_KL={cfg.LAMBDA_KL}  "
            f"λ_MCH={cfg.LAMBDA_COLLOCATION_MCH}  "
            f"λ_Dec={cfg.LAMBDA_COLLOCATION_DEC}  "
            f"λ_HMN={cfg.LAMBDA_COLLOCATION_HMN}  "
            f"λ_cycle={cfg.LAMBDA_CYCLE}"
        )
        self.logger.info(
            f"Z range: [{cfg.Z_LOW}, {cfg.Z_HIGH}]  "
            f"colloc width: {cfg.Z_COLLOC_WIDTH}  "
            f"n_colloc: {cfg.N_COLLOCATION_POINTS}"
        )
        self.logger.info(
            f"φ hidden: {cfg.PHI_HIDDEN_DIMS}  "
            f"n_cycle: {cfg.N_CYCLE_POINTS}  "
            f"cycle_T: {cfg.CYCLE_T_RANGE}  "
            f"cycle_P: {cfg.CYCLE_P_RANGE}"
        )
        self.logger.info(
            f"Physics eval T=[{self.config.t_min},{self.config.t_max}]  "
            f"P=[{self.config.p_min:.0e},{self.config.p_max:.0e}]"
        )

        start = time.time()

        X_train_pool, y_train_pool = load_viscosity_data(self.config.train_data_path)
        X_near,       y_near       = load_viscosity_data(self.config.near_extrap_path)
        X_far,        y_far        = load_viscosity_data(self.config.far_extrap_path)
        self.logger.info(
            f"Train pool: {len(X_train_pool)}  near-range: {len(X_near)}  far-range: {len(X_far)}"
        )
        self._near_data = (X_near, y_near)
        self._far_data  = (X_far,  y_far)

        folds = create_k_folds(
            X_train_pool, y_train_pool,
            n_splits=self.n_folds,
            random_state=self.config.kfold_random_state,
        )
        for f in folds:
            self.logger.info(
                f"  Fold {f['fold_idx']}: train={len(f['X_train'])}  val={len(f['X_val'])}"
            )

        for fold_data in folds:
            fold_idx = fold_data['fold_idx']
            self.logger.info(f"\n{'█' * 70}\nFold {fold_idx + 1}/{self.n_folds}\n{'█' * 70}")
            fold_dir = self.output_dir / f'fold_{fold_idx}'
            fold_dir.mkdir(exist_ok=True)

            X_near, y_near = self._near_data
            X_far,  y_far  = self._far_data
            runner = SingleFoldRunner(
                config=self.config,
                model_mch_hmn=self.model_mch_hmn,
                model_mch_dec=self.model_mch_dec,
                model_dec_hmn=self.model_dec_hmn,
            )
            result = runner.run(
                fold_idx=fold_idx,
                X_train=fold_data['X_train'], y_train=fold_data['y_train'],
                X_val=fold_data['X_val'],     y_val=fold_data['y_val'],
                X_near=X_near,                y_near=y_near,
                X_far=X_far,                  y_far=y_far,
                fold_dir=fold_dir,
            )
            self.all_results.append(result)

        self._generate_summary()
        self._best_model_evaluation()
        self.logger.info(
            f"\nAll folds complete, elapsed {timedelta(seconds=int(time.time() - start))}"
            f"  Results path: {self.output_dir}"
        )

    def _generate_summary(self) -> None:
        summary_dir = self.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)
        self._save_summary_metrics(summary_dir)
        self._save_summary_predictions(summary_dir)
        self._generate_text_report(summary_dir)

    def _save_summary_metrics(self, summary_dir: Path) -> None:
        """Save cross-fold summary metrics.

        Output columns: Metric | Mean Value | Std
        Fixed 15 rows (12 statistics + 3 physics); physics fields filled with NaN
        when unavailable; Std uses ddof=1.
        """
        STAT_KEYS  = ['train_r2',  'train_rmse', 'train_mae',
                      'val_r2',    'val_rmse',   'val_mae',
                      'near_r2',   'near_rmse',  'near_mae',
                      'far_r2',    'far_rmse',   'far_mae']
        STAT_NAMES = ['Train R²',  'Train RMSE', 'Train MAE',
                      'Val R²',    'Val RMSE',   'Val MAE',
                      'Near-Range R²',   'Near-Range RMSE', 'Near-Range MAE',
                      'Far-Range R²',    'Far-Range RMSE',  'Far-Range MAE']
        PHYS_ATTRS = [
            ('physics_score',      'Physics Score'),
            ('physics_boundary',   'Boundary Consistency'),
            ('physics_smoothness', 'Thermodynamic Smoothness'),
        ]

        rows: List[Dict[str, Any]] = []

        for key, name in zip(STAT_KEYS, STAT_NAMES):
            vals = [r['metrics']['metrics'][key] for r in self.all_results]
            rows.append({
                'Metric':     name,
                'Mean Value': float(np.mean(vals)),
                'Std':        float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })

        for attr, label in PHYS_ATTRS:
            vals = [
                r.get(attr) for r in self.all_results
                if r.get(attr) is not None
                and not np.isnan(float(r.get(attr, float('nan'))))
            ]
            if vals:
                mean_val = float(np.mean(vals))
                std_val  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                mean_val = float('nan')
                std_val  = float('nan')
            rows.append({'Metric': label, 'Mean Value': mean_val, 'Std': std_val})

        pd.DataFrame(rows, columns=['Metric', 'Mean Value', 'Std']).to_excel(
            summary_dir / 'summary_metrics.xlsx', index=False, engine='openpyxl'
        )

    def _save_summary_predictions(self, summary_dir: Path) -> None:
        for tag, data_attr in [('near', '_near_data'), ('far', '_far_data')]:
            y_true = getattr(self, data_attr)[1].flatten()
            data   = {'y_true': y_true}
            for r in self.all_results:
                data[f"y_pred_fold{r['fold_idx']}"] = (
                    r['metrics']['predictions'][tag].flatten()
                )
            preds = np.array([
                r['metrics']['predictions'][tag].flatten() for r in self.all_results
            ])
            data['y_pred_mean']   = np.mean(preds, axis=0)
            data['y_pred_std']    = np.std(preds,  axis=0, ddof=1)
            data['residual_mean'] = y_true - data['y_pred_mean']
            pd.DataFrame(data).to_excel(
                summary_dir / f'{tag}_predictions_summary.xlsx',
                index=False, engine='openpyxl',
            )

    def _generate_text_report(self, summary_dir: Path) -> None:
        cfg = self.config.cvae_config
        sep = '-' * 70
        lines = [
            '=' * 70,
            'PC-CVAE Ablation Study — K-Fold CV Summary (Viscosity Ternary System, direct φ prediction)',
            '=' * 70,
            f"Generated:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"K-Folds:     {self.n_folds}",
            f"Device:      {self.config.device}",
            '', sep, 'PC-CVAE Configuration', sep,
            f"Latent dim:           {cfg.LATENT_DIM}",
            f"Hidden dims:          {cfg.HIDDEN_DIMS}",
            f"N_EPOCHS:             {cfg.N_EPOCHS}  (fixed, no early stopping)",
            f"λ_KL:                 {cfg.LAMBDA_KL}",
            f"λ_MCH:                {cfg.LAMBDA_COLLOCATION_MCH}",
            f"λ_Dec:                {cfg.LAMBDA_COLLOCATION_DEC}",
            f"λ_HMN:                {cfg.LAMBDA_COLLOCATION_HMN}",
            f"N_collocation:        {cfg.N_COLLOCATION_POINTS}",
            f"Collocation T:        {cfg.COLLOCATION_T_RANGE}",
            f"Collocation P:        {cfg.COLLOCATION_P_RANGE}",
            f"Z range:              [{cfg.Z_LOW}, {cfg.Z_HIGH}]",
            f"Z colloc width:       {cfg.Z_COLLOC_WIDTH}",
            '', sep, 'φ head (Inverse Manifold Mapping)', sep,
            f"φ hidden dims:        {cfg.PHI_HIDDEN_DIMS}",
            f"λ_cycle:              {cfg.LAMBDA_CYCLE}",
            f"N_cycle_points:       {cfg.N_CYCLE_POINTS}",
            f"Cycle T range:        {cfg.CYCLE_T_RANGE}",
            f"Cycle P range:        {cfg.CYCLE_P_RANGE}",
            '', sep, 'Physics Evaluation Range', sep,
            f"T range:              [{self.config.t_min}, {self.config.t_max}]",
            f"P range:              [{self.config.p_min:.0e}, {self.config.p_max:.0e}]",
            '', sep, 'Summary Statistics', sep,
        ]

        keys  = ['train_r2', 'train_rmse', 'train_mae',
                 'val_r2',   'val_rmse',   'val_mae',
                 'near_r2',  'near_rmse',  'near_mae',
                 'far_r2',   'far_rmse',   'far_mae']
        names = ['Train R²', 'Train RMSE', 'Train MAE',
                 'Val R²',   'Val RMSE',   'Val MAE',
                 'Near-Range R²',   'Near-Range RMSE', 'Near-Range MAE',
                 'Far-Range R²',    'Far-Range RMSE',  'Far-Range MAE']
        for k, name in zip(keys, names):
            vals = [r['metrics']['metrics'][k] for r in self.all_results]
            lines.append(f"{name:22s}: {np.mean(vals):.6f} ± {np.std(vals, ddof=1):.6f}")

        lines.append('\nPhysics Evaluation:')
        for attr, name in [
            ('physics_score',      'Physics Score'),
            ('physics_boundary',   'Boundary Consistency'),
            ('physics_smoothness', 'Thermodynamic Smoothness'),
        ]:
            vals = [
                r.get(attr) for r in self.all_results
                if r.get(attr) is not None and not np.isnan(float(r.get(attr, float('nan'))))
            ]
            if vals:
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                lines.append(f"{name:22s}: {np.mean(vals):.6f} ± {std:.6f}")
            else:
                lines.append(f"{name:22s}: N/A")

        val_r2 = [r['metrics']['metrics']['val_r2'] for r in self.all_results]
        best   = self.all_results[int(np.argmax(val_r2))]
        lines += [
            '', sep, 'Best Fold (by Val R²)', sep,
            f"Fold {best['fold_idx']}",
            f"  Val R²:        {best['metrics']['metrics']['val_r2']:.6f}",
            f"  Near-Range R²: {best['metrics']['metrics']['near_r2']:.6f}",
            f"  Far-Range R²:  {best['metrics']['metrics']['far_r2']:.6f}",
            '', sep, 'Stability (Val R²)', sep,
            f"Std:  {np.std(val_r2, ddof=1):.6f}",
            f"CV:   {np.std(val_r2, ddof=1) / max(abs(np.mean(val_r2)), 1e-8) * 100:.2f}%",
            '', '=' * 70,
        ]

        report = '\n'.join(lines)
        (summary_dir / 'summary_report.txt').write_text(report, encoding='utf-8')
        self.logger.info(f'\n{report}')

    def _best_model_evaluation(self) -> None:
        """After all folds complete, select the best fold by Val R², load the CVAE,
        and independently evaluate on the near/far domains.

        Output written to output_dir/best_model/:
          - best_near_predictions.xlsx
          - best_far_predictions.xlsx
          - best_model_metrics.xlsx
        """
        val_r2_list = [r['metrics']['metrics']['val_r2'] for r in self.all_results]
        best_idx    = int(np.argmax(val_r2_list))
        best_result = self.all_results[best_idx]
        fold_idx    = best_result['fold_idx']

        self.logger.info(
            f"\nBest fold: Fold {fold_idx}  Val R²={val_r2_list[best_idx]:.6f}"
        )

        model_path = self.output_dir / f'fold_{fold_idx}' / 'model' / 'cvae.pth'
        if not model_path.exists():
            self.logger.error(f"Best fold model file not found, skipping best model evaluation: {model_path}")
            return

        cvae     = CVAEPhysicsModel.load(str(model_path))
        best_dir = self.output_dir / 'best_model'
        best_dir.mkdir(exist_ok=True)

        X_near, y_near = self._near_data
        X_far,  y_far  = self._far_data

        summary_rows = [
            ['Best Fold', fold_idx],
            ['Val R²',    val_r2_list[best_idx]],
        ]

        for tag, X, y_true_arr in [('near', X_near, y_near), ('far', X_far, y_far)]:
            y_pred = cvae.predict(X).flatten()
            y_true = y_true_arr.flatten()
            r2   = float(r2_score(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae  = float(mean_absolute_error(y_true, y_pred))
            label = 'Near-Range' if tag == 'near' else 'Far-Range'
            self.logger.info(
                f"  best_model {label}: r²={r2:.4f}  rmse={rmse:.4f}  mae={mae:.4f}"
            )
            summary_rows += [
                [f'{label} R²',   r2],
                [f'{label} RMSE', rmse],
                [f'{label} MAE',  mae],
            ]
            pd.DataFrame({
                'y_true':    y_true,
                'y_pred':    y_pred,
                'residual':  y_true - y_pred,
            }).to_excel(
                best_dir / f'best_{tag}_predictions.xlsx',
                index=False, engine='openpyxl',
            )

        pd.DataFrame(summary_rows, columns=['Metric', 'Value']).to_excel(
            best_dir / 'best_model_metrics.xlsx', index=False, engine='openpyxl',
        )


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    config = CVAEExperimentConfig()

    # Experiment settings
    config.k_folds            = 5
    config.kfold_random_state = 42
    config.t_min              = 20.0
    config.t_max              = 80.0
    config.p_min              = 1e5
    config.p_max              = 1e8
    config.save_predictions   = True
    config.save_metrics       = True
    config.save_cvae_history  = True
    config.log_level          = logging.INFO

    # PC-CVAE hyperparameters
    config.cvae_config.LATENT_DIM                = 2
    config.cvae_config.N_EPOCHS                  = 500
    config.cvae_config.LEARNING_RATE             = 1e-3
    config.cvae_config.LAMBDA_KL                 = 0.001
    config.cvae_config.LAMBDA_COLLOCATION_MCH    = 1
    config.cvae_config.LAMBDA_COLLOCATION_DEC    = 1
    config.cvae_config.LAMBDA_COLLOCATION_HMN    = 1
    config.cvae_config.N_COLLOCATION_POINTS      = 64
    config.cvae_config.COLLOCATION_T_RANGE       = (20.0, 80.0)
    config.cvae_config.COLLOCATION_P_RANGE       = (1e5,  1e8)
    config.cvae_config.Z_LOW                     = -2.0
    config.cvae_config.Z_HIGH                    = 2.0
    config.cvae_config.Z_COLLOC_WIDTH            = 0.5
    config.cvae_config.PHI_HIDDEN_DIMS           = [64, 64]
    config.cvae_config.LAMBDA_CYCLE              = 1.0
    config.cvae_config.N_CYCLE_POINTS            = 64
    config.cvae_config.CYCLE_T_RANGE             = (20.0, 80.0)   # covers the full physical temperature range
    config.cvae_config.CYCLE_P_RANGE             = (1e5,  1e8)    # covers the full physical pressure range
    config.cvae_config.USE_EARLY_STOPPING        = False
    config.cvae_config.DEVICE                    = config.device

    manager = KFoldExperimentManager(config, n_folds=config.k_folds)
    manager.run_all_folds()


if __name__ == '__main__':
    main()