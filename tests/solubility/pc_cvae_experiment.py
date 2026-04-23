"""
================================================================================
PC-CVAE Ablation Study - K-Fold Cross-Validation Framework
================================================================================

Evaluates PC-CVAE performance using direct deterministic prediction (φ head)
with k-fold cross-validation.

Dataset split (temperature-based):
    - Interpolation domain    : T ≤ 50°C         (training pool, k-fold CV)
    - Near-range extrapolation: 50 < T < 100°C   (held-out evaluation)
    - Far-range extrapolation : T ≥ 100°C         (held-out evaluation)

Workflow:
    1. K-fold CV on interpolation domain → per-fold train/val/near/far metrics
    2. After all folds: identify best fold (by val R²), load saved model,
       produce a dedicated extrapolation report on near / far domains.

Design notes:
    - X_val is NOT passed to CVAEPhysicsModel.fit(): fixed epochs, no early stopping.
    - Boundary models serve as collocation constraints and physics evaluators.
    - cvae.predict(X) is used for all metric computation.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from low_dim_model import LowDimEnsemble
from pc_cvae_solubility import CVAEConfig, CVAEPhysicsModel, LowDimInfo
from utils_solubility import PhysicalConsistencyEvaluator, PhysicsConfig

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
    """Load ternary data from Excel. Returns X (n,2) and y (n,)."""
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    data      = pd.read_excel(filepath, engine='openpyxl')
    col_names = data.columns.tolist()
    temp_col  = next(
        (c for c in col_names if 'T' in c or 'temp' in c.lower()), col_names[0]
    )
    comp_cols = [c for c in col_names if c != temp_col]
    if len(comp_cols) < 2:
        raise ValueError(f"Expected at least 2 composition columns, found {len(comp_cols)}")
    X = data[[temp_col, comp_cols[0]]].values.astype(np.float32)
    y = data[comp_cols[1]].values.astype(np.float32)
    return X, y


def move_low_dim_ensemble_to_device(
    model: LowDimEnsemble, device: torch.device
) -> LowDimEnsemble:
    model.to(device)
    model.device = device
    return model


def create_k_folds(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42
) -> List[Dict[str, np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [
        {
            'fold_idx':      fi,
            'X_train':       X[tr], 'y_train': y[tr],
            'X_val':         X[va], 'y_val':   y[va],
            'train_indices': tr,    'val_indices': va,
        }
        for fi, (tr, va) in enumerate(kf.split(X))
    ]


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class CVAEExperimentConfig:
    """Configuration for PC-CVAE ablation experiment."""

    # ── Data paths ──────────────────────────────────────────────────────────
    data_dir: Path = Path('./data/solubility/split_by_temperature')
    train_data_file:  str = 'interpolation domain.xlsx'
    near_extrap_file: str = 'near-range extrapolation.xlsx'
    far_extrap_file:  str = 'far-range extrapolation.xlsx'

    # ── Boundary model paths ─────────────────────────────────────────────────
    models_dir: Path = Path('./models/Low_dim_model/solubility')
    input_boundary_model_file:  str = 'MgSO4-H2O.pth'
    output_boundary_model_file: str = 'Na2SO4-H2O.pth'

    # ── PC-CVAE hyperparameters ──────────────────────────────────────────────
    cvae_config: CVAEConfig = field(default_factory=lambda: CVAEConfig(
        LATENT_DIM=1,
        HIDDEN_DIMS=[128, 256, 256, 128],
        DROPOUT=0.1,
        LEARNING_RATE=1e-3,
        BATCH_SIZE=64,
        N_EPOCHS=500,
        WEIGHT_DECAY=1e-5,
        LAMBDA_KL=0.01,
        LAMBDA_COLLOCATION_Na2SO4=1,
        LAMBDA_COLLOCATION_MgSO4=1,
        N_COLLOCATION_POINTS=64,
        COLLOCATION_T_RANGE=(-10.0, 190.0),
        Z_LOW=-2.0,
        Z_HIGH=2.0,
        Z_COLLOC_WIDTH=0.5,
        PHI_HIDDEN_DIMS=[64, 64],
        LAMBDA_CYCLE=1.0,
        N_CYCLE_POINTS=64,
        CYCLE_T_RANGE=(-40.0, 200.0),
        USE_EARLY_STOPPING=False,
        USE_LR_SCHEDULER=True,
        LR_SCHEDULER_TYPE='cosine',
        LR_MIN=1e-5,
        DEVICE='auto',
        VERBOSE=False,
    ))

    # ── Physics grid ─────────────────────────────────────────────────────────
    t_min: float = -40.0
    t_max: float = 200.0

    # ── Cross-validation ─────────────────────────────────────────────────────
    k_folds:            int = 5
    kfold_random_state: int = 42

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: Path = (
        PROJECT_ROOT / 'results' / 'solubility' / 'ablation' / 'CVAE_results-dim4'
    )
    save_predictions:  bool = True
    save_metrics:      bool = True
    save_cvae_history: bool = True
    excel_prefix:      str  = 'cvae_'

    # ── Device / logging ─────────────────────────────────────────────────────
    device:    str = 'auto'
    log_level: int = logging.INFO

    def __post_init__(self) -> None:
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cvae_config.DEVICE = self.device
        for attr in ('data_dir', 'models_dir', 'output_dir'):
            p = getattr(self, attr)
            if not p.is_absolute():
                setattr(self, attr, PROJECT_ROOT / p)
        self.train_data_path            = self.data_dir   / self.train_data_file
        self.near_extrap_path           = self.data_dir   / self.near_extrap_file
        self.far_extrap_path            = self.data_dir   / self.far_extrap_file
        self.input_boundary_model_path  = self.models_dir / self.input_boundary_model_file
        self.output_boundary_model_path = self.models_dir / self.output_boundary_model_file


# ==============================================================================
# Single Fold Runner
# ==============================================================================

class SingleFoldRunner:
    """Runs the full PC-CVAE pipeline for one K-fold split."""

    def __init__(
        self,
        config:                CVAEExperimentConfig,
        input_boundary_model:  Optional[LowDimEnsemble],
        output_boundary_model: Optional[LowDimEnsemble],
    ) -> None:
        self.config                = config
        self.logger                = get_logger(self.__class__.__name__, config.log_level)
        self.device                = torch.device(config.device)
        self.input_boundary_model  = input_boundary_model
        self.output_boundary_model = output_boundary_model

    def run(
        self,
        fold_idx: int,
        X_train:  np.ndarray, y_train: np.ndarray,
        X_val:    np.ndarray, y_val:   np.ndarray,
        X_near:   np.ndarray, y_near:  np.ndarray,
        X_far:    np.ndarray, y_far:   np.ndarray,
        fold_dir: Path,
    ) -> Dict[str, Any]:
        """Execute the three-step pipeline for one fold.

        Steps:
            1. Train PC-CVAE on interpolation domain (fixed epochs, no val).
            2. Evaluate on train / val / near / far via cvae.predict().
            3. Compute physical consistency metrics.
        """
        self.logger.info("=" * 70)
        self.logger.info(f"Fold {fold_idx}")
        self.logger.info("=" * 70)

        self.logger.info("[Step 1] Training PC-CVAE")
        cvae, cvae_history = self._train_cvae(X_train, y_train)

        self.logger.info("[Step 2] Evaluating on train / val / near / far")
        metrics = self._evaluate_cvae(
            cvae, X_train, y_train, X_val, y_val, X_near, y_near, X_far, y_far,
        )

        self.logger.info("[Step 3] Physics metrics")
        physics_results = self._compute_physics_metrics(cvae)

        self._save_fold_results(fold_idx, fold_dir, metrics, physics_results,
                                cvae_history, cvae)

        return {
            'fold_idx':           fold_idx,
            'metrics':            metrics,
            'physics_score':      physics_results.get('physics_score',      None),
            'physics_boundary':   physics_results.get('physics_boundary',   None),
            'physics_smoothness': physics_results.get('physics_smoothness', None),
        }

    # ------------------------------------------------------------------
    # Step 1 – Training
    # ------------------------------------------------------------------

    def _train_cvae(
        self, X_train: np.ndarray, y_train: np.ndarray,
    ) -> Tuple[CVAEPhysicsModel, dict]:
        low_dim_list: Optional[List[LowDimInfo]] = None

        if self.input_boundary_model is not None and self.output_boundary_model is not None:
            low_dim_list = [
                LowDimInfo(model=self.output_boundary_model,
                           name='MgSO4_H2O', constraint_type='Na2SO4'),
                LowDimInfo(model=self.input_boundary_model,
                           name='Na2SO4_H2O', constraint_type='MgSO4'),
            ]
        else:
            self.logger.warning("Boundary models unavailable — collocation losses disabled")

        cvae    = CVAEPhysicsModel(input_dim=3, condition_dim=1,
                                   config=self.config.cvae_config)
        history = cvae.fit(X=X_train, y=y_train,
                           low_dim_list=low_dim_list, X_val=None, y_val=None)

        final = history['train'][-1] if history['train'] else {}
        self.logger.info(
            f"  CVAE done — total={final.get('total', float('nan')):.5f}  "
            f"recon={final.get('recon', float('nan')):.5f}  "
            f"KL={final.get('kl', float('nan')):.5f}  "
            f"cycle={final.get('cycle', float('nan')):.5f}  "
            f"colloc_Na2SO4={final.get('colloc_Na2SO4', float('nan')):.5f}  "
            f"colloc_MgSO4={final.get('colloc_MgSO4', float('nan')):.5f}"
        )
        return cvae, history

    # ------------------------------------------------------------------
    # Step 2 – Evaluation
    # ------------------------------------------------------------------

    def _evaluate_cvae(
        self,
        cvae:    CVAEPhysicsModel,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        X_near:  np.ndarray, y_near:  np.ndarray,
        X_far:   np.ndarray, y_far:   np.ndarray,
    ) -> Dict[str, Any]:
        """Compute R², RMSE, MAE on all four splits via cvae.predict()."""
        splits = {
            'train': (X_train, y_train),
            'val':   (X_val,   y_val),
            'near':  (X_near,  y_near),
            'far':   (X_far,   y_far),
        }

        inputs = {s: X              for s, (X, _) in splits.items()}
        preds  = {s: cvae.predict(X) for s, (X, _) in splits.items()}
        trues  = {s: y              for s, (_, y) in splits.items()}

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
        return {'metrics': metrics_result, 'predictions': preds,
                'true_values': trues, 'inputs': inputs}

    # ------------------------------------------------------------------
    # Step 3 – Physics
    # ------------------------------------------------------------------

    def _compute_physics_metrics(self, cvae: CVAEPhysicsModel) -> Dict[str, float]:
        if self.input_boundary_model is None or self.output_boundary_model is None:
            self.logger.warning("Boundary models unavailable — skipping physics eval")
            return {}
        try:
            T_grid = np.linspace(self.config.t_min, self.config.t_max, 100)
            W_grid = np.linspace(0, 55, 100)
            T_mesh, W_mesh = np.meshgrid(T_grid, W_grid)
            X_phase = np.column_stack(
                [T_mesh.ravel(), W_mesh.ravel()]
            ).astype(np.float32)

            y_pred         = cvae.predict(X_phase)
            predicted_data = np.column_stack([X_phase[:, 0], X_phase[:, 1], y_pred])

            evaluator = PhysicalConsistencyEvaluator(
                input_boundary_model=self.input_boundary_model,
                output_boundary_model=self.output_boundary_model,
            )
            score, results = evaluator.evaluate_with_predictor(
                predict_fn=cvae.predict,
                predicted_data=predicted_data,
            )
            self.logger.info(
                f"  physics={score:.4f}  "
                f"boundary={results.get('boundary_score', float('nan')):.4f}  "
                f"smoothness={results.get('smoothness_score', float('nan')):.4f}"
            )
            return {
                'physics_score':      score,
                'physics_boundary':   results.get('boundary_score',  None),
                'physics_smoothness': results.get('smoothness_score', None),
            }
        except Exception as e:
            import traceback
            self.logger.error(f"Physics evaluation failed: {e}\n{traceback.format_exc()}")
            return {}

    # ------------------------------------------------------------------
    # Saving
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
        if self.config.save_metrics:
            self._save_fold_metrics(metrics, physics_results, fold_dir)
        if self.config.save_predictions:
            self._save_predictions(metrics, fold_dir)
        if self.config.save_cvae_history:
            self._save_cvae_history(cvae_history, fold_dir)
        # Save model for best-model loading
        model_dir = fold_dir / 'model'
        model_dir.mkdir(exist_ok=True)
        cvae.save(str(model_dir / 'cvae.pth'))

    def _save_fold_metrics(
        self, metrics: Dict, physics_results: Dict, fold_dir: Path
    ) -> None:
        inner = metrics.get('metrics', metrics)
        pe    = physics_results

        physics_score    = pe.get('physics_score',      float('nan')) if pe else float('nan')
        boundary_score   = pe.get('physics_boundary',   float('nan')) if pe else float('nan')
        smoothness_score = pe.get('physics_smoothness', float('nan')) if pe else float('nan')

        rows = [
            ['Train R²',                inner.get('train_r2',   float('nan'))],
            ['Train RMSE',              inner.get('train_rmse', float('nan'))],
            ['Train MAE',               inner.get('train_mae',  float('nan'))],
            ['Val R²',                  inner.get('val_r2',     float('nan'))],
            ['Val RMSE',                inner.get('val_rmse',   float('nan'))],
            ['Val MAE',                 inner.get('val_mae',    float('nan'))],
            ['Near-Range R²',           inner.get('near_r2',    float('nan'))],
            ['Near-Range RMSE',         inner.get('near_rmse',  float('nan'))],
            ['Near-Range MAE',          inner.get('near_mae',   float('nan'))],
            ['Far-Range R²',            inner.get('far_r2',     float('nan'))],
            ['Far-Range RMSE',          inner.get('far_rmse',   float('nan'))],
            ['Far-Range MAE',           inner.get('far_mae',    float('nan'))],
            ['Physics Score',           physics_score],
            ['Boundary Consistency',    boundary_score],
            ['Thermodynamic Smoothness', smoothness_score],
        ]
        pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
            fold_dir / f'{self.config.excel_prefix}metrics.xlsx',
            index=False, engine='openpyxl',
        )

    def _save_predictions(self, metrics: Dict, fold_dir: Path) -> None:
        predictions = metrics.get('predictions', {})
        true_values = metrics.get('true_values', {})
        inputs      = metrics.get('inputs', {})
        for split in ('train', 'val', 'near', 'far'):
            if split not in predictions:
                continue
            X      = inputs.get(split)
            y_true = true_values[split].flatten()
            y_pred = predictions[split].flatten()
            df_dict = {}
            if X is not None:
                df_dict['T/°C']        = X[:, 0]
                df_dict['W(MgSO4)/%']  = X[:, 1]
            df_dict['W(Na2SO4)/%_true'] = y_true
            df_dict['W(Na2SO4)/%_pred'] = y_pred
            df_dict['residual']          = y_true - y_pred
            pd.DataFrame(df_dict).to_excel(
                fold_dir / f'{self.config.excel_prefix}{split}_predictions.xlsx',
                index=False, engine='openpyxl',
            )

    def _save_cvae_history(self, history: dict, fold_dir: Path) -> None:
        train_epochs = history.get('train', [])
        if not train_epochs:
            return
        rows = [
            {
                'epoch':        ep,
                'train_total':  e.get('total',         float('nan')),
                'train_recon':  e.get('recon',         float('nan')),
                'train_kl':     e.get('kl',            float('nan')),
                'train_Na2SO4': e.get('colloc_Na2SO4', float('nan')),
                'train_MgSO4':  e.get('colloc_MgSO4',  float('nan')),
                'train_cycle':  e.get('cycle',         float('nan')),
            }
            for ep, e in enumerate(train_epochs)
        ]
        df = pd.DataFrame(rows)
        val_epochs = history.get('val', [])
        if val_epochs:
            df = df.merge(
                pd.DataFrame([
                    {'epoch': ep, 'val_total': e.get('total', float('nan'))}
                    for ep, e in enumerate(val_epochs)
                ]),
                on='epoch', how='left',
            )
        lr_list = history.get('lr', [])
        if lr_list:
            df['lr'] = lr_list[:len(df)]
        df.to_excel(
            fold_dir / f'{self.config.excel_prefix}cvae_history.xlsx',
            index=False, engine='openpyxl',
        )


# ==============================================================================
# K-Fold Experiment Manager
# ==============================================================================

class KFoldExperimentManager:
    """Orchestrates K-fold CV and best-model extrapolation evaluation."""

    def __init__(self, config: CVAEExperimentConfig, n_folds: int = 5) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        self.config      = config
        self.n_folds     = n_folds
        self.logger      = get_logger(self.__class__.__name__, level=config.log_level)
        self.output_dir  = config.output_dir
        self.all_results: List[Dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.input_boundary_model  = None
        self.output_boundary_model = None
        self._load_boundary_models()

    def _load_boundary_models(self) -> None:
        try:
            device  = torch.device(self.config.device)
            in_path = self.config.input_boundary_model_path
            ou_path = self.config.output_boundary_model_path
            if in_path.exists() and ou_path.exists():
                self.input_boundary_model = move_low_dim_ensemble_to_device(
                    LowDimEnsemble.load(str(in_path)), device
                )
                self.output_boundary_model = move_low_dim_ensemble_to_device(
                    LowDimEnsemble.load(str(ou_path)), device
                )
                self.logger.info(f"Boundary models loaded → {device}")
            else:
                self.logger.warning(
                    "Boundary model files not found — "
                    "collocation constraints and physics eval disabled"
                )
        except Exception as e:
            self.logger.error(f"Failed to load boundary models: {e}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_all_folds(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info("PC-CVAE Ablation — K-Fold CV (Direct φ Prediction)")
        self.logger.info("=" * 70)
        self.logger.info(f"K-Folds: {self.n_folds}  Device: {self.config.device}")

        start = time.time()

        # ── Load three datasets ───────────────────────────────────────────
        cfg = self.config
        X_train_pool, y_train_pool = load_ternary_data(cfg.train_data_path)
        X_near,       y_near       = load_ternary_data(cfg.near_extrap_path)
        X_far,        y_far        = load_ternary_data(cfg.far_extrap_path)

        self.logger.info(
            f"Datasets loaded — "
            f"interpolation: {len(X_train_pool)}  "
            f"near: {len(X_near)}  far: {len(X_far)}"
        )
        self.X_near, self.y_near = X_near, y_near
        self.X_far,  self.y_far  = X_far,  y_far

        folds = create_k_folds(
            X_train_pool, y_train_pool,
            n_splits=self.n_folds,
            random_state=cfg.kfold_random_state,
        )
        for f in folds:
            self.logger.info(
                f"  Fold {f['fold_idx']}: train={len(f['X_train'])}  val={len(f['X_val'])}"
            )

        # ── Run each fold ─────────────────────────────────────────────────
        for fold_data in folds:
            fi       = fold_data['fold_idx']
            fold_dir = self.output_dir / f'fold_{fi}'
            fold_dir.mkdir(exist_ok=True)
            self.logger.info(f"\n{'█'*70}\nFold {fi+1}/{self.n_folds}\n{'█'*70}")

            runner = SingleFoldRunner(
                config=cfg,
                input_boundary_model=self.input_boundary_model,
                output_boundary_model=self.output_boundary_model,
            )
            result = runner.run(
                fold_idx=fi,
                X_train=fold_data['X_train'], y_train=fold_data['y_train'],
                X_val=fold_data['X_val'],     y_val=fold_data['y_val'],
                X_near=X_near, y_near=y_near,
                X_far=X_far,   y_far=y_far,
                fold_dir=fold_dir,
            )
            self.all_results.append(result)

        # ── Best-model extrapolation ───────────────────────────────────────
        self._best_model_evaluation()

        # ── Summary ───────────────────────────────────────────────────────
        self._generate_summary()
        self.logger.info(
            f"\nAll folds done in {timedelta(seconds=int(time.time()-start))}"
            f"  Results: {self.output_dir}"
        )

    # ------------------------------------------------------------------
    # Best-model evaluation
    # ------------------------------------------------------------------

    def _best_model_evaluation(self) -> None:
        """Load the best-fold CVAE and produce a dedicated extrapolation report."""
        val_r2s  = [r['metrics']['metrics']['val_r2'] for r in self.all_results]
        best_fi  = int(np.argmax(val_r2s))
        self.logger.info(
            f"\nBest fold by val R²: Fold {best_fi}  "
            f"(val_r²={val_r2s[best_fi]:.4f})"
        )

        # Load best fold model using the built-in classmethod
        cfg      = self.config
        weights  = self.output_dir / f'fold_{best_fi}' / 'model' / 'cvae.pth'
        cvae     = CVAEPhysicsModel.load(str(weights))

        best_dir = self.output_dir / 'best_model'
        best_dir.mkdir(exist_ok=True)

        for tag, X_ext, y_ext in [
            ('near_extrap', self.X_near, self.y_near),
            ('far_extrap',  self.X_far,  self.y_far),
        ]:
            y_pred = cvae.predict(X_ext)
            r2   = float(r2_score(y_ext, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_ext, y_pred)))
            mae  = float(mean_absolute_error(y_ext, y_pred))
            self.logger.info(
                f"  Best model {tag}: R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}"
            )
            pd.DataFrame({
                'T/°C':              X_ext[:, 0],
                'W(MgSO4)/%':        X_ext[:, 1],
                'W(Na2SO4)/%_true':  y_ext,
                'W(Na2SO4)/%_pred':  y_pred,
                'residual':          y_ext - y_pred,
            }).to_excel(
                best_dir / f'{cfg.excel_prefix}{tag}_predictions.xlsx',
                index=False, engine='openpyxl',
            )

        import shutil
        shutil.copy(weights, best_dir / 'cvae_best.pth')
        self.logger.info(f"Best model artefacts → {best_dir}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _generate_summary(self) -> None:
        summary_dir = self.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)
        self._save_summary_metrics(summary_dir)
        self._save_summary_predictions(summary_dir)
        self._generate_text_report(summary_dir)

    def _save_summary_metrics(self, summary_dir: Path) -> None:
        STAT_KEYS = [
            'train_r2', 'train_rmse', 'train_mae',
            'val_r2',   'val_rmse',   'val_mae',
            'near_r2',  'near_rmse',  'near_mae',
            'far_r2',   'far_rmse',   'far_mae',
        ]
        STAT_NAMES = [
            'Train R²',      'Train RMSE',      'Train MAE',
            'Val R²',        'Val RMSE',         'Val MAE',
            'Near-Range R²', 'Near-Range RMSE', 'Near-Range MAE',
            'Far-Range R²',  'Far-Range RMSE',  'Far-Range MAE',
        ]
        PHYS_ATTRS = [
            ('physics_score',      'Physics Score'),
            ('physics_boundary',   'Boundary Consistency'),
            ('physics_smoothness', 'Thermodynamic Smoothness'),
        ]

        rows = []
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
            rows.append({
                'Metric':     label,
                'Mean Value': float(np.mean(vals)) if vals else float('nan'),
                'Std':        float(np.std(vals, ddof=1)) if len(vals) > 1 else float('nan'),
            })

        pd.DataFrame(rows, columns=['Metric', 'Mean Value', 'Std']).to_excel(
            summary_dir / 'summary_metrics.xlsx', index=False, engine='openpyxl'
        )

    def _save_summary_predictions(self, summary_dir: Path) -> None:
        for split in ('near', 'far'):
            X_ref  = self.all_results[0]['metrics']['inputs'][split]
            y_true = self.all_results[0]['metrics']['true_values'][split].flatten()
            data   = {
                'T/°C':             X_ref[:, 0],
                'W(MgSO4)/%':       X_ref[:, 1],
                'W(Na2SO4)/%_true': y_true,
            }
            for r in self.all_results:
                data[f"y_pred_fold{r['fold_idx']}"] = \
                    r['metrics']['predictions'][split].flatten()
            preds = np.array([
                r['metrics']['predictions'][split].flatten() for r in self.all_results
            ])
            data['y_pred_mean']   = np.mean(preds, axis=0)
            data['y_pred_std']    = np.std(preds,  axis=0, ddof=1)
            data['residual_mean'] = y_true - data['y_pred_mean']
            pd.DataFrame(data).to_excel(
                summary_dir / f'{split}_predictions_summary.xlsx',
                index=False, engine='openpyxl',
            )

    def _generate_text_report(self, summary_dir: Path) -> None:
        cfg = self.config
        c   = cfg.cvae_config
        sep = '-' * 70

        lines = [
            '=' * 70,
            'PC-CVAE Ablation — K-Fold CV Summary (Direct φ Prediction)',
            '=' * 70,
            f"Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"K-Folds:     {self.n_folds}",
            f"Device:      {cfg.device}",
            '', sep, 'PC-CVAE Configuration', sep,
            f"Latent dim:         {c.LATENT_DIM}",
            f"Hidden dims:        {c.HIDDEN_DIMS}",
            f"N_EPOCHS:           {c.N_EPOCHS}  (fixed, no early stopping)",
            f"λ_KL:               {c.LAMBDA_KL}",
            f"λ_Na2SO4:           {c.LAMBDA_COLLOCATION_Na2SO4}",
            f"λ_MgSO4:            {c.LAMBDA_COLLOCATION_MgSO4}",
            f"N_collocation:      {c.N_COLLOCATION_POINTS}",
            f"Collocation T:      {c.COLLOCATION_T_RANGE}",
            f"Z range:            [{c.Z_LOW}, {c.Z_HIGH}]",
            f"Z colloc width:     {c.Z_COLLOC_WIDTH}",
            '', sep, 'φ Head', sep,
            f"φ hidden dims:      {c.PHI_HIDDEN_DIMS}",
            f"λ_cycle:            {c.LAMBDA_CYCLE}",
            f"N_cycle_points:     {c.N_CYCLE_POINTS}",
            f"Cycle T range:      {c.CYCLE_T_RANGE}",
            '', sep, 'Dataset Files', sep,
            f"Interpolation domain:      {cfg.train_data_file}",
            f"Near-range extrapolation:  {cfg.near_extrap_file}",
            f"Far-range extrapolation:   {cfg.far_extrap_file}",
            '', sep, 'K-Fold Summary Statistics', sep,
        ]

        keys  = ['train_r2', 'train_rmse', 'train_mae',
                 'val_r2',   'val_rmse',   'val_mae',
                 'near_r2',  'near_rmse',  'near_mae',
                 'far_r2',   'far_rmse',   'far_mae']
        names = ['Train R²',      'Train RMSE',      'Train MAE',
                 'Val R²',        'Val RMSE',         'Val MAE',
                 'Near-Range R²', 'Near-Range RMSE', 'Near-Range MAE',
                 'Far-Range R²',  'Far-Range RMSE',  'Far-Range MAE']
        for k, name in zip(keys, names):
            vals = [r['metrics']['metrics'][k] for r in self.all_results]
            lines.append(f"{name:22s}: {np.mean(vals):.6f} ± {np.std(vals, ddof=1):.6f}")

        lines += ['\nPhysics Evaluation:']
        for attr, name in [
            ('physics_score',      'Physics Score'),
            ('physics_boundary',   'Boundary Consistency'),
            ('physics_smoothness', 'Thermodynamic Smoothness'),
        ]:
            vals = [r.get(attr) for r in self.all_results
                    if r.get(attr) is not None
                    and not np.isnan(float(r.get(attr, float('nan'))))]
            if vals:
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                lines.append(f"{name:22s}: {np.mean(vals):.6f} ± {std:.6f}")
            else:
                lines.append(f"{name:22s}: N/A")

        val_r2s  = [r['metrics']['metrics']['val_r2'] for r in self.all_results]
        best     = self.all_results[int(np.argmax(val_r2s))]
        bm       = best['metrics']['metrics']
        lines += [
            '', sep, 'Best Fold (by Val R²)', sep,
            f"Fold {best['fold_idx']}",
            f"  Val R²:        {bm['val_r2']:.6f}",
            f"  Near-Range R²: {bm['near_r2']:.6f}",
            f"  Far-Range R²:  {bm['far_r2']:.6f}",
            '', sep, 'Stability (Val R²)', sep,
            f"Std:  {np.std(val_r2s, ddof=1):.6f}",
            f"CV:   {np.std(val_r2s, ddof=1) / np.mean(val_r2s) * 100:.2f}%",
            '', '=' * 70,
        ]

        report = '\n'.join(lines)
        (summary_dir / 'summary_report.txt').write_text(report, encoding='utf-8')
        self.logger.info(f'\n{report}')


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    config = CVAEExperimentConfig()

    config.k_folds            = 5
    config.kfold_random_state = 42
    config.t_min              = -10.0
    config.t_max              = 200.0
    config.save_predictions   = True
    config.save_metrics       = True
    config.save_cvae_history  = True
    config.log_level          = logging.INFO

    config.cvae_config.LATENT_DIM                = 4
    config.cvae_config.N_EPOCHS                  = 500
    config.cvae_config.LEARNING_RATE             = 1e-3
    config.cvae_config.LAMBDA_KL                 = 0.001
    config.cvae_config.LAMBDA_COLLOCATION_Na2SO4 = 0
    config.cvae_config.LAMBDA_COLLOCATION_MgSO4  = 0
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

    manager = KFoldExperimentManager(config, n_folds=config.k_folds)
    manager.run_all_folds()


if __name__ == '__main__':
    main()