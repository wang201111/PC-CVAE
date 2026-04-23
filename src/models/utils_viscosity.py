"""
Viscosity ternary system physical consistency evaluation utilities.

Provides boundary consistency evaluation (three boundaries: MCH=0, Dec=0, HMN=0),
thermodynamic smoothness evaluation (4D Laplacian + P99 quantile), and a
dual-pillar comprehensive evaluation framework.
Depends on LowDimEnsemble from low_dim_model.py as the low-dimensional
system boundary model interface.

v2 additions (fully symmetric interface with the solubility version):
    DNN            — fully connected baseline network, input_dim=4 (T, P, MCH, Dec).
    PhysicsConfig  — TSTREvaluator hyperparameter configuration.
    TSTREvaluator  — standard training/evaluation pipeline, interface fully
                     symmetric with the solubility version.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from low_dim_model import LowDimEnsemble

warnings.filterwarnings('ignore')


# ==============================================================================
# Logging
# ==============================================================================

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a configured Logger instance."""
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


# ==============================================================================
# Evaluation utility functions
# ==============================================================================

def calculate_boundary_nrmse(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    physical_max: float,
) -> float:
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return float(rmse / (physical_max + 1e-8))


def exponential_decay_score(total_error: float, decay_lambda: float = 5.0) -> float:
    return float(np.exp(-decay_lambda * total_error))


# ==============================================================================
# DNN baseline network
# ==============================================================================

class DNN(nn.Module):
    """Fully connected baseline DNN (viscosity version, input_dim=4).

    Args:
        input_dim:  Input feature dimension; fixed to 4 for the viscosity system.
        layer_dim:  Number of hidden layers.
        node_dim:   Number of nodes per layer.
    """

    def __init__(
        self,
        input_dim: int = 4,
        layer_dim: int = 4,
        node_dim:  int = 128,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(layer_dim):
            layers += [nn.Linear(in_dim, node_dim), nn.ReLU()]
            in_dim = node_dim
        layers.append(nn.Linear(node_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# PhysicsConfig
# ==============================================================================

@dataclass
class PhysicsConfig:
    """TSTREvaluator hyperparameter configuration (fully consistent interface with the solubility version).

    Checkpoint criterion: save when val MSE (original scale) decreases,
    recorded from epoch 1, no epoch threshold.

    Args:
        tstr_epochs:   Number of training epochs.
        tstr_lr:       Learning rate.
        dnn_layer_dim: Number of DNN hidden layers.
        dnn_node_dim:  Number of nodes per layer.
        tstr_device:   Compute device ('auto' / 'cuda' / 'cpu').
    """
    tstr_epochs:   int   = 1000
    tstr_lr:       float = 0.00831
    dnn_layer_dim: int   = 4
    dnn_node_dim:  int   = 128
    tstr_device:   str   = 'auto'

    def __post_init__(self) -> None:
        if self.tstr_device == 'auto':
            self.tstr_device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# TSTREvaluator
# ==============================================================================

class TSTREvaluator:
    """Viscosity system standard training/evaluator (fully symmetric interface with the solubility version).

    Training strategy
    -----------------
    - Train for ``config.tstr_epochs`` epochs (default 1000).
    - After each epoch, compute validation MSE on the original scale;
      save state_dict in memory if it is below the historical minimum
      (recorded from epoch 1, no epoch threshold).
    - After training, load the best checkpoint and perform final evaluation
      on train / val / test sets.

    Args:
        X_val:   Validation features, (N, 4).
        y_val:   Validation targets, (N,).
        X_test:  Test features, (N, 4).
        y_test:  Test targets, (N,).
        X_train: Training features, (N, 4).
        y_train: Training targets, (N,).
        config:  PhysicsConfig hyperparameter configuration.
    """

    def __init__(
        self,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config:  PhysicsConfig,
    ) -> None:
        self.X_val   = X_val
        self.y_val   = y_val
        self.X_test  = X_test
        self.y_test  = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.config  = config
        self.device  = torch.device(config.tstr_device)
        self.logger  = get_logger(self.__class__.__name__)

    def evaluate(
        self,
        X_syn:   np.ndarray,
        y_syn:   np.ndarray,
        epochs:  int  = 1000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train a DNN on synthetic (or real) data and evaluate on val/test sets.

        Checkpoint criterion: save when val MSE (original scale) decreases,
        recorded from epoch 1, no epoch threshold.

        Args:
            X_syn:   Training features (may be real or synthetic data).
            y_syn:   Training targets.
            epochs:  Number of training epochs.
            verbose: Whether to print training progress.

        Returns:
            Dict with the following keys (fully aligned with the solubility version):
                'metrics':      dict with train_*/val_*/test_* metrics.
                'history':      dict with per-epoch train_r2/val_r2/val_loss/test_r2 curves.
                'model':        DNN loaded with best weights.
                'x_scaler':     Feature StandardScaler.
                'y_scaler':     Target StandardScaler.
                'predictions':  dict with 'train'/'val'/'test' prediction arrays (N,).
                'true_values':  dict with 'train'/'val'/'test' true-value arrays (N,).
                'inputs':       dict with 'train'/'val'/'test' input feature arrays.
                'n_synthetic':  Number of training samples.
                'epochs':       Number of training epochs.
                'best_epoch':   Epoch of the best checkpoint (1-indexed).
                'best_val_mse': Best validation MSE (original scale).
        """
        # ── Standardise ────────────────────────────────────────────────────────
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_syn_sc  = x_scaler.fit_transform(X_syn)
        y_syn_sc  = y_scaler.fit_transform(y_syn.reshape(-1, 1))   # (N, 1)
        X_val_sc  = x_scaler.transform(self.X_val)
        X_test_sc = x_scaler.transform(self.X_test)

        # ── Build model ─────────────────────────────────────────────────────────
        model = DNN(
            input_dim=X_syn.shape[1],
            layer_dim=self.config.dnn_layer_dim,
            node_dim=self.config.dnn_node_dim,
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.config.tstr_lr)
        criterion = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_syn_sc).to(self.device),
                torch.FloatTensor(y_syn_sc).to(self.device),   # (N, 1)
            ),
            batch_size=64,
            shuffle=True,
        )

        # ── Checkpoint state (val MSE decreases, recorded from epoch 1) ────────
        best_val_mse     = float('inf')
        best_model_state = None
        best_epoch       = 0

        history: Dict[str, List[float]] = {
            'train_r2': [], 'val_r2': [], 'val_loss': [], 'test_r2': []
        }

        log_every = max(1, epochs // 10)

        # ── Training loop ───────────────────────────────────────────────────────
        for ep in range(epochs):
            # train
            model.train()
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)   # yb: (batch, 1)
                loss.backward()
                optimizer.step()

            # eval (original scale)
            model.eval()
            with torch.no_grad():
                def _inv(X_sc: np.ndarray) -> np.ndarray:
                    out = model(
                        torch.FloatTensor(X_sc).to(self.device)
                    ).cpu().numpy()           # (N, 1)
                    return y_scaler.inverse_transform(out).flatten()

                y_tr_p = _inv(X_syn_sc)
                y_va_p = _inv(X_val_sc)
                y_te_p = _inv(X_test_sc)

            train_r2 = float(r2_score(y_syn,       y_tr_p))
            val_r2   = float(r2_score(self.y_val,  y_va_p))
            val_mse  = float(mean_squared_error(self.y_val, y_va_p))
            test_r2  = float(r2_score(self.y_test, y_te_p))

            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)
            history['val_loss'].append(val_mse)
            history['test_r2'].append(test_r2)

            # Checkpoint: save when val MSE decreases, from epoch 1
            if val_mse < best_val_mse:
                best_val_mse     = val_mse
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch       = ep + 1   # 1-indexed

            if verbose and ((ep + 1) % log_every == 0 or ep + 1 == epochs):
                self.logger.info(
                    f"  Epoch {ep + 1:>4d}/{epochs}  "
                    f"train_r²={train_r2:.4f}  "
                    f"val_r²={val_r2:.4f}  "
                    f"val_mse={val_mse:.6f}  "
                    f"best_ep={best_epoch}"
                )

        # ── Restore best weights (guaranteed, as epoch 1 always triggers) ───────
        model.load_state_dict(best_model_state)
        model.eval()
        self.logger.info(
            f"Best model loaded: epoch {best_epoch} (val MSE={best_val_mse:.6f})"
        )

        # ── Final inference (using best weights) ────────────────────────────────
        with torch.no_grad():
            def _final_pred(X_sc: np.ndarray) -> np.ndarray:
                out = model(
                    torch.FloatTensor(X_sc).to(self.device)
                ).cpu().numpy()
                return y_scaler.inverse_transform(out).flatten()

            y_train_final = _final_pred(X_syn_sc)
            y_val_final   = _final_pred(X_val_sc)
            y_test_final  = _final_pred(X_test_sc)

        # ── Compute final metrics ────────────────────────────────────────────────
        metrics: Dict[str, float] = {
            'train_r2':   float(r2_score(y_syn,       y_train_final)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_syn,       y_train_final))),
            'train_mae':  float(mean_absolute_error(y_syn,       y_train_final)),
            'val_r2':     float(r2_score(self.y_val,  y_val_final)),
            'val_rmse':   float(np.sqrt(mean_squared_error(self.y_val,  y_val_final))),
            'val_mae':    float(mean_absolute_error(self.y_val,  y_val_final)),
            'test_r2':    float(r2_score(self.y_test, y_test_final)),
            'test_rmse':  float(np.sqrt(mean_squared_error(self.y_test, y_test_final))),
            'test_mae':   float(mean_absolute_error(self.y_test, y_test_final)),
        }

        if verbose:
            self.logger.info(
                f"  [Final] train_r²={metrics['train_r2']:.4f}  "
                f"val_r²={metrics['val_r2']:.4f}  "
                f"test_r²={metrics['test_r2']:.4f}  "
                f"best_epoch={best_epoch}"
            )

        return {
            'metrics':      metrics,
            'history':      history,
            'model':        model,
            'x_scaler':     x_scaler,
            'y_scaler':     y_scaler,
            'predictions':  {
                'train': y_train_final,
                'val':   y_val_final,
                'test':  y_test_final,
            },
            'true_values':  {
                'train': y_syn,
                'val':   self.y_val,
                'test':  self.y_test,
            },
            'inputs':       {
                'train': X_syn,
                'val':   self.X_val,
                'test':  self.X_test,
            },
            'n_synthetic':  len(X_syn),
            'epochs':       epochs,
            'best_epoch':   best_epoch,
            'best_val_mse': float(best_val_mse),
        }


# ==============================================================================
# Boundary consistency evaluator
# ==============================================================================

class ViscosityBoundaryEvaluator:
    """Viscosity system boundary consistency evaluator (three boundaries).

    Args:
        model_mch_hmn: MCH-HMN binary system model; predict input is [T, P, x_MCH] (3 columns).
        model_dec_hmn: Dec-HMN binary system model; predict input is [T, P, x_Dec] (3 columns).
        model_mch_dec: MCH-Dec binary system model; predict input is [T, P, x_MCH] (3 columns).
        temp_range: Temperature range (T_min, T_max) in °C.
        pressure_range: Pressure range (P_min, P_max) in Pa.
        decay_lambda: Boundary score decay coefficient.
        n_samples: Number of sample points per boundary.
        log_level: Logging level.
    """

    def __init__(
        self,
        model_mch_hmn: LowDimEnsemble,
        model_dec_hmn: LowDimEnsemble,
        model_mch_dec: LowDimEnsemble,
        temp_range: Tuple[float, float] = (20.0, 80.0),
        pressure_range: Tuple[float, float] = (1e5, 1e8),
        decay_lambda: float = 5.0,
        n_samples: int = 10,
        log_level: int = logging.INFO,
    ):
        self.model_mch_hmn  = model_mch_hmn
        self.model_dec_hmn  = model_dec_hmn
        self.model_mch_dec  = model_mch_dec
        self.temp_range     = temp_range
        self.pressure_range = pressure_range
        self.decay_lambda   = decay_lambda
        self.n_samples      = n_samples
        self.logger         = get_logger(self.__class__.__name__, log_level)

        self._generate_boundary_test_points()

    def _generate_boundary_test_points(self) -> None:
        """Pre-generate test points and low-dimensional model true values for all three boundaries.

        boundary_*_X stores the full 4-column array [T, P, MCH, Dec] for the ternary model.
        Binary models receive column slices (each model accepts only 3 columns):
          MCH=0 → model_dec_hmn, slice [:, [0,1,3]] (T, P, Dec)
          Dec=0 → model_mch_hmn, slice [:, [0,1,2]] (T, P, MCH)
          HMN=0 → model_mch_dec, slice [:, [0,1,2]] (T, P, MCH)
        """
        self.logger.info(f"Generating boundary test points ({self.n_samples} points per boundary)...")

        T_test = np.linspace(*self.temp_range, self.n_samples)
        P_test = np.linspace(*self.pressure_range, self.n_samples)
        T_grid, P_grid = np.meshgrid(T_test, P_test)
        T_flat = T_grid.flatten()
        P_flat = P_grid.flatten()
        n_tp   = len(T_flat)

        # MCH=0 boundary
        Dec_samples = np.linspace(0, 100, self.n_samples)
        rows = []
        for i in range(n_tp):
            for dec in Dec_samples:
                rows.append([T_flat[i], P_flat[i], 0.0, dec])
        self.boundary_mch_zero_X = np.array(rows)
        self.boundary_mch_zero_y_true = self.model_dec_hmn.predict(
            self.boundary_mch_zero_X[:, [0, 1, 3]]
        ).flatten()

        # Dec=0 boundary
        MCH_samples = np.linspace(0, 100, self.n_samples)
        rows = []
        for i in range(n_tp):
            for mch in MCH_samples:
                rows.append([T_flat[i], P_flat[i], mch, 0.0])
        self.boundary_dec_zero_X = np.array(rows)
        self.boundary_dec_zero_y_true = self.model_mch_hmn.predict(
            self.boundary_dec_zero_X[:, [0, 1, 2]]
        ).flatten()

        # HMN=0 boundary
        MCH_samples_hmn = np.linspace(0, 100, self.n_samples)
        rows = []
        for i in range(n_tp):
            for mch in MCH_samples_hmn:
                rows.append([T_flat[i], P_flat[i], mch, 100.0 - mch])
        self.boundary_hmn_zero_X = np.array(rows)
        self.boundary_hmn_zero_y_true = self.model_mch_dec.predict(
            self.boundary_hmn_zero_X[:, [0, 1, 2]]
        ).flatten()

        self.logger.info("Boundary test points generated")

    def evaluate_parl_boundary(self, trainer: Any) -> Dict[str, Any]:
        """Evaluate model consistency along all three boundaries.

        Args:
            trainer: Object with a predict(X, return_original_scale) method.
        """
        self.logger.info("Boundary consistency evaluation started")
        results = {}

        def _eval_one(name: str, X: np.ndarray, y_true: np.ndarray) -> Tuple[float, Dict]:
            y_pred = trainer.predict(X, return_original_scale=True).flatten()
            r2     = r2_score(y_true, y_pred)
            rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae    = float(mean_absolute_error(y_true, y_pred))
            nrmse  = calculate_boundary_nrmse(y_pred, y_true, float(np.max(y_true)))
            self.logger.info(f"  {name}  R²={r2:.4f}  RMSE={rmse:.4f}  NRMSE={nrmse:.6f}")
            return nrmse, {
                'r2': r2, 'rmse': rmse, 'mae': mae,
                'y_true': y_true.copy(), 'y_pred': y_pred.copy(),
                'X': X.copy(),
            }

        nrmse_1, results['mch_zero_boundary'] = _eval_one(
            'MCH=0', self.boundary_mch_zero_X, self.boundary_mch_zero_y_true
        )
        nrmse_2, results['dec_zero_boundary'] = _eval_one(
            'Dec=0', self.boundary_dec_zero_X, self.boundary_dec_zero_y_true
        )
        nrmse_3, results['hmn_zero_boundary'] = _eval_one(
            'HMN=0', self.boundary_hmn_zero_X, self.boundary_hmn_zero_y_true
        )

        total_error    = nrmse_1 + nrmse_2 + nrmse_3
        boundary_score = exponential_decay_score(total_error, self.decay_lambda)

        results['combined'] = {
            'nrmse_mch_zero':        nrmse_1,
            'nrmse_dec_zero':        nrmse_2,
            'nrmse_hmn_zero':        nrmse_3,
            'total_error':           total_error,
            'boundary_score':        boundary_score,
            'physical_max_mch_zero': float(np.max(self.boundary_mch_zero_y_true)),
            'physical_max_dec_zero': float(np.max(self.boundary_dec_zero_y_true)),
            'physical_max_hmn_zero': float(np.max(self.boundary_hmn_zero_y_true)),
            'decay_lambda':          self.decay_lambda,
        }

        self.logger.info(
            f"Boundary overall score: {boundary_score:.6f}  "
            f"(total error={nrmse_1:.4f}+{nrmse_2:.4f}+{nrmse_3:.4f}={total_error:.4f})"
        )
        return results


# ==============================================================================
# Smoothness evaluator
# ==============================================================================

class ViscositySmoothnessEvaluator:
    """Viscosity system thermodynamic smoothness evaluator (4D Laplacian + P99 quantile).

    Args:
        temp_range: Temperature range (T_min, T_max) in °C.
        pressure_range: Pressure range (P_min, P_max) in Pa.
        mch_range: MCH concentration range (min, max) in %.
        dec_range: Dec concentration range (min, max) in %.
        grid_resolution: 4D grid resolution (n_T, n_P, n_MCH, n_Dec).
        smoothness_decay_lambda: Smoothness score decay coefficient.
        log_level: Logging level.
    """

    def __init__(
        self,
        temp_range: Tuple[float, float] = (20.0, 80.0),
        pressure_range: Tuple[float, float] = (1e5, 1e8),
        mch_range: Tuple[float, float] = (0.0, 100.0),
        dec_range: Tuple[float, float] = (0.0, 100.0),
        grid_resolution: Tuple[int, int, int, int] = (20, 20, 20, 20),
        smoothness_decay_lambda: float = 15.0,
        log_level: int = logging.INFO,
    ):
        self.temp_range      = temp_range
        self.pressure_range  = pressure_range
        self.mch_range       = mch_range
        self.dec_range       = dec_range
        self.grid_resolution = grid_resolution
        self.decay_lambda    = smoothness_decay_lambda
        self.logger          = get_logger(self.__class__.__name__, log_level)

    def generate_regular_grid(self) -> np.ndarray:
        """Generate a regular 4D grid; returns (N, 4) array of [T, P, MCH, Dec]."""
        n_T, n_P, n_MCH, n_Dec = self.grid_resolution
        T_s   = np.linspace(*self.temp_range, n_T)
        P_s   = np.linspace(*self.pressure_range, n_P)
        MCH_s = np.linspace(*self.mch_range, n_MCH)
        Dec_s = np.linspace(*self.dec_range, n_Dec)
        T_g, P_g, MCH_g, Dec_g = np.meshgrid(T_s, P_s, MCH_s, Dec_s, indexing='ij')
        X_grid = np.column_stack([
            T_g.flatten(), P_g.flatten(),
            MCH_g.flatten(), Dec_g.flatten(),
        ])
        self.logger.info(
            f"4D regular grid generated: resolution={self.grid_resolution}, "
            f"total points={len(X_grid):,}"
        )
        return X_grid

    def evaluate_smoothness(self, trainer: Any) -> Tuple[float, Dict[str, Any]]:
        """Evaluate thermodynamic smoothness using the 4D Laplacian method."""
        self.logger.info("Thermodynamic smoothness evaluation started (4D Laplacian + P99 quantile)")

        X_grid    = self.generate_regular_grid()
        self.logger.info("Running model predictions...")
        Visc_pred = trainer.predict(X_grid, return_original_scale=True).flatten()
        Visc_4d   = Visc_pred.reshape(self.grid_resolution)
        self.logger.info(f"Predictions complete, 4D tensor shape: {Visc_4d.shape}")

        self.logger.info("Computing 4D Laplacian operator...")
        d2_dT2   = np.gradient(np.gradient(Visc_4d, axis=0), axis=0)
        d2_dP2   = np.gradient(np.gradient(Visc_4d, axis=1), axis=1)
        d2_dMCH2 = np.gradient(np.gradient(Visc_4d, axis=2), axis=2)
        d2_dDec2 = np.gradient(np.gradient(Visc_4d, axis=3), axis=3)

        grad_T   = np.gradient(Visc_4d, axis=0)
        grad_P   = np.gradient(Visc_4d, axis=1)
        grad_MCH = np.gradient(Visc_4d, axis=2)
        grad_Dec = np.gradient(Visc_4d, axis=3)

        laplacian_4d  = d2_dT2 + d2_dP2 + d2_dMCH2 + d2_dDec2
        laplacian_abs = np.abs(laplacian_4d)

        l1_norm   = float(np.mean(laplacian_abs))
        l2_norm   = float(np.sqrt(np.mean(laplacian_4d ** 2)))
        l4_norm   = float(np.power(np.mean(laplacian_abs ** 4), 0.25))
        linf_norm = float(np.max(laplacian_abs))

        p50  = float(np.percentile(laplacian_abs, 50))
        p90  = float(np.percentile(laplacian_abs, 90))
        p95  = float(np.percentile(laplacian_abs, 95))
        p99  = float(np.percentile(laplacian_abs, 99))

        data_range = float(np.ptp(Visc_4d)) + 1e-8
        eta_p99    = p99 / data_range

        smoothness_score = float(np.exp(-self.decay_lambda * eta_p99))
        score_l2  = float(np.exp(-self.decay_lambda * l2_norm  / data_range))
        score_l4  = float(np.exp(-self.decay_lambda * l4_norm  / data_range))
        score_p95 = float(np.exp(-self.decay_lambda * p95      / data_range))

        self.logger.info(f"Smoothness score: {smoothness_score:.6f}  η(P99)={eta_p99:.6f}")

        grad_magnitude = np.sqrt(grad_T**2 + grad_P**2 + grad_MCH**2 + grad_Dec**2)
        tail_thickness = float((p99 - p95) / (p95 + 1e-8))
        quantile_ok    = (p50 <= p90 <= p95 <= p99 <= linf_norm)
        holder_ok      = (l2_norm <= l4_norm + 1e-6) and (l4_norm <= linf_norm + 1e-6)

        if smoothness_score >= 0.99:
            quality, description = 'Excellent', f'Thermodynamic surface is highly smooth (η={eta_p99:.6f})'
        elif smoothness_score >= 0.95:
            quality, description = 'Good', f'Surface is smooth with minor fluctuations (η={eta_p99:.6f})'
        elif smoothness_score >= 0.90:
            quality, description = 'Acceptable', f'Smoothness is adequate with visible noise (η={eta_p99:.6f})'
        elif smoothness_score >= 0.80:
            quality, description = 'Poor', f'Notable roughness detected (η={eta_p99:.6f})'
        else:
            quality, description = 'Unacceptable', f'Severe thermodynamic inconsistency (η={eta_p99:.6f})'

        if tail_thickness > 0.5:
            tail_interp = 'Heavy-tailed (severe outliers present)'
        elif tail_thickness > 0.2:
            tail_interp = 'Moderate tail'
        else:
            tail_interp = 'Light-tailed (well-behaved)'

        details = {
            'normalized_roughness': float(eta_p99),
            'smoothness_score':     smoothness_score,
            'quality':              quality,
            'description':          description,
            'laplacian_p50':  p50,
            'laplacian_p90':  p90,
            'laplacian_p95':  p95,
            'laplacian_p99':  p99,
            'laplacian_max':  linf_norm,
            'normalized_p50':  float(p50  / data_range),
            'normalized_p90':  float(p90  / data_range),
            'normalized_p95':  float(p95  / data_range),
            'normalized_p99':  float(eta_p99),
            'normalized_max':  float(linf_norm / data_range),
            'l1_norm':   l1_norm,
            'l2_norm':   l2_norm,
            'l4_norm':   l4_norm,
            'linf_norm': linf_norm,
            'normalized_l1':   float(l1_norm   / data_range),
            'normalized_l2':   float(l2_norm   / data_range),
            'normalized_l4':   float(l4_norm   / data_range),
            'normalized_linf': float(linf_norm / data_range),
            'score_p99_method': smoothness_score,
            'score_l2_method':  score_l2,
            'score_l4_method':  score_l4,
            'score_p95_method': score_p95,
            'score_difference': float(abs(smoothness_score - score_l2)),
            'data_range': data_range,
            'data_min':   float(np.min(Visc_4d)),
            'data_max':   float(np.max(Visc_4d)),
            'data_mean':  float(np.mean(Visc_4d)),
            'data_std':   float(np.std(Visc_4d)),
            'laplacian_mean':           float(np.mean(laplacian_4d)),
            'laplacian_std':            float(np.std(laplacian_4d)),
            'laplacian_abs_mean':       float(np.mean(laplacian_abs)),
            'laplacian_abs_std':        float(np.std(laplacian_abs)),
            'laplacian_positive_ratio': float(np.sum(laplacian_4d > 0) / laplacian_4d.size),
            'tail_thickness':      tail_thickness,
            'tail_interpretation': tail_interp,
            'p99_to_max_ratio':    float(p99 / (linf_norm + 1e-10)),
            'gradient_magnitude_mean': float(np.mean(grad_magnitude)),
            'gradient_magnitude_max':  float(np.max(grad_magnitude)),
            'gradient_T_rms':          float(np.sqrt(np.mean(grad_T   ** 2))),
            'gradient_P_rms':          float(np.sqrt(np.mean(grad_P   ** 2))),
            'gradient_MCH_rms':        float(np.sqrt(np.mean(grad_MCH ** 2))),
            'gradient_Dec_rms':        float(np.sqrt(np.mean(grad_Dec ** 2))),
            'quantile_order_satisfied':    bool(quantile_ok),
            'holder_inequality_satisfied': bool(holder_ok),
            'theory_consistency':          'Pass' if (quantile_ok and holder_ok) else 'Warning',
            'lambda':          self.decay_lambda,
            'grid_resolution': list(self.grid_resolution),
            'actual_shape':    list(Visc_4d.shape),
            'total_elements':  int(Visc_4d.size),
            'method':          'P99 quantile (CVaR proxy)',
        }

        self.logger.info(f"Smoothness evaluation complete  score={smoothness_score:.6f}  level={quality}")
        return smoothness_score, details


# ==============================================================================
# Comprehensive physics evaluator
# ==============================================================================

class ViscosityPhysicsEvaluator:
    """Viscosity system comprehensive physical consistency evaluator
    (boundary consistency + thermodynamic smoothness dual-pillar framework).

    Args:
        teacher_models: (model_mch_hmn, model_dec_hmn, model_mch_dec).
        temp_range: Temperature range (T_min, T_max).
        pressure_range: Pressure range (P_min, P_max).
        mch_range: MCH concentration range (min, max).
        dec_range: Dec concentration range (min, max).
        boundary_decay_lambda: Boundary score decay coefficient.
        smoothness_decay_lambda: Smoothness score decay coefficient.
        n_boundary_samples: Number of sample points per boundary.
        grid_resolution: 4D grid resolution for smoothness evaluation.
        log_level: Logging level.
    """

    def __init__(
        self,
        teacher_models: Tuple[LowDimEnsemble, LowDimEnsemble, LowDimEnsemble],
        temp_range: Tuple[float, float] = (20.0, 80.0),
        pressure_range: Tuple[float, float] = (1e5, 1e8),
        mch_range: Tuple[float, float] = (0.0, 100.0),
        dec_range: Tuple[float, float] = (0.0, 100.0),
        boundary_decay_lambda: float = 5.0,
        smoothness_decay_lambda: float = 15.0,
        n_boundary_samples: int = 100,
        grid_resolution: Tuple[int, int, int, int] = (20, 20, 20, 20),
        log_level: int = logging.INFO,
    ):
        self.logger = get_logger(self.__class__.__name__, log_level)
        model_mch_hmn, model_dec_hmn, model_mch_dec = teacher_models

        self.boundary_evaluator = ViscosityBoundaryEvaluator(
            model_mch_hmn=model_mch_hmn,
            model_dec_hmn=model_dec_hmn,
            model_mch_dec=model_mch_dec,
            temp_range=temp_range,
            pressure_range=pressure_range,
            decay_lambda=boundary_decay_lambda,
            n_samples=n_boundary_samples,
            log_level=log_level,
        )

        self.smoothness_evaluator = ViscositySmoothnessEvaluator(
            temp_range=temp_range,
            pressure_range=pressure_range,
            mch_range=mch_range,
            dec_range=dec_range,
            grid_resolution=grid_resolution,
            smoothness_decay_lambda=smoothness_decay_lambda,
            log_level=log_level,
        )

        self.logger.info("ViscosityPhysicsEvaluator initialized (dual-pillar framework)")

    def evaluate_full(self, trainer: Any) -> Tuple[float, Dict[str, Any]]:
        """Full physical consistency evaluation. overall_score = 0.5 * boundary + 0.5 * smoothness."""
        self.logger.info("Comprehensive physical consistency evaluation started (dual-pillar framework)")

        self.logger.info("Pillar 1: Boundary consistency evaluation")
        boundary_results = self.boundary_evaluator.evaluate_parl_boundary(trainer)
        boundary_score   = boundary_results['combined']['boundary_score']

        self.logger.info("Pillar 2: Thermodynamic smoothness evaluation")
        smoothness_score, smoothness_details = self.smoothness_evaluator.evaluate_smoothness(trainer)

        overall_score = 0.5 * boundary_score + 0.5 * smoothness_score

        self.logger.info(
            f"Overall evaluation results  boundary={boundary_score:.6f}  "
            f"smoothness={smoothness_score:.6f}  overall={overall_score:.6f}"
        )

        return overall_score, {
            'boundary':         boundary_results,
            'smoothness':       smoothness_details,
            'overall_score':    float(overall_score),
            'boundary_score':   float(boundary_score),
            'smoothness_score': float(smoothness_score),
        }

    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a formatted evaluation report string."""
        bd = results['boundary']['combined']
        sm = results['smoothness']
        lines = [
            '=' * 70,
            'Viscosity System Physical Consistency Evaluation Report',
            '=' * 70,
            '',
            f"Overall score: {results['overall_score']:.6f}",
            '',
            'Pillar 1: Boundary Consistency',
            '-' * 70,
            f"  Boundary score:         {bd['boundary_score']:.6f}",
            f"  MCH=0 boundary NRMSE:   {bd['nrmse_mch_zero']:.6f}",
            f"  Dec=0 boundary NRMSE:   {bd['nrmse_dec_zero']:.6f}",
            f"  HMN=0 boundary NRMSE:   {bd['nrmse_hmn_zero']:.6f}",
            f"  Total error:            {bd['total_error']:.6f}",
            '',
            'Pillar 2: Thermodynamic Smoothness',
            '-' * 70,
            f"  Smoothness score:            {sm['smoothness_score']:.6f}",
            f"  Normalised roughness η(P99): {sm['normalized_roughness']:.6f}",
            f"  Quality level:               {sm['quality']}",
            f"  Description:                 {sm['description']}",
            '',
            '=' * 70,
        ]
        return '\n'.join(lines)


# ==============================================================================
# Public interface
# ==============================================================================

__all__ = [
    'calculate_boundary_nrmse',
    'exponential_decay_score',
    'get_logger',
    'DNN',
    'PhysicsConfig',
    'TSTREvaluator',
    'ViscosityBoundaryEvaluator',
    'ViscositySmoothnessEvaluator',
    'ViscosityPhysicsEvaluator',
]