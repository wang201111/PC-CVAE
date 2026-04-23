"""
Ternary system solubility physical consistency evaluation utilities.

Provides boundary consistency evaluation, thermodynamic smoothness evaluation
(2D Laplacian), TSTR (Train on Synthetic, Test on Real) evaluation, and
supporting data loading utilities.
Depends on LowDimEnsemble from low_dim_model.py as the low-dimensional
system boundary model interface.
"""

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import griddata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings('ignore')


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a configured Logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger(__name__)


models_path = Path(__file__).parent.parent.parent / 'src' / 'models'
sys.path.insert(0, str(models_path))

from low_dim_model import LowDimEnsemble


class FcBlock(nn.Module):
    """Fully connected block: Linear → BatchNorm1d → PReLU."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc  = nn.Linear(in_dim, out_dim)
        self.bn  = nn.BatchNorm1d(out_dim)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.fc(x)))


class DNN(nn.Module):
    """Configurable deep fully connected network: input layer → FcBlock × N → output layer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layer_dim: int = 3,
        node_dim: int = 100,
    ):
        super().__init__()
        self.fc1    = FcBlock(in_dim, node_dim)
        self.fcList = nn.ModuleList([
            FcBlock(node_dim, node_dim) for _ in range(layer_dim - 2)
        ])
        self.fcn = FcBlock(node_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        for fc in self.fcList:
            x = fc(x)
        return self.fcn(x)


@dataclass
class PhysicsConfig:
    """Physical evaluation parameter configuration."""

    # Temperature range
    T_min: float = -10.0
    T_max: float = 190.0
    T_boundary_points: int = 100

    # Concentration range
    W_min: float = 0.0
    W_max: float = 50.0
    output_ref_range: Tuple[float, float] = (0.0, 80.0)

    # Grid resolution
    grid_resolution_T: int = 300
    grid_resolution_W: int = 200

    # Visualization (boundary line width/alpha, for external plotting)
    boundary_line_width: float = 3.0
    boundary_line_alpha: float = 0.9

    # TSTR training
    # Checkpoint criterion: val MSE decreases, recorded from epoch 1, no start_epoch threshold
    tstr_epochs: int = 1000
    tstr_batch_size: int = 2000
    tstr_lr: float = 0.00831
    tstr_device: str = 'cuda'

    # DNN architecture
    dnn_layer_dim: int = 4
    dnn_node_dim: int = 128

    # Score decay coefficients: larger values yield stricter scoring
    boundary_decay_lambda: float = 5.0
    smoothness_decay_lambda: float = 15.0

    log_level: int = logging.INFO


def calculate_boundary_nrmse(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    physical_max: float,
) -> float:
    """Calculate the normalised root mean square error (NRMSE) for boundary predictions.

    Args:
        y_pred: Predicted values, shape (N,).
        y_true: True values, shape (N,).
        physical_max: Normalisation reference (maximum physical value).

    Returns:
        NRMSE in the range [0, 1].
    """
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return float(rmse / (physical_max + 1e-8))


def exponential_decay_score(total_error: float, decay_lambda: float = 5.0) -> float:
    """Exponential decay scoring function.

    Formula: S = exp(-λ · ε)

    Args:
        total_error: Accumulated error.
        decay_lambda: Decay coefficient; larger values yield stricter scoring.

    Returns:
        Score in the range [0, 1].
    """
    return float(np.exp(-decay_lambda * total_error))


def load_ternary_data(
    filepath: str,
    input_cols: List[str] = None,
    output_col: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ternary system data from a file.

    Args:
        filepath: Path to the data file (CSV or Excel).
        input_cols: List of input column names; if None, all columns except the last are used.
        output_col: Output column name; if None, the last column is used.

    Returns:
        Tuple of (X, y) arrays.
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    if input_cols is None:
        input_cols = df.columns[:-1].tolist()
    if output_col is None:
        output_col = df.columns[-1]

    X = df[input_cols].values
    y = df[output_col].values

    logger.info(f"Data loaded: {filepath}, {len(X)} samples")
    return X, y


def split_data_three_way(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a dataset into train / validation / test subsets.

    Args:
        X: Input features, shape (N, D).
        y: Target values, shape (N,).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_state: Random seed.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "The three ratios must sum to 1.0"

    from sklearn.model_selection import train_test_split

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )
    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adj, random_state=random_state
    )

    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_low_dim_model(model_path: str) -> LowDimEnsemble:
    """Load a pre-trained low-dimensional system prediction model."""
    model = LowDimEnsemble.load(model_path)
    logger.info(f"Low-dimensional model loaded: {model_path}")
    return model


def load_boundary_models(
    input_model_path: str,
    output_model_path: str,
) -> Tuple[LowDimEnsemble, LowDimEnsemble]:
    """Load both the input-side and output-side boundary models.

    Returns:
        (input_boundary_model, output_boundary_model).
    """
    input_model  = load_low_dim_model(input_model_path)
    output_model = load_low_dim_model(output_model_path)
    logger.info("Both boundary models loaded")
    return input_model, output_model


class DNNBoundaryEvaluator:
    """DNN boundary consistency evaluator.

    Compares DNN predictions along two boundary lines (input side, output side)
    against the corresponding low-dimensional system models.
    """

    def __init__(
        self,
        input_boundary_model: LowDimEnsemble,
        output_boundary_model: LowDimEnsemble,
        config: PhysicsConfig = None,
    ):
        if config is None:
            config = PhysicsConfig()

        self.config = config
        self.input_boundary_model  = input_boundary_model
        self.output_boundary_model = output_boundary_model
        self.logger = get_logger(self.__class__.__name__, config.log_level)

        self.logger.info("DNNBoundaryEvaluator initialized")

    def evaluate_boundaries(
        self,
        dnn_predict_fn: Callable[[np.ndarray], np.ndarray],
        T_points: np.ndarray = None,
    ) -> Dict[str, Any]:
        """Evaluate DNN NRMSE along both boundary lines.

        Args:
            dnn_predict_fn: Function that accepts a [T, W_input] array and returns W_output predictions.
            T_points: Temperature points for evaluation, shape (N, 1); uses config defaults when None.

        Returns:
            Dict containing input_nrmse, output_nrmse, and boundary prediction arrays.
        """
        if T_points is None:
            T_points = np.linspace(
                self.config.T_min, self.config.T_max, self.config.T_boundary_points
            ).reshape(-1, 1)

        self.logger.debug(f"Evaluating boundaries at {len(T_points)} temperature points")

        input_boundary_true = self.input_boundary_model.predict(T_points).flatten()
        X_input_boundary    = np.column_stack([T_points.flatten(), input_boundary_true])
        input_boundary_pred = dnn_predict_fn(X_input_boundary)
        if input_boundary_pred.ndim > 1:
            input_boundary_pred = input_boundary_pred.flatten()

        output_boundary_true = self.output_boundary_model.predict(T_points).flatten()
        X_output_boundary    = np.column_stack([
            T_points.flatten(), np.zeros_like(T_points.flatten())
        ])
        output_boundary_pred = dnn_predict_fn(X_output_boundary)
        if output_boundary_pred.ndim > 1:
            output_boundary_pred = output_boundary_pred.flatten()

        input_nrmse = calculate_boundary_nrmse(
            input_boundary_pred,
            np.zeros_like(input_boundary_pred),
            self.config.output_ref_range[1],
        )
        output_nrmse = calculate_boundary_nrmse(
            output_boundary_pred,
            output_boundary_true,
            self.config.output_ref_range[1],
        )

        self.logger.info(f"Boundary evaluation: input NRMSE={input_nrmse:.4f}, "
                         f"output NRMSE={output_nrmse:.4f}")

        return {
            'input_nrmse':          float(input_nrmse),
            'output_nrmse':         float(output_nrmse),
            'input_boundary_true':  np.zeros_like(input_boundary_pred),
            'input_boundary_pred':  input_boundary_pred,
            'output_boundary_true': output_boundary_true,
            'output_boundary_pred': output_boundary_pred,
            'T_points':             T_points.flatten(),
        }


class PhysicalConsistencyEvaluator:
    """Physical consistency evaluator (boundary consistency + thermodynamic smoothness dual-pillar framework)."""

    def __init__(
        self,
        input_boundary_model: LowDimEnsemble,
        output_boundary_model: LowDimEnsemble,
        config: PhysicsConfig = None,
        boundary_decay_lambda: float = None,
        smoothness_decay_lambda: float = None,
    ):
        if config is None:
            config = PhysicsConfig()

        self.config = config
        self.boundary_evaluator = DNNBoundaryEvaluator(
            input_boundary_model, output_boundary_model, config
        )
        self.boundary_decay_lambda  = boundary_decay_lambda  or config.boundary_decay_lambda
        self.smoothness_decay_lambda = smoothness_decay_lambda or config.smoothness_decay_lambda
        self.logger = get_logger(self.__class__.__name__, config.log_level)

        self.logger.info("PhysicalConsistencyEvaluator initialized")

    def evaluate_smoothness(self, predicted_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate thermodynamic smoothness using the 2D Laplacian operator.

        Args:
            predicted_data: Predicted data, shape (N, 3), columns [T, W_input, W_output].

        Returns:
            Dict containing eta (normalised curvature), laplacian_p99, data_range, quality_level.
        """
        self.logger.debug("Evaluating thermodynamic smoothness")

        T        = predicted_data[:, 0]
        W_input  = predicted_data[:, 1]
        W_output = predicted_data[:, 2]

        T_grid = np.linspace(T.min(), T.max(), self.config.grid_resolution_T)
        W_grid = np.linspace(W_input.min(), W_input.max(), self.config.grid_resolution_W)
        T_mesh, W_mesh = np.meshgrid(T_grid, W_grid)

        W_output_grid = griddata(
            (T, W_input), W_output, (T_mesh, W_mesh), method='cubic'
        )

        dT = T_grid[1] - T_grid[0]
        dW = W_grid[1] - W_grid[0]

        laplacian = (
            np.roll(W_output_grid,  1, axis=0) +
            np.roll(W_output_grid, -1, axis=0) +
            np.roll(W_output_grid,  1, axis=1) +
            np.roll(W_output_grid, -1, axis=1) -
            4 * W_output_grid
        ) / (dT * dW)

        laplacian      = laplacian[1:-1, 1:-1]
        laplacian_abs  = np.abs(laplacian[~np.isnan(laplacian)])
        laplacian_p99  = np.percentile(laplacian_abs, 99)
        data_range     = W_output.max() - W_output.min()
        eta            = laplacian_p99 / (data_range + 1e-8)

        if eta < 0.02:
            quality_level = 'Excellent'
        elif eta < 0.04:
            quality_level = 'Good'
        elif eta < 0.08:
            quality_level = 'Acceptable'
        elif eta < 0.15:
            quality_level = 'Poor'
        else:
            quality_level = 'Unacceptable'

        self.logger.info(f"Smoothness evaluation: η={eta:.4f} ({quality_level})")

        return {
            'eta':           float(eta),
            'laplacian_p99': float(laplacian_p99),
            'data_range':    float(data_range),
            'quality_level': quality_level,
        }

    def evaluate_full(
        self,
        dnn_model: nn.Module,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        predicted_data: np.ndarray,
        device: str = 'cuda',
    ) -> Tuple[float, Dict]:
        """Full physical consistency evaluation (requires DNN model and scalers).

        Args:
            dnn_model: Trained DNN model.
            x_scaler: Input scaler.
            y_scaler: Output scaler.
            predicted_data: Predicted data, shape (N, 3), columns [T, W_input, W_output].
            device: Compute device.

        Returns:
            (overall_score, results_dict).
        """
        self.logger.info("Starting full physical consistency evaluation")

        def predict_fn(X: np.ndarray) -> np.ndarray:
            dnn_model.eval()
            with torch.no_grad():
                X_scaled      = x_scaler.transform(X)
                X_tensor      = torch.FloatTensor(X_scaled).to(device)
                y_pred_scaled = dnn_model(X_tensor).cpu().numpy()
                return y_scaler.inverse_transform(y_pred_scaled).flatten()

        boundary_results = self.boundary_evaluator.evaluate_boundaries(predict_fn)
        boundary_error   = boundary_results['input_nrmse'] + boundary_results['output_nrmse']
        boundary_score   = exponential_decay_score(boundary_error, self.boundary_decay_lambda)

        smoothness_results = self.evaluate_smoothness(predicted_data)
        smoothness_score   = exponential_decay_score(
            smoothness_results['eta'], self.smoothness_decay_lambda
        )

        overall_score = 0.5 * boundary_score + 0.5 * smoothness_score

        self.logger.info(f"Boundary score: {boundary_score:.6f}")
        self.logger.info(f"Smoothness score: {smoothness_score:.6f}")
        self.logger.info(f"Overall score: {overall_score:.6f}")

        results = {
            'overall_score':    float(overall_score),
            'boundary_score':   float(boundary_score),
            'smoothness_score': float(smoothness_score),
            'boundary_error':   float(boundary_error),
            'smoothness_eta':   float(smoothness_results['eta']),
            'boundary_details':   boundary_results,
            'smoothness_details': smoothness_results,
        }
        return overall_score, results

    def evaluate_with_predictor(
        self,
        predict_fn: Callable,
        predicted_data: np.ndarray,
    ) -> Tuple[float, Dict]:
        """Simplified evaluation using a predictor function interface.

        Args:
            predict_fn: Function that accepts a [T, W_input] array and returns W_output predictions.
            predicted_data: Predicted data, shape (N, 3).

        Returns:
            (overall_score, results_dict).
        """
        boundary_results = self.boundary_evaluator.evaluate_boundaries(predict_fn)
        boundary_error   = boundary_results['input_nrmse'] + boundary_results['output_nrmse']
        boundary_score   = exponential_decay_score(boundary_error, self.boundary_decay_lambda)

        smoothness_results = self.evaluate_smoothness(predicted_data)
        smoothness_score   = exponential_decay_score(
            smoothness_results['eta'], self.smoothness_decay_lambda
        )

        overall_score = 0.5 * boundary_score + 0.5 * smoothness_score

        results = {
            'overall_score':    float(overall_score),
            'boundary_score':   float(boundary_score),
            'smoothness_score': float(smoothness_score),
            'boundary_error':   float(boundary_error),
            'smoothness_eta':   float(smoothness_results['eta']),
            'boundary_details':   boundary_results,
            'smoothness_details': smoothness_results,
        }
        return overall_score, results

    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate a formatted evaluation report string."""
        bd = results['boundary_details']
        sd = results['smoothness_details']
        lines = [
            '=' * 70,
            'Physical Consistency Evaluation Report',
            '=' * 70,
            '',
            'Overall Evaluation',
            '-' * 70,
            f"Overall score:          {results['overall_score']:.6f}",
            '',
            'Boundary Consistency',
            '-' * 70,
            f"Boundary score:         {results['boundary_score']:.6f}",
            f"Total boundary error:   {results['boundary_error']:.4f}",
            f"  Input-side NRMSE:     {bd['input_nrmse']:.4f}",
            f"  Output-side NRMSE:    {bd['output_nrmse']:.4f}",
            '',
            'Thermodynamic Smoothness',
            '-' * 70,
            f"Smoothness score:       {results['smoothness_score']:.6f}",
            f"η (normalised curvature): {results['smoothness_eta']:.4f}",
            f"Smoothness level:       {sd['quality_level']}",
            '',
            '=' * 70,
        ]
        return '\n'.join(lines)


class TSTREvaluator:
    """TSTR (Train on Synthetic, Test on Real) evaluator.

    Training strategy
    -----------------
    - Train for ``config.tstr_epochs`` epochs (default 1000).
    - After each epoch, compute validation MSE on the original scale;
      save state_dict in memory if it is below the historical minimum
      (recorded from epoch 1, no start_epoch threshold).
    - After training, load the best checkpoint and perform final evaluation
      on train / val / test sets.
    """

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        config: PhysicsConfig = None,
    ):
        if config is None:
            config = PhysicsConfig()

        self.config  = config
        self.X_val   = X_val
        self.y_val   = y_val
        self.X_test  = X_test
        self.y_test  = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.logger  = get_logger(self.__class__.__name__, config.log_level)

        self.logger.info("TSTREvaluator initialized")
        self.logger.debug(f"Val size: {len(X_val)}, test size: {len(X_test)}")

    def evaluate(
        self,
        X_syn: np.ndarray,
        y_syn: np.ndarray,
        epochs: int = None,
        random_seed: int = 42,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train a DNN on synthetic data and evaluate on real data.

        Checkpoint criterion: save when val MSE (original scale) decreases,
        recorded from epoch 1, no epoch threshold.

        Args:
            X_syn: Synthetic training features, shape (N_syn, D).
            y_syn: Synthetic training targets, shape (N_syn,).
            epochs: Number of training epochs; uses config default when None.
            random_seed: Random seed.
            verbose: Whether to display the training progress bar.

        Returns:
            Dict containing metrics, history, model, x_scaler, y_scaler,
            predictions, true_values, inputs, best_epoch, best_val_mse.
        """
        if epochs is None:
            epochs = self.config.tstr_epochs

        self.logger.info(f"Starting TSTR evaluation: n_synthetic={len(X_syn)}, epochs={epochs}")

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

        device = torch.device(
            self.config.tstr_device if torch.cuda.is_available() else 'cpu'
        )
        self.logger.debug(f"Compute device: {device}")

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_syn_scaled  = x_scaler.fit_transform(X_syn)
        y_syn_scaled  = y_scaler.fit_transform(y_syn.reshape(-1, 1)).flatten()
        X_val_scaled  = x_scaler.transform(self.X_val)
        X_test_scaled = x_scaler.transform(self.X_test)

        g = torch.Generator()
        g.manual_seed(random_seed)

        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_syn_scaled),
                torch.FloatTensor(y_syn_scaled),
            ),
            batch_size=self.config.tstr_batch_size,
            shuffle=True,
            generator=g,
        )

        model = DNN(
            in_dim=X_syn.shape[1],
            out_dim=1,
            layer_dim=self.config.dnn_layer_dim,
            node_dim=self.config.dnn_node_dim,
        ).to(device)

        self.logger.debug(f"Network: {self.config.dnn_layer_dim} layers, "
                          f"{self.config.dnn_node_dim} nodes per layer")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.tstr_lr)
        criterion = nn.MSELoss()

        # ── Checkpoint state (val MSE decreases, recorded from epoch 1) ──────
        best_val_mse     = float('inf')
        best_model_state = None
        best_epoch       = 0

        history = {'train_r2': [], 'val_r2': [], 'val_loss': [], 'test_r2': []}

        pbar = tqdm(range(epochs), desc="TSTR Training") if verbose else range(epochs)

        for epoch in pbar:
            # ── Train ──
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()

            # ── Evaluate (original scale) ──
            model.eval()
            with torch.no_grad():
                X_tr_t = torch.FloatTensor(X_syn_scaled).to(device)
                y_tr_p = y_scaler.inverse_transform(
                    model(X_tr_t).cpu().numpy()
                ).flatten()
                train_r2 = r2_score(y_syn, y_tr_p)

                X_va_t = torch.FloatTensor(X_val_scaled).to(device)
                y_va_p = y_scaler.inverse_transform(
                    model(X_va_t).cpu().numpy()
                ).flatten()
                val_r2  = r2_score(self.y_val, y_va_p)
                val_mse = float(mean_squared_error(self.y_val, y_va_p))

                X_te_t = torch.FloatTensor(X_test_scaled).to(device)
                y_te_p = y_scaler.inverse_transform(
                    model(X_te_t).cpu().numpy()
                ).flatten()
                test_r2 = r2_score(self.y_test, y_te_p)

            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)
            history['val_loss'].append(val_mse)
            history['test_r2'].append(test_r2)

            # ── Checkpoint: save when val MSE decreases, from epoch 1 ──
            if val_mse < best_val_mse:
                best_val_mse     = val_mse
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch       = epoch + 1  # 1-indexed

            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'train_r2': f'{train_r2:.4f}',
                    'val_r2':   f'{val_r2:.4f}',
                    'val_mse':  f'{val_mse:.6f}',
                    'best':     f'ep{best_epoch}',
                })

        # ── Restore best weights (guaranteed, as epoch 1 always triggers) ─────
        model.load_state_dict(best_model_state)
        self.logger.info(
            f"Best model loaded: epoch {best_epoch} (val MSE={best_val_mse:.6f})"
        )

        # ── Final inference (using best weights) ──
        model.eval()
        with torch.no_grad():
            y_train_final = y_scaler.inverse_transform(
                model(torch.FloatTensor(X_syn_scaled).to(device)).cpu().numpy()
            ).flatten()
            y_val_final = y_scaler.inverse_transform(
                model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
            ).flatten()
            y_test_final = y_scaler.inverse_transform(
                model(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
            ).flatten()

        metrics_result = {
            'train_r2':   float(r2_score(y_syn, y_train_final)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_syn, y_train_final))),
            'train_mae':  float(mean_absolute_error(y_syn, y_train_final)),
            'val_r2':     float(r2_score(self.y_val, y_val_final)),
            'val_rmse':   float(np.sqrt(mean_squared_error(self.y_val, y_val_final))),
            'val_mae':    float(mean_absolute_error(self.y_val, y_val_final)),
            'test_r2':    float(r2_score(self.y_test, y_test_final)),
            'test_rmse':  float(np.sqrt(mean_squared_error(self.y_test, y_test_final))),
            'test_mae':   float(mean_absolute_error(self.y_test, y_test_final)),
        }

        print("\n" + "=" * 70)
        print("TSTR Evaluation Results")
        print("=" * 70)
        print(f"\nTrain set metrics:")
        print(f"  R²:   {metrics_result['train_r2']:.6f}")
        print(f"  RMSE: {metrics_result['train_rmse']:.6f}")
        print(f"  MAE:  {metrics_result['train_mae']:.6f}")
        print(f"\nValidation set metrics:")
        print(f"  R²:   {metrics_result['val_r2']:.6f}")
        print(f"  RMSE: {metrics_result['val_rmse']:.6f}")
        print(f"  MAE:  {metrics_result['val_mae']:.6f}")
        print(f"\nTest set metrics:")
        print(f"  R²:   {metrics_result['test_r2']:.6f}")
        print(f"  RMSE: {metrics_result['test_rmse']:.6f}")
        print(f"  MAE:  {metrics_result['test_mae']:.6f}")
        print(f"\nBest model info:")
        print(f"  Epoch:        {best_epoch}")
        print(f"  Val MSE:      {best_val_mse:.6f}")
        print("=" * 70 + "\n")

        self.logger.info("TSTR evaluation complete")

        return {
            'metrics':     metrics_result,
            'history':     history,
            'model':       model,
            'x_scaler':    x_scaler,
            'y_scaler':    y_scaler,
            'predictions': {
                'train': y_train_final,
                'val':   y_val_final,
                'test':  y_test_final,
            },
            'true_values': {
                'train': y_syn,
                'val':   self.y_val,
                'test':  self.y_test,
            },
            'inputs': {
                'train': X_syn,
                'val':   self.X_val,
                'test':  self.X_test,
            },
            'n_synthetic': len(X_syn),
            'epochs':      epochs,
            'best_epoch':  best_epoch,
            'best_val_mse': float(best_val_mse),
        }

    def evaluate_with_synthetic(
        self,
        X_syn: np.ndarray,
        y_syn: np.ndarray,
        epochs: int = None,
    ) -> Dict[str, Any]:
        """Simplified alias interface for evaluate()."""
        return self.evaluate(X_syn, y_syn, epochs=epochs, verbose=False)

    def save_predictions_to_excel(self, result: Dict, filepath: str):
        """Save prediction results to an Excel file (val / test / train sheets)."""
        val_df = pd.DataFrame({
            'Input_T':    result['inputs']['val'][:, 0],
            'Input_W':    result['inputs']['val'][:, 1],
            'True_value': self.y_val,
            'Predicted':  result['predictions']['val'],
            'Error':      self.y_val - result['predictions']['val'],
        })
        test_df = pd.DataFrame({
            'Input_T':    result['inputs']['test'][:, 0],
            'Input_W':    result['inputs']['test'][:, 1],
            'True_value': self.y_test,
            'Predicted':  result['predictions']['test'],
            'Error':      self.y_test - result['predictions']['test'],
        })
        train_df = pd.DataFrame({
            'Input_T':    result['inputs']['train'][:, 0],
            'Input_W':    result['inputs']['train'][:, 1],
            'True_value': result['true_values']['train'],
            'Predicted':  result['predictions']['train'],
            'Error':      result['true_values']['train'] - result['predictions']['train'],
        })
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            val_df.to_excel(writer,   sheet_name='Validation', index=False)
            test_df.to_excel(writer,  sheet_name='Test',       index=False)
            train_df.to_excel(writer, sheet_name='Train',      index=False)

        self.logger.info(f"Predictions saved to: {filepath}")

    def save_metrics_to_excel(
        self,
        tstr_result: Dict,
        physics_results: Dict,
        filepath: str,
    ):
        """Save evaluation metrics to an Excel file."""
        m = tstr_result['metrics']
        df = pd.DataFrame({
            'Metric': [
                'Train R²', 'Train RMSE', 'Train MAE',
                'Val R²',   'Val RMSE',   'Val MAE',
                'Test R²',  'Test RMSE',  'Test MAE',
                'Best epoch', 'Best val MSE',
                'Boundary score', 'Smoothness score', 'Overall physics score',
                'N synthetic', 'Epochs',
            ],
            'Value': [
                m['train_r2'],  m['train_rmse'],  m['train_mae'],
                m['val_r2'],    m['val_rmse'],     m['val_mae'],
                m['test_r2'],   m['test_rmse'],    m['test_mae'],
                tstr_result['best_epoch'],
                tstr_result['best_val_mse'],
                physics_results['boundary_score'],
                physics_results['smoothness_score'],
                physics_results['overall_score'],
                tstr_result['n_synthetic'],
                tstr_result['epochs'],
            ],
        })
        df.to_excel(filepath, index=False)
        self.logger.info(f"Metrics saved to: {filepath}")


def evaluate_dnn_phase_diagram(
    predicted_data_array: np.ndarray,
    input_boundary_model: LowDimEnsemble,
    output_boundary_model: LowDimEnsemble,
    dnn_model: nn.Module = None,
    x_scaler: StandardScaler = None,
    y_scaler: StandardScaler = None,
    predict_fn: Callable = None,
    device: str = 'cuda',
    save_report: bool = True,
    report_path: str = 'evaluation_report.txt',
    boundary_decay_lambda: float = 5.0,
    smoothness_decay_lambda: float = 15.0,
    log_level: int = logging.INFO,
) -> Tuple[float, Dict]:
    """Convenience entry point for phase diagram physical consistency evaluation.

    Args:
        predicted_data_array: Predicted data, shape (N, 3), columns [T, W_input, W_output].
        input_boundary_model: Input-side low-dimensional system boundary model.
        output_boundary_model: Output-side low-dimensional system boundary model.
        dnn_model: DNN model (used together with x_scaler and y_scaler when provided).
        x_scaler: Input scaler.
        y_scaler: Output scaler.
        predict_fn: Prediction function (used when dnn_model is not provided).
        device: Compute device.
        save_report: Whether to save the evaluation report to a file.
        report_path: Report save path.
        boundary_decay_lambda: Boundary score decay coefficient.
        smoothness_decay_lambda: Smoothness score decay coefficient.
        log_level: Logging level.

    Returns:
        (overall_score, detailed_results_dict).
    """
    eval_logger = get_logger(__name__, log_level)
    eval_logger.info("Starting phase diagram physical consistency evaluation")

    config = PhysicsConfig(log_level=log_level)
    evaluator = PhysicalConsistencyEvaluator(
        input_boundary_model=input_boundary_model,
        output_boundary_model=output_boundary_model,
        config=config,
        boundary_decay_lambda=boundary_decay_lambda,
        smoothness_decay_lambda=smoothness_decay_lambda,
    )

    if dnn_model is not None and x_scaler is not None and y_scaler is not None:
        score, results = evaluator.evaluate_full(
            dnn_model, x_scaler, y_scaler, predicted_data_array, device
        )
    elif predict_fn is not None:
        score, results = evaluator.evaluate_with_predictor(
            predict_fn, predicted_data_array
        )
    else:
        raise ValueError("Must provide either (dnn_model, x_scaler, y_scaler) or predict_fn")

    if save_report:
        report = evaluator.generate_evaluation_report(results)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        eval_logger.info(f"Evaluation report saved: {report_path}")

    return score, results


__all__ = [
    'PhysicsConfig',
    'get_logger',
    'load_low_dim_model',
    'load_boundary_models',
    'load_ternary_data',
    'split_data_three_way',
    'DNN',
    'FcBlock',
    'DNNBoundaryEvaluator',
    'PhysicalConsistencyEvaluator',
    'TSTREvaluator',
    'evaluate_dnn_phase_diagram',
    'calculate_boundary_nrmse',
    'exponential_decay_score',
]