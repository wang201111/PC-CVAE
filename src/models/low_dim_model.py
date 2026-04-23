"""
Low-dimensional subsystem property prediction model.

Bootstrap ensemble DNN that takes temperature as input to predict
single-component system solubility. Normalization parameters are stored
as PyTorch buffers, enabling pure GPU inference.
"""

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LowDimConfig:
    """Hyperparameter configuration for the low-dimensional subsystem model."""

    # Model architecture
    HIDDEN_DIMS: List[int] = None
    DROPOUT: float = 0.1
    ACTIVATION: str = 'relu'
    USE_BATCH_NORM: bool = False

    # Training hyperparameters
    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 1000
    EARLY_STOP_PATIENCE: int = 50
    WEIGHT_DECAY: float = 0.0

    # Learning rate scheduling
    USE_LR_SCHEDULER: bool = True
    LR_SCHEDULER_TYPE: str = 'cosine'
    LR_SCHEDULER_FACTOR: float = 0.5
    LR_SCHEDULER_PATIENCE: int = 10
    LR_MIN: float = 1e-5

    # Ensemble configuration
    USE_ENSEMBLE: bool = True
    N_ENSEMBLE: int = 5
    BOOTSTRAP_RATIO: float = 0.8

    # Data processing
    VALIDATION_SPLIT: float = 0.15
    SHUFFLE_DATA: bool = True
    NORMALIZE_INPUT: bool = True
    NORMALIZE_OUTPUT: bool = True

    # Optimizer and initialization
    OPTIMIZER: str = 'adam'
    WEIGHT_INIT: str = 'kaiming'

    # Miscellaneous
    RANDOM_SEED: int = 42
    DEVICE: str = 'auto'

    def __post_init__(self):
        if self.HIDDEN_DIMS is None:
            self.HIDDEN_DIMS = [256, 512, 512, 256]

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unknown configuration parameter: {key}")

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


ACTIVATION_MAP = {
    'relu':       nn.ReLU(),
    'gelu':       nn.GELU(),
    'elu':        nn.ELU(),
    'tanh':       nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(0.2),
    'silu':       nn.SiLU(),
}


class BaseDNN(nn.Module):
    """Fully connected feedforward network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        activation: str = 'relu',
        dropout: float = 0.1,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            if activation in ACTIVATION_MAP:
                layers.append(ACTIVATION_MAP[activation])
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module, method: str = 'kaiming'):
    """Initialize network weights."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if method == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif method == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif method == 'normal':
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def _train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: LowDimConfig,
    device: torch.device,
    verbose: bool = False,
) -> Dict:
    """Train a single sub-model and return training history."""

    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
    elif config.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY,
        )

    scheduler = None
    if config.USE_LR_SCHEDULER:
        if config.LR_SCHEDULER_TYPE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.N_EPOCHS, eta_min=config.LR_MIN
            )
        elif config.LR_SCHEDULER_TYPE == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.LR_SCHEDULER_FACTOR,
                patience=config.LR_SCHEDULER_PATIENCE,
                verbose=verbose,
            )
        elif config.LR_SCHEDULER_TYPE == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=100, gamma=0.5
            )

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.N_EPOCHS):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                val_losses.append(nn.MSELoss()(model(X_batch), y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if scheduler is not None:
            if config.LR_SCHEDULER_TYPE == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            if verbose:
                print(f"    Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    return history


class LowDimEnsemble(nn.Module):
    """Ensemble prediction model for low-dimensional subsystem properties.

    Normalization parameters are stored via register_buffer, so they migrate
    automatically with the model device and are saved/restored in state_dict
    without maintaining a separate scaler object.
    predict_torch() runs entirely on GPU and can be called directly by the
    physics-constrained loss functions.
    """

    def __init__(self, input_dim: int, config: Optional[LowDimConfig] = None):
        super().__init__()

        self.input_dim = input_dim
        self.config = config if config is not None else LowDimConfig()

        self.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.config.DEVICE == 'auto'
            else torch.device(self.config.DEVICE)
        )

        self.n_models = self.config.N_ENSEMBLE if self.config.USE_ENSEMBLE else 1

        self.models = nn.ModuleList([
            BaseDNN(
                input_dim=input_dim,
                hidden_dims=self.config.HIDDEN_DIMS,
                output_dim=1,
                activation=self.config.ACTIVATION,
                dropout=self.config.DROPOUT,
                use_batch_norm=self.config.USE_BATCH_NORM,
            )
            for _ in range(self.n_models)
        ])

        for model in self.models:
            initialize_weights(model, self.config.WEIGHT_INIT)

        self.to(self.device)

        # Normalization parameters stored as buffers; migrate automatically with model device
        self.register_buffer('x_mean',  torch.zeros(input_dim))
        self.register_buffer('x_scale', torch.ones(input_dim))
        self.register_buffer('y_mean',  torch.zeros(1))
        self.register_buffer('y_scale', torch.ones(1))

        self.is_scaler_fitted = False
        self.is_fitted = False

    def _fit_scalers(self, X: np.ndarray, y: np.ndarray):
        """Compute normalization parameters with sklearn and store them in buffers.
        No scaler object is retained after fitting."""
        if self.config.NORMALIZE_INPUT:
            sx = StandardScaler().fit(X)
            self.x_mean.data  = torch.from_numpy(sx.mean_).float().to(self.device)
            self.x_scale.data = torch.from_numpy(sx.scale_).float().to(self.device)
        else:
            self.x_mean.data  = torch.zeros(self.input_dim, device=self.device)
            self.x_scale.data = torch.ones(self.input_dim,  device=self.device)

        if self.config.NORMALIZE_OUTPUT:
            sy = StandardScaler().fit(y)
            self.y_mean.data  = torch.from_numpy(sy.mean_).float().to(self.device)
            self.y_scale.data = torch.from_numpy(sy.scale_).float().to(self.device)
        else:
            self.y_mean.data  = torch.zeros(1, device=self.device)
            self.y_scale.data = torch.ones(1,  device=self.device)

        self.is_scaler_fitted = True

    def _transform_input(self, X: np.ndarray) -> np.ndarray:
        if not self.is_scaler_fitted or not self.config.NORMALIZE_INPUT:
            return X
        return (X - self.x_mean.cpu().numpy()) / (self.x_scale.cpu().numpy() + 1e-8)

    def _transform_output(self, y: np.ndarray) -> np.ndarray:
        if not self.is_scaler_fitted or not self.config.NORMALIZE_OUTPUT:
            return y
        return (y - self.y_mean.cpu().numpy()) / (self.y_scale.cpu().numpy() + 1e-8)

    def _inverse_transform_output(self, y_scaled: np.ndarray) -> np.ndarray:
        if not self.is_scaler_fitted or not self.config.NORMALIZE_OUTPUT:
            return y_scaled
        return y_scaled * self.y_scale.cpu().numpy() + self.y_mean.cpu().numpy()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Dict:
        """Train the ensemble model."""
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        self._fit_scalers(X_train, y_train)
        X_scaled = self._transform_input(X_train)
        y_scaled = self._transform_output(y_train)

        if X_val is None or y_val is None:
            n_val = int(len(X_scaled) * self.config.VALIDATION_SPLIT)
            idx = (
                np.random.permutation(len(X_scaled))
                if self.config.SHUFFLE_DATA
                else np.arange(len(X_scaled))
            )
            X_val_sc = X_scaled[idx[:n_val]]
            y_val_sc = y_scaled[idx[:n_val]]
            X_scaled  = X_scaled[idx[n_val:]]
            y_scaled  = y_scaled[idx[n_val:]]
        else:
            if X_val.ndim == 1:
                X_val = X_val.reshape(-1, 1)
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            X_val_sc = self._transform_input(X_val)
            y_val_sc = self._transform_output(y_val)

        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val_sc), torch.FloatTensor(y_val_sc)),
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
        )

        all_histories = []

        for i, model in enumerate(self.models):
            if verbose:
                print(f"\n  Training sub-model {i + 1}/{self.n_models}...")

            if self.config.USE_ENSEMBLE:
                n_boot = int(len(X_scaled) * self.config.BOOTSTRAP_RATIO)
                boot_idx = np.random.choice(len(X_scaled), n_boot, replace=True)
                X_b, y_b = X_scaled[boot_idx], y_scaled[boot_idx]
            else:
                X_b, y_b = X_scaled, y_scaled

            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_b), torch.FloatTensor(y_b)),
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
            )

            history = _train_single_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=self.config,
                device=self.device,
                verbose=verbose,
            )
            all_histories.append(history)

            if verbose:
                print(f"    Best validation loss: {min(history['val_loss']):.6f}")

        self.is_fitted = True
        return {'histories': all_histories, 'n_models': self.n_models}

    def predict_torch(
        self,
        X_tensor: torch.Tensor,
        return_std: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU inference interface for direct use by physics-constrained loss functions.

        Args:
            X_tensor: Input tensor of shape (N, input_dim).
            return_std: Whether to return ensemble standard deviation.

        Returns:
            y_mean: Ensemble mean in physical space, shape (N, 1).
            y_std:  Ensemble std in physical space, shape (N, 1); None if return_std=False.
            preds_scaled: Per-sub-model predictions in normalized space, shape (n_models, N, 1).
        """
        self.eval()

        if X_tensor.device != self.device:
            X_tensor = X_tensor.to(self.device)

        with torch.no_grad():
            if self.is_scaler_fitted and self.config.NORMALIZE_INPUT:
                X_scaled = (X_tensor - self.x_mean) / (self.x_scale + 1e-8)
            else:
                X_scaled = X_tensor

            preds_scaled = torch.stack([m(X_scaled) for m in self.models])

            y_mean_sc = preds_scaled.mean(dim=0)
            y_std_sc  = preds_scaled.std(dim=0)

            if self.is_scaler_fitted and self.config.NORMALIZE_OUTPUT:
                y_mean = y_mean_sc * self.y_scale + self.y_mean
                y_std  = y_std_sc  * self.y_scale  # std is only affected by scale
            else:
                y_mean = y_mean_sc
                y_std  = y_std_sc

        if return_std:
            return y_mean, y_std, preds_scaled
        else:
            return y_mean, None, preds_scaled

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """NumPy interface; internally calls predict_torch."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_mean, y_std, _ = self.predict_torch(X_tensor, return_std=True)
        if return_std:
            return y_mean.cpu().numpy(), y_std.cpu().numpy()
        return y_mean.cpu().numpy()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X, return_std=False)

    def compute_confidence(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = 'r_squared',
    ) -> float:
        """Compute model fit confidence."""
        if method == 'uniform':
            return 1.0

        y_pred = self.predict(X_val)

        if method == 'r_squared':
            from sklearn.metrics import r2_score
            return float(np.clip(r2_score(y_val, y_pred), 0.0, 1.0))
        elif method == 'rmse_based':
            rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
            y_range = y_val.max() - y_val.min()
            return float(np.clip(1.0 - rmse / y_range, 0.0, 1.0))
        else:
            raise ValueError(f"Unknown confidence computation method: {method}")

    def save(self, path: str):
        """Save model weights and configuration (normalization buffers included in state_dict)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config':           self.config.to_dict(),
            'input_dim':        self.input_dim,
            'n_models':         self.n_models,
            'is_fitted':        self.is_fitted,
            'is_scaler_fitted': self.is_scaler_fitted,
            'state_dict':       self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str) -> 'LowDimEnsemble':
        """Restore model from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        ckpt = torch.load(path, map_location='cpu')
        config = LowDimConfig(**ckpt['config'])
        ensemble = cls(input_dim=ckpt['input_dim'], config=config)
        ensemble.load_state_dict(ckpt['state_dict'])
        ensemble.is_fitted        = ckpt['is_fitted']
        ensemble.is_scaler_fitted = ckpt['is_scaler_fitted']
        ensemble.to(ensemble.device)
        return ensemble


def prepare_low_dim_predictor(
    system_name: str,
    data_path: str,
    model_path: str,
    force_retrain: bool = False,
    verbose: bool = False,
    config_params: Optional[Dict] = None,
) -> Callable:
    """Load or train a low-dimensional subsystem prediction model.

    If the model file already exists and force_retrain=False, the model is
    loaded directly; otherwise it is trained from the data file and saved.

    Args:
        system_name:   System name for logging.
        data_path:     Path to Excel data file; first column is T, second is solubility.
        model_path:    Path for saving or loading the model.
        force_retrain: Force retraining even if a saved model exists.
        verbose:       Print training progress.
        config_params: Dictionary of parameters to override the default configuration.

    Returns:
        Trained LowDimEnsemble instance; returns None if loading or training fails.
    """
    model_path = Path(model_path)
    data_path  = Path(data_path)

    config = LowDimConfig()
    if config_params is not None:
        config.update(**config_params)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Low-dimensional subsystem prediction model: {system_name}")
        print(f"  Data path:      {data_path}")
        print(f"  Model path:     {model_path}")
        print(f"  Force retrain:  {force_retrain}")

    need_train = force_retrain or not model_path.exists()

    if not need_train:
        if verbose:
            print("\nLoading existing model...")
        try:
            ensemble = LowDimEnsemble.load(str(model_path))
            if verbose:
                print(f"  Model loaded successfully. Number of sub-models: {ensemble.n_models}")
            return ensemble
        except Exception as e:
            if verbose:
                print(f"  Loading failed: {e}. Retraining.")
            need_train = True

    if need_train:
        if verbose:
            print("\nTraining new model...")
        try:
            import pandas as pd
            df = pd.read_excel(data_path)
            if len(df.columns) < 2:
                raise ValueError("Data file must contain at least two columns (T, solubility).")
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)
            if verbose:
                print(f"  Number of samples: {len(X)}")
                print(f"  T range:           [{X.min():.2f}, {X.max():.2f}]")
                print(f"  Solubility range:  [{y.min():.4f}, {y.max():.4f}]")
        except Exception as e:
            print(f"Data loading failed: {e}")
            return None

        ensemble = LowDimEnsemble(input_dim=X.shape[1], config=config)
        try:
            ensemble.fit(X, y, verbose=verbose)
            ensemble.save(str(model_path))
            if verbose:
                print(f"  Model saved to: {model_path}")
            return ensemble
        except Exception as e:
            print(f"Training failed: {e}")
            return None


__all__ = [
    'LowDimConfig',
    'BaseDNN',
    'LowDimEnsemble',
    'prepare_low_dim_predictor',
]