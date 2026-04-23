"""
Physics-Constrained Conditional Variational Autoencoder (PC-CVAE) — Solubility version.

Designed for thermodynamic manifold modelling of ternary aqueous salt systems,
with temperature T as the conditioning variable.
Boundary neighborhoods of the latent variable z are anchored to low-dimensional
subsystem models via collocation constraints.
Depends on LowDimEnsemble from low_dim_model.py as the boundary model interface.

Components
----------
phi (Inverse Manifold Mapping structure)
    Lightweight MLP with inputs (T_norm, W1_norm) and output z_tilde ≈ mu.
    At inference, z_tilde replaces random sampling to achieve deterministic prediction.

L_cycle (prior self-sampling cycle consistency loss)
    Samples z_rand from a uniform prior, covering the full temperature range
    (including the high-temperature extrapolation domain).
    A frozen Decoder generates synthetic observables; phi then inversely infers z_tilde.
    Minimises ||z_tilde - z_rand||^2, enabling phi to learn a global inverse mapping
    without real high-temperature labels. Decoder gradients are fully detached,
    leaving the collocation constraint unaffected.

Public interface
----------------
predict(X)  — deterministic inference via z = phi(T, W1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from low_dim_model import LowDimEnsemble


# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

@dataclass
class CVAEConfig:
    """CVAE hyperparameter configuration."""

    LATENT_DIM: int = 2
    HIDDEN_DIMS: List[int] = field(default_factory=lambda: [128, 256, 256, 128])
    DROPOUT: float = 0.1

    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 200
    WEIGHT_DECAY: float = 1e-5

    LAMBDA_KL: float = 1.0
    LAMBDA_COLLOCATION_Na2SO4: float = 3.0
    LAMBDA_COLLOCATION_MgSO4: float = 0.01

    N_COLLOCATION_POINTS: int = 64
    COLLOCATION_T_RANGE: Optional[Tuple[float, float]] = None

    Z_LOW: float = -2.0
    Z_HIGH: float = 2.0
    Z_COLLOC_WIDTH: float = 0.5

    PHI_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [64, 64])
    LAMBDA_CYCLE: float = 1.0
    N_CYCLE_POINTS: int = 64
    # Temperature sampling range for cycle loss; should be set explicitly to the full
    # physical temperature range (including high-temperature extrapolation domain),
    # e.g. (-34.0, 200.0). Defaults to the training temperature range when None.
    CYCLE_T_RANGE: Optional[Tuple[float, float]] = None

    USE_EARLY_STOPPING: bool = True
    EARLY_STOP_PATIENCE: int = 20
    USE_LR_SCHEDULER: bool = True
    LR_SCHEDULER_TYPE: str = 'cosine'
    LR_MIN: float = 1e-6
    DEVICE: str = 'auto'
    VERBOSE: bool = True

    def __post_init__(self):
        if self.DEVICE == 'auto':
            self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ──────────────────────────────────────────────
#  Low-dimensional model descriptor
# ──────────────────────────────────────────────

@dataclass
class LowDimInfo:
    """Descriptor for a low-dimensional subsystem model.

    Args:
        model: LowDimEnsemble instance; must implement predict_torch(T, return_std).
        name: System name identifier (used in logging).
        constraint_type: Boundary type.
            comp1 end accepts 'output'/'Na2SO4'/'na2so4'/'mgcl2'.
            comp2 end accepts 'input'/'MgSO4'/'mgso4'/'kcl'.
    """
    model: LowDimEnsemble
    name: str
    constraint_type: str


# ──────────────────────────────────────────────
#  Network
# ──────────────────────────────────────────────

class PhysicsConstrainedCVAE(nn.Module):
    """Physics-Constrained Conditional Variational Autoencoder with Inverse Manifold Mapping structure phi.

    Args:
        input_dim:       Encoder input dimension (T + W_comp1 + W_comp2 = 3).
        condition_dim:   Decoder conditioning dimension (currently 1, temperature T only).
        latent_dim:      Latent variable dimension, determined by the Gibbs phase rule F = C - P + 1.
        hidden_dims:     Hidden layer widths for encoder and decoder.
        phi_hidden_dims: Hidden layer widths for phi.
        dropout:         Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        phi_hidden_dims: List[int],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim     = input_dim
        self.condition_dim = condition_dim
        self.latent_dim    = latent_dim

        self.encoder   = self._build_mlp(input_dim, hidden_dims, dropout)
        self.fc_mean   = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        self.decoder = self._build_mlp(
            latent_dim + condition_dim, list(reversed(hidden_dims)), dropout
        )
        self.fc_out = nn.Linear(hidden_dims[0], 2)

        # phi: Inverse Manifold Mapping structure; inputs [T_norm, W1_norm], output z_tilde
        self.phi     = self._build_mlp(condition_dim + 1, phi_hidden_dims, dropout)
        self.phi_out = nn.Linear(phi_hidden_dims[-1], latent_dim)

        self.data_min:   Optional[torch.Tensor] = None
        self.data_max:   Optional[torch.Tensor] = None
        self.data_range: Optional[torch.Tensor] = None

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(T, W1, W2) -> (mu, log_var)."""
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """(z, T_norm) -> (W_comp1, W_comp2), shape (B, 2)."""
        h = self.decoder(torch.cat([z, conditions], dim=-1))
        return self.fc_out(h)

    def infer_z(self, phi_input: torch.Tensor) -> torch.Tensor:
        """phi inference: (T_norm, W1_norm) -> z_tilde, shape (B, latent_dim)."""
        return self.phi_out(self.phi(phi_input))

    def forward(
        self,
        x: Optional[torch.Tensor],
        conditions: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Training mode (x is not None) returns (output, mu, log_var);
        generation mode (x is None) samples from the prior and returns output.
        """
        if x is not None:
            mu, log_var = self.encode(x)
            z    = self._reparameterize(mu, log_var)
            cond = x[:, :self.condition_dim]
            return self.decode(z, cond), mu, log_var
        else:
            z = torch.randn(conditions.size(0), self.latent_dim, device=conditions.device)
            return self.decode(z, conditions)


# ──────────────────────────────────────────────
#  Loss functions
# ──────────────────────────────────────────────

class CVAELoss:
    """Composite CVAE loss function.

    Total loss:
        L = L_recon + lambda_KL * L_KL
            + lambda_Na2SO4 * L_colloc,1
            + lambda_MgSO4  * L_colloc,2
            + lambda_cycle  * L_cycle
    """

    _COMP1_TYPES = frozenset({'output', 'Na2SO4', 'na2so4', 'mgcl2'})
    _COMP2_TYPES = frozenset({'input',  'MgSO4',  'mgso4',  'kcl'})

    def __init__(self, model: PhysicsConstrainedCVAE, config: CVAEConfig):
        self.model  = model
        self.config = config
        self.device = next(model.parameters()).device

    def _collocation_loss(
        self,
        conditions: torch.Tensor,
        low_dim_list: List[LowDimInfo],
    ) -> Tuple[dict, dict]:
        """Compute boundary collocation losses at both ends."""
        zero = torch.tensor(0.0, device=self.device)
        if self.model.data_min is None or not low_dim_list:
            return {'colloc_Na2SO4': zero, 'colloc_MgSO4': zero}, {}

        device   = conditions.device
        T_min,  T_range  = self.model.data_min[0], self.model.data_range[0]
        W1_min, W1_range = self.model.data_min[1], self.model.data_range[1]
        W2_min, W2_range = self.model.data_min[2], self.model.data_range[2]

        n   = self.config.N_COLLOCATION_POINTS
        cfg = self.config
        z_lo_a, z_lo_b = cfg.Z_LOW, cfg.Z_LOW + cfg.Z_COLLOC_WIDTH
        z_hi_a, z_hi_b = cfg.Z_HIGH - cfg.Z_COLLOC_WIDTH, cfg.Z_HIGH

        if cfg.COLLOCATION_T_RANGE is not None:
            T_cmin, T_cmax = cfg.COLLOCATION_T_RANGE
        else:
            T_cmin, T_cmax = T_min.item(), (T_min + T_range).item()

        losses      = {'colloc_Na2SO4': zero.clone(), 'colloc_MgSO4': zero.clone()}
        diagnostics = {}

        for ti in low_dim_list:
            T_raw  = torch.rand(n, 1, device=device) * (T_cmax - T_cmin) + T_cmin
            T_norm = (T_raw - T_min) / T_range

            if ti.constraint_type in self._COMP1_TYPES:
                z   = torch.rand(n, self.model.latent_dim, device=device) \
                      * (z_lo_b - z_lo_a) + z_lo_a
                out = self.model.decode(z, T_norm)
                W1_pred = out[:, 0:1] * W1_range + W1_min
                W2_pred = out[:, 1:2] * W2_range + W2_min

                with torch.no_grad():
                    W2_target, _, _ = ti.model.predict_torch(T_raw, return_std=False)

                diff = W2_pred - W2_target
                loss = (W1_pred ** 2).mean() / W1_range ** 2 \
                     + (diff ** 2).mean()    / W2_range ** 2
                losses['colloc_Na2SO4'] = loss

                diagnostics.update({
                    f'{ti.name}_colloc_loss':      float(loss.item()),
                    f'{ti.name}_colloc_z_range':   f"[{z.min().item():.2f}, {z.max().item():.2f}]",
                    f'{ti.name}_colloc_T_range':   f"[{T_raw.min().item():.1f}, {T_raw.max().item():.1f}]",
                    f'{ti.name}_colloc_diff_mean': float(diff.mean().item()),
                    f'{ti.name}_colloc_diff_std':  float(diff.std().item()),
                })
                self._add_temperature_segment_mae(
                    diagnostics, ti.name,
                    T_raw.detach().cpu().numpy().flatten(),
                    diff.detach().cpu().numpy().flatten(),
                )

            elif ti.constraint_type in self._COMP2_TYPES:
                z   = torch.rand(n, self.model.latent_dim, device=device) \
                      * (z_hi_b - z_hi_a) + z_hi_a
                out = self.model.decode(z, T_norm)
                W1_pred = out[:, 0:1] * W1_range + W1_min
                W2_pred = out[:, 1:2] * W2_range + W2_min

                with torch.no_grad():
                    W1_target, _, _ = ti.model.predict_torch(T_raw, return_std=False)

                diff = W1_pred - W1_target
                loss = (diff ** 2).mean()    / W1_range ** 2 \
                     + (W2_pred ** 2).mean() / W2_range ** 2
                losses['colloc_MgSO4'] = loss

                diagnostics.update({
                    f'{ti.name}_colloc_loss':      float(loss.item()),
                    f'{ti.name}_colloc_z_range':   f"[{z.min().item():.2f}, {z.max().item():.2f}]",
                    f'{ti.name}_colloc_T_range':   f"[{T_raw.min().item():.1f}, {T_raw.max().item():.1f}]",
                    f'{ti.name}_colloc_pred_mean': float(W2_pred.mean().item()),
                    f'{ti.name}_colloc_pred_std':  float(W2_pred.std().item()),
                })
                self._add_temperature_segment_mae(
                    diagnostics, ti.name,
                    T_raw.detach().cpu().numpy().flatten(),
                    W2_pred.detach().cpu().numpy().flatten(),
                )

        return losses, diagnostics

    @staticmethod
    def _add_temperature_segment_mae(
        diag: dict, name: str, T_np: np.ndarray, residual_np: np.ndarray
    ) -> None:
        for T_lo, T_hi, label in [(-1e9, 0, 'cold'), (0, 100, 'mid'), (100, 1e9, 'hot')]:
            mask = (T_np >= T_lo) & (T_np < T_hi)
            if mask.sum() > 0:
                diag[f'{name}_colloc_mae_{label}'] = float(np.abs(residual_np[mask]).mean())

    def _cycle_loss(self) -> torch.Tensor:
        """Prior self-sampling cycle consistency loss.

        z_rand ~ U(Z_LOW, Z_HIGH); T_rand covers the full CYCLE_T_RANGE.
        W_fake = Decoder(z_rand, T_rand)  [gradients detached; Decoder not updated]
        z_tilde = phi(T_rand, W1_fake_norm)
        L_cycle = mean(||z_tilde - z_rand||^2)
        """
        if self.model.data_min is None:
            return torch.tensor(0.0, device=self.device)

        cfg    = self.config
        device = self.device
        n      = cfg.N_CYCLE_POINTS

        T_min,  T_range  = self.model.data_min[0], self.model.data_range[0]
        W1_min, W1_range = self.model.data_min[1], self.model.data_range[1]

        if cfg.CYCLE_T_RANGE is not None:
            T_cmin, T_cmax = cfg.CYCLE_T_RANGE
        else:
            T_cmin, T_cmax = T_min.item(), (T_min + T_range).item()

        z_rand = torch.rand(n, self.model.latent_dim, device=device) \
                 * (cfg.Z_HIGH - cfg.Z_LOW) + cfg.Z_LOW

        T_raw  = torch.rand(n, 1, device=device) * (T_cmax - T_cmin) + T_cmin
        T_norm = (T_raw - T_min) / T_range

        with torch.no_grad():
            W_fake      = self.model.decode(z_rand, T_norm)
            W1_fake_raw = W_fake[:, 0:1] * W1_range + W1_min

        W1_fake_norm = (W1_fake_raw - W1_min) / W1_range
        phi_input    = torch.cat([T_norm, W1_fake_norm], dim=-1)
        z_tilde      = self.model.infer_z(phi_input)

        return F.mse_loss(z_tilde, z_rand.detach())

    def compute_total_loss(
        self,
        output:       torch.Tensor,
        x_true:       torch.Tensor,
        z_mean:       torch.Tensor,
        z_logvar:     torch.Tensor,
        conditions:   torch.Tensor,
        low_dim_list: List[LowDimInfo],
    ) -> Tuple[torch.Tensor, dict]:
        """Compute total loss and return individual components."""
        B = x_true.size(0)
        L_recon = F.mse_loss(output, x_true)
        L_kl    = -0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp()) / B

        colloc_losses, boundary_diag = self._collocation_loss(conditions, low_dim_list)
        L_cycle = self._cycle_loss()

        loss_total = (
            L_recon
            + self.config.LAMBDA_KL                 * L_kl
            + self.config.LAMBDA_COLLOCATION_Na2SO4 * colloc_losses['colloc_Na2SO4']
            + self.config.LAMBDA_COLLOCATION_MgSO4  * colloc_losses['colloc_MgSO4']
            + self.config.LAMBDA_CYCLE              * L_cycle
        )

        return loss_total, {
            'total':         loss_total.item(),
            'recon':         L_recon.item(),
            'kl':            L_kl.item(),
            'colloc_Na2SO4': colloc_losses['colloc_Na2SO4'].item(),
            'colloc_MgSO4':  colloc_losses['colloc_MgSO4'].item(),
            'cycle':         L_cycle.item(),
            'boundary_diag': boundary_diag,
        }


# ──────────────────────────────────────────────
#  Training and inference interface
# ──────────────────────────────────────────────

class CVAEPhysicsModel:
    """PC-CVAE training and inference interface — solubility version.

    Encapsulates data preprocessing, training loop, learning rate scheduling,
    early stopping, and model persistence.

    Args:
        input_dim:     Input feature dimension (T + W_comp1 + W_comp2 = 3).
        condition_dim: Conditioning variable dimension (currently 1, temperature T only).
        config:        CVAEConfig instance.
    """

    def __init__(self, input_dim: int, condition_dim: int, config: CVAEConfig):
        self.input_dim     = input_dim
        self.condition_dim = condition_dim
        self.config        = config
        self.device        = torch.device(config.DEVICE)

        self.model = PhysicsConstrainedCVAE(
            input_dim       = input_dim,
            condition_dim   = condition_dim,
            latent_dim      = config.LATENT_DIM,
            hidden_dims     = config.HIDDEN_DIMS,
            phi_hidden_dims = config.PHI_HIDDEN_DIMS,
            dropout         = config.DROPOUT,
        ).to(self.device)

        self.scaler           = MinMaxScaler(feature_range=(0, 1))
        self.is_scaler_fitted = False
        self.data_min:   Optional[torch.Tensor] = None
        self.data_max:   Optional[torch.Tensor] = None
        self.data_range: Optional[torch.Tensor] = None

        self.loss_fn  = CVAELoss(self.model, config)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.scheduler = None
        if config.USE_LR_SCHEDULER and config.LR_SCHEDULER_TYPE == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.N_EPOCHS, eta_min=config.LR_MIN
            )

        self.history: dict = {'train': [], 'val': [], 'lr': []}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        low_dim_list: Optional[List[LowDimInfo]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict:
        """Fit the model.

        Args:
            X:            (N, 2) [T, W_comp1].
            y:            (N,) or (N, 1) W_comp2.
            low_dim_list: List of low-dimensional subsystem models; collocation loss
                          is skipped when None.
            X_val:        Validation set inputs, optional.
            y_val:        Validation set targets, optional.

        Returns:
            Training history dictionary with keys 'train', 'val', 'lr'.
        """
        X_full_raw    = np.column_stack([X, y])
        X_full_scaled = self.scaler.fit_transform(X_full_raw)
        self.is_scaler_fitted = True

        self.data_min   = torch.tensor(self.scaler.data_min_,  device=self.device).float()
        self.data_max   = torch.tensor(self.scaler.data_max_,  device=self.device).float()
        self.data_range = self.data_max - self.data_min
        self.model.data_min   = self.data_min
        self.model.data_max   = self.data_max
        self.model.data_range = self.data_range

        if self.config.VERBOSE:
            self._print_fit_info()

        X_t = torch.FloatTensor(X_full_scaled).to(self.device)
        y_t = torch.FloatTensor(X_full_scaled[:, 1:3]).to(self.device)
        train_loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(np.column_stack([X_val, y_val]))
            val_loader   = DataLoader(
                TensorDataset(
                    torch.FloatTensor(X_val_scaled).to(self.device),
                    torch.FloatTensor(X_val_scaled[:, 1:3]).to(self.device),
                ),
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
            )

        best_val_loss    = float('inf')
        patience_counter = 0

        for epoch in range(self.config.N_EPOCHS):
            train_losses = self._train_epoch(train_loader, low_dim_list)
            val_losses   = self._validate_epoch(val_loader, low_dim_list) \
                           if val_loader else None

            self.history['train'].append(train_losses)
            if val_losses is not None:
                self.history['val'].append(val_losses)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            if self.scheduler is not None:
                self.scheduler.step()

            if self.config.USE_EARLY_STOPPING and val_losses is not None:
                if val_losses['total'] < best_val_loss:
                    best_val_loss    = val_losses['total']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.EARLY_STOP_PATIENCE:
                        if self.config.VERBOSE:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            if self.config.VERBOSE and (epoch + 1) % 10 == 0:
                self._print_epoch(epoch, train_losses, val_losses)

        return self.history

    def _train_epoch(self, loader: DataLoader, low_dim_list) -> dict:
        self.model.train()
        sums = {
            'total': 0.0, 'recon': 0.0, 'kl': 0.0,
            'colloc_Na2SO4': 0.0, 'colloc_MgSO4': 0.0, 'cycle': 0.0,
        }
        diag_sum: dict = {}
        n = 0

        for X_batch, y_batch in loader:
            output, z_mean, z_logvar = self.model(X_batch)
            conditions = X_batch[:, :self.condition_dim]

            loss_total, losses = self.loss_fn.compute_total_loss(
                output, y_batch, z_mean, z_logvar, conditions, low_dim_list
            )

            self.optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for k in sums:
                if k in losses:
                    sums[k] += losses[k]
            for k, v in losses.get('boundary_diag', {}).items():
                if k not in diag_sum:
                    diag_sum[k] = 0 if not isinstance(v, str) else v
                if not isinstance(v, str):
                    diag_sum[k] += v
            n += 1

        result = {k: v / n for k, v in sums.items()}
        if diag_sum:
            result['boundary_diag'] = {
                k: v if isinstance(v, str) else v / n
                for k, v in diag_sum.items()
            }
        return result

    def _validate_epoch(self, loader: DataLoader, low_dim_list) -> dict:
        self.model.eval()
        sums = {
            'total': 0.0, 'recon': 0.0, 'kl': 0.0,
            'colloc_Na2SO4': 0.0, 'colloc_MgSO4': 0.0, 'cycle': 0.0,
        }
        n = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                output, z_mean, z_logvar = self.model(X_batch)
                conditions = X_batch[:, :self.condition_dim]
                _, losses  = self.loss_fn.compute_total_loss(
                    output, y_batch, z_mean, z_logvar, conditions, low_dim_list
                )
                for k in sums:
                    if k in losses:
                        sums[k] += losses[k]
                n += 1

        return {k: v / n for k, v in sums.items()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Deterministic inference: z = phi(T, W1) -> W_comp2.

        Args:
            X: (N, 2) [T, W_comp1].

        Returns:
            (N,) predicted W_comp2 values.
        """
        if not self.is_scaler_fitted:
            raise RuntimeError("Model has not been fitted. Please call fit() first.")

        self.model.eval()
        X = np.asarray(X, dtype=np.float32)

        T_min  = self.scaler.data_min_[0];  T_rng  = self.scaler.data_max_[0] - T_min
        W1_min = self.scaler.data_min_[1];  W1_rng = self.scaler.data_max_[1] - W1_min
        W2_min = self.scaler.data_min_[2];  W2_rng = self.scaler.data_max_[2] - W2_min

        T_norm  = (X[:, 0:1] - T_min)  / T_rng
        W1_norm = (X[:, 1:2] - W1_min) / W1_rng

        phi_input = torch.FloatTensor(
            np.column_stack([T_norm, W1_norm])
        ).to(self.device)
        T_norm_t = torch.FloatTensor(T_norm).to(self.device)

        with torch.no_grad():
            z   = self.model.infer_z(phi_input)
            out = self.model.decode(z, T_norm_t).cpu().numpy()

        return (out[:, 1:2] * W2_rng + W2_min).flatten()

    def generate_samples(
        self,
        n_samples: int,
        T_range: Tuple[float, float],
    ) -> np.ndarray:
        """Sample from the learned solubility manifold.

        Args:
            n_samples: Number of samples to generate.
            T_range:   Temperature range (T_min, T_max).

        Returns:
            (n_samples, 3) array [T, W_comp1, W_comp2].
        """
        if not self.is_scaler_fitted:
            raise RuntimeError("Model has not been fitted. Please call fit() first.")

        self.model.eval()
        with torch.no_grad():
            T_raw    = np.random.uniform(T_range[0], T_range[1], (n_samples, 1))
            T_min_np = self.scaler.data_min_[0]
            T_rng_np = self.scaler.data_max_[0] - T_min_np
            T_norm   = (T_raw - T_min_np) / T_rng_np
            T_tensor = torch.FloatTensor(T_norm).to(self.device)

            z   = (torch.rand(n_samples, self.config.LATENT_DIM, device=self.device)
                   * 2 - 1) * self.config.Z_HIGH
            out = self.model.decode(z, T_tensor).cpu().numpy()

            W1_min = self.scaler.data_min_[1];  W1_rng = self.scaler.data_max_[1] - W1_min
            W2_min = self.scaler.data_min_[2];  W2_rng = self.scaler.data_max_[2] - W2_min
            W1 = out[:, 0:1] * W1_rng + W1_min
            W2 = out[:, 1:2] * W2_rng + W2_min

        return np.column_stack([T_raw, W1, W2])

    def scan_latent_space(self, T_values: List[float], n_z: int = 50) -> dict:
        """Scan the latent axis and return composition mass fractions as a function of z.

        Args:
            T_values: List of temperature values to scan.
            n_z:      Number of scan steps.

        Returns:
            Dictionary keyed by temperature value; each entry contains 'z', 'W1', 'W2' arrays.
        """
        if not self.is_scaler_fitted:
            raise RuntimeError("Model has not been fitted. Please call fit() first.")

        self.model.eval()
        T_min_np = self.scaler.data_min_[0];  T_rng_np = self.scaler.data_max_[0] - T_min_np
        W1_min   = self.scaler.data_min_[1];  W1_rng   = self.scaler.data_max_[1] - W1_min
        W2_min   = self.scaler.data_min_[2];  W2_rng   = self.scaler.data_max_[2] - W2_min

        z_vals  = np.linspace(self.config.Z_LOW, self.config.Z_HIGH, n_z)
        results = {}

        with torch.no_grad():
            for T_val in T_values:
                T_norm   = (T_val - T_min_np) / T_rng_np
                T_tensor = torch.full((n_z, 1), T_norm, device=self.device)
                z_tensor = torch.FloatTensor(z_vals.reshape(-1, 1)).to(self.device)
                out      = self.model.decode(z_tensor, T_tensor).cpu().numpy()
                results[T_val] = {
                    'z':  z_vals,
                    'W1': out[:, 0] * W1_rng + W1_min,
                    'W2': out[:, 1] * W2_rng + W2_min,
                }

        return results

    def save(self, path) -> None:
        """Save model (phi weights are included in model.state_dict())."""
        torch.save({
            'config':           self.config,
            'model_state_dict': self.model.state_dict(),
            'scaler':           self.scaler,
            'is_scaler_fitted': self.is_scaler_fitted,
            'history':          self.history,
        }, path)

    @classmethod
    def load(cls, path) -> 'CVAEPhysicsModel':
        """Restore model from file."""
        ckpt     = torch.load(path, map_location='cpu')
        config   = ckpt['config']
        instance = cls(input_dim=3, condition_dim=1, config=config)
        instance.model.load_state_dict(ckpt['model_state_dict'])
        instance.scaler           = ckpt['scaler']
        instance.is_scaler_fitted = ckpt['is_scaler_fitted']
        instance.history          = ckpt['history']

        if instance.is_scaler_fitted:
            instance.data_min   = torch.tensor(
                instance.scaler.data_min_, device=instance.device).float()
            instance.data_max   = torch.tensor(
                instance.scaler.data_max_, device=instance.device).float()
            instance.data_range = instance.data_max - instance.data_min
            instance.model.data_min   = instance.data_min
            instance.model.data_max   = instance.data_max
            instance.model.data_range = instance.data_range

        instance.model.to(instance.device)
        return instance

    def _print_fit_info(self) -> None:
        cfg = self.config
        print(f"\nMinMaxScaler fitted range:")
        print(f"  T       [{self.scaler.data_min_[0]:.1f}, {self.scaler.data_max_[0]:.1f}]")
        print(f"  W_comp1 [{self.scaler.data_min_[1]:.3f}, {self.scaler.data_max_[1]:.3f}]")
        print(f"  W_comp2 [{self.scaler.data_min_[2]:.3f}, {self.scaler.data_max_[2]:.3f}]")
        print(f"\nModel configuration: condition_dim={self.condition_dim}, latent_dim={cfg.LATENT_DIM}")
        print(f"  comp1 probe point range z ~ U({cfg.Z_LOW:.1f}, {cfg.Z_LOW + cfg.Z_COLLOC_WIDTH:.1f})")
        print(f"  comp2 probe point range z ~ U({cfg.Z_HIGH - cfg.Z_COLLOC_WIDTH:.1f}, {cfg.Z_HIGH:.1f})")
        print(f"  lambda_KL={cfg.LAMBDA_KL}, lambda_Na2SO4={cfg.LAMBDA_COLLOCATION_Na2SO4}, "
              f"lambda_MgSO4={cfg.LAMBDA_COLLOCATION_MgSO4}")
        print(f"  phi hidden dims={cfg.PHI_HIDDEN_DIMS}, lambda_cycle={cfg.LAMBDA_CYCLE}, "
              f"N_cycle={cfg.N_CYCLE_POINTS}")
        T_str = str(cfg.CYCLE_T_RANGE) if cfg.CYCLE_T_RANGE else \
                "training range (recommended: set explicitly to include high-temperature extrapolation domain)"
        print(f"  cycle loss temperature range={T_str}")

    def _print_epoch(self, epoch: int, train: dict, val: Optional[dict]) -> None:
        print(f"\nEpoch {epoch + 1}/{self.config.N_EPOCHS}")
        print(f"  Train — total: {train['total']:.6f}  recon: {train['recon']:.6f}  "
              f"KL: {train['kl']:.6f}  cycle: {train.get('cycle', 0):.6f}")
        print(f"  Colloc (Na2SO4): {train.get('colloc_Na2SO4', 0):.6f}  "
              f"Colloc (MgSO4): {train.get('colloc_MgSO4', 0):.6f}")
        if val is not None:
            print(f"  Val   — total: {val['total']:.6f}  cycle: {val.get('cycle', 0):.6f}")


# ──────────────────────────────────────────────
#  Self-test
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("PC-CVAE solubility version — self-test")
    print("-" * 40)

    config = CVAEConfig(
        LATENT_DIM=1,
        LAMBDA_KL=0.001,
        LAMBDA_COLLOCATION_Na2SO4=3.0,
        LAMBDA_COLLOCATION_MgSO4=0.01,
        Z_LOW=-2.0,
        Z_HIGH=2.0,
        Z_COLLOC_WIDTH=0.5,
        PHI_HIDDEN_DIMS=[64, 64],
        LAMBDA_CYCLE=1.0,
        N_CYCLE_POINTS=64,
        CYCLE_T_RANGE=(-34.0, 200.0),
        N_EPOCHS=3,
        VERBOSE=True,
    )

    np.random.seed(0)
    torch.manual_seed(0)
    N = 120
    X = np.column_stack([
        np.random.uniform(-30, 100, N),
        np.random.uniform(0, 0.3, N),
    ]).astype(np.float32)
    y = np.random.uniform(0, 0.35, N).astype(np.float32)

    cvae = CVAEPhysicsModel(input_dim=3, condition_dim=1, config=config)
    cvae.fit(X, y, low_dim_list=None)

    W2_pred = cvae.predict(X[:10])
    assert W2_pred.shape == (10,), f"Expected (10,), got {W2_pred.shape}"
    print(f"\npredict() output shape: {W2_pred.shape} ✓")

    samples = cvae.generate_samples(50, (-30, 200))
    assert samples.shape == (50, 3)
    print(f"generate_samples output shape: {samples.shape} ✓")

    scan = cvae.scan_latent_space([0, 50, 100])
    assert len(scan) == 3
    print(f"scan_latent_space number of temperatures: {len(scan)} ✓")

    print("\nSelf-test passed ✓")