"""
Physics-Constrained Conditional Variational Autoencoder (PC-CVAE) — Viscosity version.

Designed for the MCH / cis-Decalin / HMN ternary viscosity system, with temperature T
and pressure P as conditioning variables. The latent space uses an equilateral triangle
geometry to constrain the three binary composition boundaries.
Depends on LowDimEnsemble from low_dim_model.py as the boundary model interface.

Components
----------
phi (Inverse Manifold Mapping structure)
    Lightweight MLP with inputs (T_norm, P_norm, MCH_norm, Dec_norm) — 4 dimensions —
    and output z_tilde ≈ mu with dimension equal to latent_dim.
    At inference, z_tilde replaces random sampling or encoder inference to achieve
    deterministic prediction.

L_cycle (prior self-sampling cycle consistency loss)
    Samples z_rand from the geometric prior corresponding to latent_dim:
      latent_dim=1 : z ~ U(Z_LOW, Z_HIGH)
      latent_dim=2 : Dirichlet(1,1,1) uniformly covering the equilateral triangle
      latent_dim>2 : first 2 dimensions from Dirichlet, remainder ~ N(0,1)
    Covers the full temperature and pressure range (including extrapolation domain).
    A frozen Decoder generates (MCH_fake, Dec_fake); phi then inversely infers z_tilde.
    Minimises ||z_tilde - z_rand||^2. Decoder gradients are fully detached, leaving
    the collocation constraint unaffected.

Public interface
----------------
predict(X)  — deterministic inference via z = phi(T, P, MCH, Dec)

Latent dimensionality and ablation study
-----------------------------------------
latent_dim = 1
    Uses Z_LOW/Z_HIGH endpoints to constrain the MCH=0 and Dec=0 boundaries.
    The HMN=0 boundary has no natural position in 1D and is skipped during training.
    phi input remains 4-dimensional (T/P/MCH/Dec); output is 1-dimensional z.

latent_dim = 2 (reference configuration)
    Full equilateral triangle; each of the three edges corresponds to a binary boundary.
    phi input is 4-dimensional; output is 2-dimensional z.

latent_dim > 2 (ablation comparison)
    Triangle embedded in the first 2 latent dimensions; additional dimensions z[2:] float freely.
    phi output has latent_dim dimensions; L_cycle also constrains all dimensions.

Equilateral triangle vertices (R = Z_HIGH = 2.0):
  P_MCH = (0,    R)  = ( 0.000,  2.000)  -> pure MCH
  P_Dec = (+√3, -1)  = ( 1.732, -1.000)  -> pure cis-Decalin
  P_HMN = (-√3, -1)  = (-1.732, -1.000)  -> pure HMN
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from low_dim_model import LowDimEnsemble

_SQRT3 = math.sqrt(3)


def barycentric_coords(z1: float, z2: float, R: float = 2.0) -> Tuple[float, float, float]:
    """Convert Cartesian coordinates (z1, z2) to barycentric coordinates (lambda_MCH, lambda_Dec, lambda_HMN)."""
    R2      = R / 2.0
    lam_mch = (z2 + R2) / (3.0 * R2)
    lam_dec = (R - z2 + _SQRT3 * z1) / (3.0 * R)
    lam_hmn = (R - z2 - _SQRT3 * z1) / (3.0 * R)
    return lam_mch, lam_dec, lam_hmn


# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

@dataclass
class CVAEConfig:
    """CVAE hyperparameter configuration (ternary viscosity system; supports ablation studies)."""

    LATENT_DIM: int = 2
    HIDDEN_DIMS: List[int] = field(default_factory=lambda: [128, 256, 256, 128])
    DROPOUT: float = 0.1

    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 200
    WEIGHT_DECAY: float = 1e-5

    LAMBDA_KL: float = 0.001
    LAMBDA_COLLOCATION_MCH: float = 1.0
    LAMBDA_COLLOCATION_DEC: float = 1.0
    LAMBDA_COLLOCATION_HMN: float = 1.0

    N_COLLOCATION_POINTS: int = 64
    COLLOCATION_T_RANGE: Optional[Tuple[float, float]] = None
    COLLOCATION_P_RANGE: Optional[Tuple[float, float]] = None

    Z_LOW: float = -2.0
    Z_HIGH: float = +2.0
    Z_COLLOC_WIDTH: float = 0.5

    # phi: Inverse Manifold Mapping structure
    PHI_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [64, 64])
    LAMBDA_CYCLE: float = 1.0
    N_CYCLE_POINTS: int = 64
    # Temperature/pressure sampling range for cycle loss; should be set explicitly to the
    # full physical range (including high-temperature/high-pressure extrapolation domain),
    # e.g. CYCLE_T_RANGE=(20.0, 80.0), CYCLE_P_RANGE=(1e5, 1e8).
    # Defaults to the training data range when None (limiting phi extrapolation capability).
    CYCLE_T_RANGE: Optional[Tuple[float, float]] = None
    CYCLE_P_RANGE: Optional[Tuple[float, float]] = None

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
        model: LowDimEnsemble instance; must implement predict_torch(X, return_std).
        name:  System name identifier (used in logging).
        boundary_type: Boundary type:
            'mch_zero' — x_MCH=0 boundary (cis-Decalin–HMN); model input [T, P, x_Dec]
            'dec_zero' — x_Dec=0 boundary (MCH–HMN);          model input [T, P, x_MCH]
            'hmn_zero' — x_HMN=0 boundary (MCH–cis-Decalin);  model input [T, P, x_MCH]
    """
    model: LowDimEnsemble
    name: str
    boundary_type: str


# ──────────────────────────────────────────────
#  Network
# ──────────────────────────────────────────────

class PhysicsConstrainedCVAE(nn.Module):
    """Physics-Constrained Conditional Variational Autoencoder with Inverse Manifold Mapping structure phi.

    Encoder: [T, P, MCH, Dec, Visc] -> z(latent_dim)
    Decoder: [z, T, P]              -> (MCH, Dec, Visc)
    phi:     [T_norm, P_norm, MCH_norm, Dec_norm] -> z_tilde ≈ mu

    Args:
        input_dim:       Encoder input dimension; fixed at 5 [T, P, MCH, Dec, Visc].
        condition_dim:   Decoder conditioning dimension; fixed at 2 [T, P].
        latent_dim:      Latent variable dimension.
        hidden_dims:     Hidden layer widths for encoder and decoder.
        phi_hidden_dims: Hidden layer widths for phi.
        dropout:         Dropout probability.
    """

    # phi input dimension = input_dim - 1 (excluding Visc) = 4 [T, P, MCH, Dec]
    PHI_INPUT_DIM = 4

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

        # Encoder
        self.encoder   = self._build_mlp(input_dim, hidden_dims, dropout)
        self.fc_mean   = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder: outputs 3 dimensions (MCH, Dec, Visc)
        self.decoder = self._build_mlp(
            latent_dim + condition_dim, list(reversed(hidden_dims)), dropout
        )
        self.fc_out = nn.Linear(hidden_dims[0], 3)

        # phi: Inverse Manifold Mapping structure; (T_norm, P_norm, MCH_norm, Dec_norm) -> z_tilde
        self.phi     = self._build_mlp(self.PHI_INPUT_DIM, phi_hidden_dims, dropout)
        self.phi_out = nn.Linear(phi_hidden_dims[-1], latent_dim)

        # Written by CVAEPhysicsModel.fit(); used by collocation and cycle losses
        self.data_min = self.data_max = self.data_range = None

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return posterior distribution parameters (mu, log_var)."""
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def decode(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """(z, [T_norm, P_norm]) -> (MCH, Dec, Visc), shape (B, 3)."""
        h = self.decoder(torch.cat([z, conditions], dim=-1))
        return self.fc_out(h)

    def infer_z(self, phi_input: torch.Tensor) -> torch.Tensor:
        """phi inference: (T_norm, P_norm, MCH_norm, Dec_norm) -> z_tilde, shape (B, latent_dim)."""
        return self.phi_out(self.phi(phi_input))

    def forward(
        self,
        x: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_logvar = self.encode(x)
        z    = self.reparameterize(z_mean, z_logvar)
        cond = x[:, :self.condition_dim]
        return self.decode(z, cond), z_mean, z_logvar


# ──────────────────────────────────────────────
#  Loss functions
# ──────────────────────────────────────────────

class CVAELoss:
    """Composite CVAE loss function.

    Total loss:
        L = L_recon + lambda_KL  * L_KL
            + lambda_MCH * L_colloc,MCH
            + lambda_Dec * L_colloc,Dec
            + lambda_HMN * L_colloc,HMN
            + lambda_cycle * L_cycle
    """

    def __init__(self, model: PhysicsConstrainedCVAE, config: CVAEConfig):
        self.model  = model
        self.config = config
        self.device = next(model.parameters()).device

    def _sample_z_prior(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample z_rand from the prior corresponding to the latent space geometry.

        latent_dim=1 : z ~ U(Z_LOW, Z_HIGH), shape (n, 1)
        latent_dim=2 : Dirichlet(1,1,1) uniformly covering the equilateral triangle, shape (n, 2)
        latent_dim>2 : first 2 dimensions from Dirichlet, remainder ~ N(0,1), shape (n, latent_dim)

        Sampling logic is fully symmetric with generate_samples, ensuring that the
        coverage of L_cycle is consistent with the actual manifold geometry used.
        """
        ld     = self.model.latent_dim
        Z_LOW  = self.config.Z_LOW
        Z_HIGH = self.config.Z_HIGH
        R      = Z_HIGH;  R2 = R / 2.0;  SR = _SQRT3 * R2

        if ld == 1:
            return (torch.rand(n, 1, device=device) * (Z_HIGH - Z_LOW) + Z_LOW)

        # Uniform sampling of the equilateral triangle via Dirichlet
        alpha = torch.ones(3, device=device)
        lam   = torch.distributions.Dirichlet(alpha).sample((n,))  # (n, 3)
        P_MCH = torch.tensor([0.0,  R],  device=device)
        P_Dec = torch.tensor([ SR, -R2], device=device)
        P_HMN = torch.tensor([-SR, -R2], device=device)
        z_2d  = (lam[:, 0:1] * P_MCH
                 + lam[:, 1:2] * P_Dec
                 + lam[:, 2:3] * P_HMN)            # (n, 2)

        if ld == 2:
            return z_2d
        extra = torch.randn(n, ld - 2, device=device)
        return torch.cat([z_2d, extra], dim=1)

    def _pad_z(self, z_2d: torch.Tensor) -> torch.Tensor:
        """Pad 2D triangle probe point coordinates to latent_dim."""
        ld = self.model.latent_dim
        if ld == 2:
            return z_2d
        if ld == 1:
            return z_2d[:, 0:1]
        extra = torch.randn(z_2d.shape[0], ld - 2, device=z_2d.device)
        return torch.cat([z_2d, extra], dim=1)

    def _cycle_loss(self) -> torch.Tensor:
        """Prior self-sampling cycle consistency loss.

        Procedure:
          1. Sample z_rand from the geometric prior (covering the full latent space, including triangle boundaries).
          2. Uniformly sample T_rand, P_rand over the full physical range (CYCLE_T_RANGE, CYCLE_P_RANGE).
          3. Decoder(z_rand, [T_norm, P_norm]) [gradients detached] -> (MCH_fake, Dec_fake, Visc_fake).
          4. Denormalize MCH_fake, Dec_fake -> renormalize to MCH_fake_norm, Dec_fake_norm.
          5. phi(T_norm, P_norm, MCH_fake_norm, Dec_fake_norm) -> z_tilde.
          6. L_cycle = MSE(z_tilde, z_rand.detach()).

        Decoder gradients are fully detached via no_grad; collocation and reconstruction losses are unaffected.
        """
        if self.model.data_min is None or self.config.N_CYCLE_POINTS == 0:
            return torch.tensor(0.0, device=self.device)

        cfg    = self.config
        device = self.device
        n      = cfg.N_CYCLE_POINTS

        T_min,   T_rng   = self.model.data_min[0], self.model.data_range[0]
        P_min,   P_rng   = self.model.data_min[1], self.model.data_range[1]
        MCH_min, MCH_rng = self.model.data_min[2], self.model.data_range[2]
        Dec_min, Dec_rng = self.model.data_min[3], self.model.data_range[3]

        Tc_min = cfg.CYCLE_T_RANGE[0] if cfg.CYCLE_T_RANGE else float(T_min)
        Tc_max = cfg.CYCLE_T_RANGE[1] if cfg.CYCLE_T_RANGE else float(T_min + T_rng)
        Pc_min = cfg.CYCLE_P_RANGE[0] if cfg.CYCLE_P_RANGE else float(P_min)
        Pc_max = cfg.CYCLE_P_RANGE[1] if cfg.CYCLE_P_RANGE else float(P_min + P_rng)

        # Step 1: sample z_rand
        z_rand = self._sample_z_prior(n, device)

        # Step 2: sample T_rand, P_rand
        T_raw  = torch.rand(n, 1, device=device) * (Tc_max - Tc_min) + Tc_min
        P_raw  = torch.rand(n, 1, device=device) * (Pc_max - Pc_min) + Pc_min
        T_norm = (T_raw - T_min) / T_rng
        P_norm = (P_raw - P_min) / P_rng
        cond   = torch.cat([T_norm, P_norm], dim=1)

        # Step 3: Decoder forward pass (gradients detached)
        with torch.no_grad():
            out_fake     = self.model.decode(z_rand, cond)    # (n, 3)
            MCH_fake_raw = out_fake[:, 0:1] * MCH_rng + MCH_min
            Dec_fake_raw = out_fake[:, 1:2] * Dec_rng + Dec_min

        # Step 4: denormalize -> renormalize (phi input is in the same normalized domain as training data)
        MCH_fake_norm = (MCH_fake_raw - MCH_min) / MCH_rng
        Dec_fake_norm = (Dec_fake_raw - Dec_min) / Dec_rng

        # Step 5: phi inference z_tilde
        phi_input = torch.cat([T_norm, P_norm, MCH_fake_norm, Dec_fake_norm], dim=1)
        z_tilde   = self.model.infer_z(phi_input)

        # Step 6: cycle consistency error
        return F.mse_loss(z_tilde, z_rand.detach())

    def compute_boundary_loss(
        self,
        conditions: torch.Tensor,
        low_dim_list: List[LowDimInfo],
    ) -> Tuple[dict, dict]:
        """Compute boundary collocation losses for all three edges."""
        if self.model.data_min is None or not low_dim_list:
            z = torch.tensor(0.0, device=self.device)
            return {'colloc_mch': z, 'colloc_dec': z, 'colloc_hmn': z}, {}

        device = conditions.device
        ld     = self.model.latent_dim

        T_min,   T_rng   = self.model.data_min[0], self.model.data_range[0]
        P_min,   P_rng   = self.model.data_min[1], self.model.data_range[1]
        MCH_min, MCH_rng = self.model.data_min[2], self.model.data_range[2]
        Dec_min, Dec_rng = self.model.data_min[3], self.model.data_range[3]
        V_min,   V_rng   = self.model.data_min[4], self.model.data_range[4]

        n  = self.config.N_COLLOCATION_POINTS
        R  = self.config.Z_HIGH;  R2 = R / 2.0;  SR = _SQRT3 * R2
        dW = self.config.Z_COLLOC_WIDTH

        Tc_min = self.config.COLLOCATION_T_RANGE[0] if self.config.COLLOCATION_T_RANGE \
                 else float(T_min)
        Tc_max = self.config.COLLOCATION_T_RANGE[1] if self.config.COLLOCATION_T_RANGE \
                 else float(T_min + T_rng)
        Pc_min = self.config.COLLOCATION_P_RANGE[0] if self.config.COLLOCATION_P_RANGE \
                 else float(P_min)
        Pc_max = self.config.COLLOCATION_P_RANGE[1] if self.config.COLLOCATION_P_RANGE \
                 else float(P_min + P_rng)

        losses = {k: torch.tensor(0.0, device=device)
                  for k in ('colloc_mch', 'colloc_dec', 'colloc_hmn')}
        diag = {}

        for ti in low_dim_list:
            T_c    = torch.rand(n, 1, device=device) * (Tc_max - Tc_min) + Tc_min
            P_c    = torch.rand(n, 1, device=device) * (Pc_max - Pc_min) + Pc_min
            cond_c = torch.cat([(T_c - T_min) / T_rng,
                                 (P_c - P_min) / P_rng], dim=1)

            # ── latent_dim == 1: endpoint probe points ──────────────────────────────
            if ld == 1:
                delta = torch.rand(n, 1, device=device) * dW

                if ti.boundary_type == 'mch_zero':
                    z_c          = self.config.Z_LOW + delta
                    x_Dec_target = torch.rand(n, 1, device=device) * 100.0
                    with torch.no_grad():
                        Vt, _, _ = ti.model.predict_torch(
                            torch.cat([T_c, P_c, x_Dec_target], dim=1), return_std=False)
                    out   = self.model.decode(z_c, cond_c)
                    x_MCH = out[:, 0:1] * MCH_rng + MCH_min
                    x_Dec = out[:, 1:2] * Dec_rng + Dec_min
                    Visc  = out[:, 2:3] * V_rng   + V_min
                    losses['colloc_mch'] = (
                        (x_MCH / MCH_rng).pow(2).mean()
                        + ((x_Dec - x_Dec_target) / Dec_rng).pow(2).mean()
                        + ((Visc - Vt) / V_rng).pow(2).mean()
                    )

                elif ti.boundary_type == 'dec_zero':
                    z_c           = self.config.Z_HIGH - delta
                    x_MCH_target  = torch.rand(n, 1, device=device) * 100.0
                    with torch.no_grad():
                        Vt, _, _ = ti.model.predict_torch(
                            torch.cat([T_c, P_c, x_MCH_target], dim=1), return_std=False)
                    out   = self.model.decode(z_c, cond_c)
                    x_MCH = out[:, 0:1] * MCH_rng + MCH_min
                    x_Dec = out[:, 1:2] * Dec_rng + Dec_min
                    Visc  = out[:, 2:3] * V_rng   + V_min
                    losses['colloc_dec'] = (
                        (x_Dec / Dec_rng).pow(2).mean()
                        + ((x_MCH - x_MCH_target) / MCH_rng).pow(2).mean()
                        + ((Visc - Vt) / V_rng).pow(2).mean()
                    )
                # HMN=0 has no natural position in 1D; skip
                continue

            # ── latent_dim >= 2: equilateral triangle probe points ───────────────────────
            t     = torch.rand(n, 1, device=device)
            delta = torch.rand(n, 1, device=device) * dW

            if ti.boundary_type == 'mch_zero':
                z1_e = SR * (1.0 - 2.0 * t)
                z2_e = torch.full((n, 1), -R2, device=device)
                z_c  = self._pad_z(torch.cat([z1_e, z2_e + delta], dim=1))
                x_Dec_target = 100.0 * (1.0 - t)
                with torch.no_grad():
                    Vt, _, _ = ti.model.predict_torch(
                        torch.cat([T_c, P_c, x_Dec_target], dim=1), return_std=False)
                out   = self.model.decode(z_c, cond_c)
                x_MCH = out[:, 0:1] * MCH_rng + MCH_min
                x_Dec = out[:, 1:2] * Dec_rng + Dec_min
                Visc  = out[:, 2:3] * V_rng   + V_min
                losses['colloc_mch'] = (
                    (x_MCH / MCH_rng).pow(2).mean()
                    + ((x_Dec - x_Dec_target) / Dec_rng).pow(2).mean()
                    + ((Visc - Vt) / V_rng).pow(2).mean()
                )
                diag[f'{ti.name}_x_MCH_mean'] = float(x_MCH.mean())

            elif ti.boundary_type == 'dec_zero':
                z1_e = -SR * t
                z2_e = R - 3.0 * R2 * t
                z_c  = self._pad_z(torch.cat([
                    z1_e + delta * (_SQRT3 / 2.0),
                    z2_e + delta * (-0.5),
                ], dim=1))
                x_MCH_target = 100.0 * (1.0 - t)
                with torch.no_grad():
                    Vt, _, _ = ti.model.predict_torch(
                        torch.cat([T_c, P_c, x_MCH_target], dim=1), return_std=False)
                out   = self.model.decode(z_c, cond_c)
                x_MCH = out[:, 0:1] * MCH_rng + MCH_min
                x_Dec = out[:, 1:2] * Dec_rng + Dec_min
                Visc  = out[:, 2:3] * V_rng   + V_min
                losses['colloc_dec'] = (
                    (x_Dec / Dec_rng).pow(2).mean()
                    + ((x_MCH - x_MCH_target) / MCH_rng).pow(2).mean()
                    + ((Visc - Vt) / V_rng).pow(2).mean()
                )
                diag[f'{ti.name}_x_Dec_mean'] = float(x_Dec.mean())

            elif ti.boundary_type == 'hmn_zero':
                z1_e = SR * t
                z2_e = R - 3.0 * R2 * t
                z_c  = self._pad_z(torch.cat([
                    z1_e + delta * (-_SQRT3 / 2.0),
                    z2_e + delta * (-0.5),
                ], dim=1))
                x_MCH_target = 100.0 * (1.0 - t)
                x_Dec_target = 100.0 * t
                with torch.no_grad():
                    Vt, _, _ = ti.model.predict_torch(
                        torch.cat([T_c, P_c, x_MCH_target], dim=1), return_std=False)
                out   = self.model.decode(z_c, cond_c)
                x_MCH = out[:, 0:1] * MCH_rng + MCH_min
                x_Dec = out[:, 1:2] * Dec_rng + Dec_min
                Visc  = out[:, 2:3] * V_rng   + V_min
                x_HMN = 100.0 - x_MCH - x_Dec
                losses['colloc_hmn'] = (
                    (x_HMN / 100.0).pow(2).mean()
                    + ((x_MCH - x_MCH_target) / MCH_rng).pow(2).mean()
                    + ((x_Dec - x_Dec_target) / Dec_rng).pow(2).mean()
                    + ((Visc - Vt) / V_rng).pow(2).mean()
                )
                diag[f'{ti.name}_x_HMN_mean'] = float(x_HMN.mean())

        return losses, diag

    def _pad_z(self, z_2d: torch.Tensor) -> torch.Tensor:
        ld = self.model.latent_dim
        if ld == 2:
            return z_2d
        if ld == 1:
            return z_2d[:, 0:1]
        extra = torch.randn(z_2d.shape[0], ld - 2, device=z_2d.device)
        return torch.cat([z_2d, extra], dim=1)

    def compute_total_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        conditions: torch.Tensor,
        low_dim_list: List[LowDimInfo],
    ) -> Tuple[torch.Tensor, dict]:
        """Compute total loss and return individual components."""
        L_recon = F.mse_loss(output, target)
        L_kl    = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        bl, diag = self.compute_boundary_loss(conditions, low_dim_list)
        L_cycle  = self._cycle_loss()

        loss_total = (
            L_recon
            + self.config.LAMBDA_KL              * L_kl
            + self.config.LAMBDA_COLLOCATION_MCH * bl['colloc_mch']
            + self.config.LAMBDA_COLLOCATION_DEC * bl['colloc_dec']
            + self.config.LAMBDA_COLLOCATION_HMN * bl['colloc_hmn']
            + self.config.LAMBDA_CYCLE           * L_cycle
        )
        return loss_total, {
            'total':      loss_total.item(),
            'recon':      L_recon.item(),
            'kl':         L_kl.item(),
            'colloc_mch': bl['colloc_mch'].item(),
            'colloc_dec': bl['colloc_dec'].item(),
            'colloc_hmn': bl['colloc_hmn'].item(),
            'cycle':      L_cycle.item(),
        }


# ──────────────────────────────────────────────
#  Training and inference interface
# ──────────────────────────────────────────────

class CVAEPhysicsModel:
    """PC-CVAE training and inference interface — viscosity version.

    Encapsulates data preprocessing, training loop, learning rate scheduling,
    early stopping, sampling, and model persistence.
    Input dimension is fixed at 5: [T, P, MCH, Dec, Visc].
    Conditioning dimension is 2: [T, P]. Output dimension is 3: [MCH, Dec, Visc].
    """

    def __init__(self, config: CVAEConfig):
        self.config        = config
        self.device        = torch.device(config.DEVICE)
        self.input_dim     = 5
        self.condition_dim = 2
        self.output_dim    = 3

        self.model = PhysicsConstrainedCVAE(
            input_dim       = self.input_dim,
            condition_dim   = self.condition_dim,
            latent_dim      = config.LATENT_DIM,
            hidden_dims     = config.HIDDEN_DIMS,
            phi_hidden_dims = config.PHI_HIDDEN_DIMS,
            dropout         = config.DROPOUT,
        ).to(self.device)

        self.scaler           = MinMaxScaler(feature_range=(0, 1))
        self.is_scaler_fitted = False

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
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_recon': [], 'train_kl': [], 'train_cycle': [],
            'train_colloc_mch': [], 'train_colloc_dec': [], 'train_colloc_hmn': [],
        }

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
            X:            Training inputs, shape (N, 4), columns [T, P, MCH, Dec].
            y:            Training targets, shape (N, 1), dynamic viscosity.
            low_dim_list: List of low-dimensional subsystem models; collocation loss
                          is skipped when None.
            X_val:        Validation set inputs, optional.
            y_val:        Validation set targets, optional.

        Returns:
            Training history dictionary.
        """
        X_full   = np.column_stack([X, y])
        X_scaled = self.scaler.fit_transform(X_full)
        self.is_scaler_fitted = True

        self.model.data_min   = torch.tensor(self.scaler.data_min_, device=self.device).float()
        self.model.data_max   = torch.tensor(self.scaler.data_max_, device=self.device).float()
        self.model.data_range = self.model.data_max - self.model.data_min

        if self.config.VERBOSE:
            self._print_fit_info()

        X_t = torch.FloatTensor(X_scaled).to(self.device)
        y_t = torch.FloatTensor(X_scaled[:, 2:5]).to(self.device)
        train_loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            Xvs = self.scaler.transform(np.column_stack([X_val, y_val]))
            val_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(Xvs).to(self.device),
                    torch.FloatTensor(Xvs[:, 2:5]).to(self.device),
                ),
                batch_size=self.config.BATCH_SIZE,
            )

        best_val = float('inf');  patience = 0

        for epoch in range(self.config.N_EPOCHS):
            tr = self._train_epoch(train_loader, low_dim_list)
            self.history['train_loss'].append(tr['total'])
            for k in ('recon', 'kl', 'cycle', 'colloc_mch', 'colloc_dec', 'colloc_hmn'):
                self.history[f'train_{k}'].append(tr.get(k, 0.0))

            if self.scheduler:
                self.scheduler.step()

            vl = None
            if val_loader:
                vl = self._validate_epoch(val_loader, low_dim_list)
                self.history['val_loss'].append(vl['total'])
                if self.config.USE_EARLY_STOPPING:
                    if vl['total'] < best_val:
                        best_val = vl['total'];  patience = 0
                    else:
                        patience += 1
                        if patience >= self.config.EARLY_STOP_PATIENCE:
                            if self.config.VERBOSE:
                                print(f"\nEarly stopping at epoch {epoch + 1}")
                            break

            if self.config.VERBOSE and (epoch + 1) % 10 == 0:
                self._print_progress(epoch, tr, vl)

        return self.history

    def _train_epoch(self, loader: DataLoader, low_dim_list) -> dict:
        self.model.train()
        sums = {k: 0.0 for k in (
            'total', 'recon', 'kl', 'cycle',
            'colloc_mch', 'colloc_dec', 'colloc_hmn'
        )}
        nb = 0
        for X_b, y_b in loader:
            out, z_mean, z_logvar = self.model(X_b)
            loss_total, losses = self.loss_fn.compute_total_loss(
                out, y_b, z_mean, z_logvar, X_b[:, :self.condition_dim], low_dim_list
            )
            self.optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            for k in sums:
                sums[k] += losses.get(k, 0.0)
            nb += 1
        return {k: v / nb for k, v in sums.items()}

    def _validate_epoch(self, loader: DataLoader, low_dim_list) -> dict:
        self.model.eval()
        sums = {k: 0.0 for k in (
            'total', 'recon', 'kl', 'cycle',
            'colloc_mch', 'colloc_dec', 'colloc_hmn'
        )}
        nb = 0
        with torch.no_grad():
            for X_b, y_b in loader:
                out, z_mean, z_logvar = self.model(X_b)
                _, losses = self.loss_fn.compute_total_loss(
                    out, y_b, z_mean, z_logvar, X_b[:, :self.condition_dim], low_dim_list
                )
                for k in sums:
                    sums[k] += losses.get(k, 0.0)
                nb += 1
        return {k: v / nb for k, v in sums.items()}

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Deterministic inference: z = phi(T, P, MCH, Dec) -> Visc.

        Args:
            X:          Input features, shape (N, 4), columns [T, P, MCH, Dec].
            return_std: When True, returns (y_pred, None) for interface compatibility
                        with LowDimEnsemble.

        Returns:
            Predicted dynamic viscosity, shape (N, 1).
        """
        if not self.is_scaler_fitted:
            raise RuntimeError("Model has not been fitted. Please call fit() first.")

        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        sc = self.scaler

        T_min   = sc.data_min_[0];  T_rng   = sc.data_max_[0] - T_min
        P_min   = sc.data_min_[1];  P_rng   = sc.data_max_[1] - P_min
        MCH_min = sc.data_min_[2];  MCH_rng = sc.data_max_[2] - MCH_min
        Dec_min = sc.data_min_[3];  Dec_rng = sc.data_max_[3] - Dec_min
        V_min   = sc.data_min_[4];  V_rng   = sc.data_max_[4] - V_min

        T_norm   = (X[:, 0:1] - T_min)   / T_rng
        P_norm   = (X[:, 1:2] - P_min)   / P_rng
        MCH_norm = (X[:, 2:3] - MCH_min) / MCH_rng
        Dec_norm = (X[:, 3:4] - Dec_min) / Dec_rng

        phi_input = torch.FloatTensor(
            np.hstack([T_norm, P_norm, MCH_norm, Dec_norm])
        ).to(self.device)
        cond = torch.FloatTensor(np.hstack([T_norm, P_norm])).to(self.device)

        with torch.no_grad():
            z   = self.model.infer_z(phi_input)
            out = self.model.decode(z, cond).cpu().numpy()

        y_pred = out[:, 2:3] * V_rng + V_min
        return (y_pred, None) if return_std else y_pred

    def infer_z(self, X: np.ndarray) -> np.ndarray:
        """Public phi inference interface; returns latent variable coordinates z_tilde.

        Args:
            X: (N, 4) [T, P, MCH, Dec].

        Returns:
            (N, latent_dim) z_tilde array.
        """
        if not self.is_scaler_fitted:
            raise RuntimeError("Model has not been fitted. Please call fit() first.")

        self.model.eval()
        X  = np.asarray(X, dtype=np.float32)
        sc = self.scaler

        T_norm   = (X[:, 0:1] - sc.data_min_[0]) / (sc.data_max_[0] - sc.data_min_[0])
        P_norm   = (X[:, 1:2] - sc.data_min_[1]) / (sc.data_max_[1] - sc.data_min_[1])
        MCH_norm = (X[:, 2:3] - sc.data_min_[2]) / (sc.data_max_[2] - sc.data_min_[2])
        Dec_norm = (X[:, 3:4] - sc.data_min_[3]) / (sc.data_max_[3] - sc.data_min_[3])

        phi_input = torch.FloatTensor(
            np.hstack([T_norm, P_norm, MCH_norm, Dec_norm])
        ).to(self.device)

        with torch.no_grad():
            z = self.model.infer_z(phi_input)
        return z.cpu().numpy()

    def generate_samples(
        self,
        n_samples: int,
        T_range: Tuple[float, float],
        P_range: Tuple[float, float],
    ) -> np.ndarray:
        """Sample from the learned viscosity manifold.

        Args:
            n_samples: Number of samples to generate.
            T_range:   Temperature range (T_min, T_max).
            P_range:   Pressure range (P_min, P_max).

        Returns:
            Array of shape (n_samples, 5), columns [T, P, MCH, Dec, Visc].
        """
        if not self.is_scaler_fitted:
            raise RuntimeError("Model has not been fitted. Please call fit() first.")

        self.model.eval()
        sc = self.scaler

        T_min   = float(sc.data_min_[0]);  T_rng   = float(sc.data_max_[0] - T_min)
        P_min   = float(sc.data_min_[1]);  P_rng   = float(sc.data_max_[1] - P_min)
        MCH_min = float(sc.data_min_[2]);  MCH_rng = float(sc.data_max_[2] - MCH_min)
        Dec_min = float(sc.data_min_[3]);  Dec_rng = float(sc.data_max_[3] - Dec_min)
        V_min   = float(sc.data_min_[4]);  V_rng   = float(sc.data_max_[4] - V_min)

        R  = float(self.config.Z_HIGH);  R2 = R / 2.0;  SR = _SQRT3 * R2
        ld = self.config.LATENT_DIM

        with torch.no_grad():
            T_raw = np.random.uniform(T_range[0], T_range[1], (n_samples, 1))
            P_raw = np.random.uniform(P_range[0], P_range[1], (n_samples, 1))
            cond  = torch.FloatTensor(np.hstack([
                (T_raw - T_min) / T_rng,
                (P_raw - P_min) / P_rng,
            ])).to(self.device)

            if ld == 1:
                z_np = np.random.uniform(
                    self.config.Z_LOW, self.config.Z_HIGH, (n_samples, 1)
                ).astype(np.float32)
            else:
                P_MCH = np.array([0.0,  R])
                P_Dec = np.array([SR,  -R2])
                P_HMN = np.array([-SR, -R2])
                lam   = np.random.dirichlet([1, 1, 1], size=n_samples)
                z_2d  = (lam[:, 0:1] * P_MCH + lam[:, 1:2] * P_Dec + lam[:, 2:3] * P_HMN)
                if ld == 2:
                    z_np = z_2d.astype(np.float32)
                else:
                    z_extra = np.random.randn(n_samples, ld - 2).astype(np.float32)
                    z_np    = np.hstack([z_2d, z_extra]).astype(np.float32)

            out = self.model.decode(torch.FloatTensor(z_np).to(self.device), cond).cpu().numpy()

        return np.hstack([
            T_raw, P_raw,
            out[:, 0:1] * MCH_rng + MCH_min,
            out[:, 1:2] * Dec_rng + Dec_min,
            out[:, 2:3] * V_rng   + V_min,
        ])

    def scan_latent_space(
        self,
        T_val: float,
        P_val: float,
        n_z: int = 50,
    ) -> dict:
        """Diagnostic tool: scan the latent space along the three edges of the equilateral triangle."""
        if not self.is_scaler_fitted:
            raise RuntimeError("Model has not been fitted. Please call fit() first.")

        self.model.eval()
        sc = self.scaler

        T_min   = float(sc.data_min_[0]);  T_rng   = float(sc.data_max_[0] - T_min)
        P_min   = float(sc.data_min_[1]);  P_rng   = float(sc.data_max_[1] - P_min)
        MCH_min = float(sc.data_min_[2]);  MCH_rng = float(sc.data_max_[2] - MCH_min)
        Dec_min = float(sc.data_min_[3]);  Dec_rng = float(sc.data_max_[3] - Dec_min)
        V_min   = float(sc.data_min_[4]);  V_rng   = float(sc.data_max_[4] - V_min)

        R  = float(self.config.Z_HIGH);  R2 = R / 2.0;  SR = _SQRT3 * R2
        ld = self.config.LATENT_DIM
        t_vals = np.linspace(0.0, 1.0, n_z)
        T_norm = (T_val - T_min) / T_rng
        P_norm = (P_val - P_min) / P_rng

        def decode_edge(z1_arr: np.ndarray, z2_arr: Optional[np.ndarray]) -> dict:
            with torch.no_grad():
                cond = torch.zeros(n_z, 2, device=self.device)
                cond[:, 0] = T_norm;  cond[:, 1] = P_norm
                if ld == 1:
                    z_np = z1_arr.reshape(-1, 1).astype(np.float32)
                elif ld == 2:
                    z_np = np.column_stack([z1_arr, z2_arr]).astype(np.float32)
                else:
                    z_np = np.column_stack([
                        z1_arr, z2_arr, np.zeros((n_z, ld - 2))
                    ]).astype(np.float32)
                out = self.model.decode(torch.FloatTensor(z_np).to(self.device), cond).cpu().numpy()
            return {
                'MCH':  out[:, 0] * MCH_rng + MCH_min,
                'Dec':  out[:, 1] * Dec_rng + Dec_min,
                'Visc': out[:, 2] * V_rng   + V_min,
            }

        result = {}
        if ld == 1:
            z_line = np.linspace(self.config.Z_LOW, self.config.Z_HIGH, n_z)
            dec    = decode_edge(z_line, None)
            for key in ('edge_mch0', 'edge_dec0', 'edge_hmn0'):
                result[key] = {'t': t_vals, 'z1': z_line, 'z2': np.zeros(n_z), **dec}
        else:
            z1_e1 = SR * (1.0 - 2.0 * t_vals);  z2_e1 = np.full(n_z, -R2)
            result['edge_mch0'] = {'t': t_vals, 'z1': z1_e1, 'z2': z2_e1,
                                   **decode_edge(z1_e1, z2_e1)}
            z1_e2 = -SR * t_vals;  z2_e2 = R - 3.0 * R2 * t_vals
            result['edge_dec0'] = {'t': t_vals, 'z1': z1_e2, 'z2': z2_e2,
                                   **decode_edge(z1_e2, z2_e2)}
            z1_e3 = SR * t_vals;  z2_e3 = R - 3.0 * R2 * t_vals
            result['edge_hmn0'] = {'t': t_vals, 'z1': z1_e3, 'z2': z2_e3,
                                   **decode_edge(z1_e3, z2_e3)}
        return result

    def analyze_latent_dimensionality(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """Ablation study: effective dimensionality analysis of the latent space."""
        self.model.eval()
        X_sc = self.scaler.transform(np.column_stack([X_val, y_val]))
        X_t  = torch.FloatTensor(X_sc).to(self.device)

        with torch.no_grad():
            z_mean, z_logvar = self.model.encode(X_t)
            z_mean   = z_mean.cpu().numpy()
            z_logvar = z_logvar.cpu().numpy()

        ld  = self.config.LATENT_DIM
        cov = np.cov(z_mean.T) if ld > 1 else np.array([[np.var(z_mean[:, 0])]])
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        THRESHOLD   = 1.5
        active_dims = int(np.sum(eigenvalues > THRESHOLD))
        kl_per_dim  = -0.5 * np.mean(1 + z_logvar - z_mean ** 2 - np.exp(z_logvar), axis=0)

        print(f"\n  Latent space effective dimensionality analysis  latent_dim={ld}")
        print(f"  {'Dim':<6} {'Covariance eigenvalue':>22} {'KL divergence':>14} Status")
        print("  " + "-" * 60)
        for i, (ev, kl) in enumerate(zip(eigenvalues, kl_per_dim)):
            status = "Active" if ev > THRESHOLD else "Collapsed (redundant)"
            print(f"  z[{i}]   {ev:>22.4f}   {kl:>14.4f}   {status}")
        print(f"\n  Active dimensions: {active_dims}  |  "
              f"Theoretical expectation: 2 (degrees of freedom of the ternary composition space)")

        return {
            'eigenvalues': eigenvalues,
            'active_dims': active_dims,
            'kl_per_dim':  kl_per_dim,
            'z_mean':      z_mean,
        }

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
        d = torch.load(path, map_location='cpu')
        m = cls(config=d['config'])
        m.model.load_state_dict(d['model_state_dict'])
        m.scaler           = d['scaler']
        m.is_scaler_fitted = d['is_scaler_fitted']
        m.history          = d['history']
        if m.is_scaler_fitted:
            m.model.data_min   = torch.tensor(m.scaler.data_min_, device=m.device).float()
            m.model.data_max   = torch.tensor(m.scaler.data_max_, device=m.device).float()
            m.model.data_range = m.model.data_max - m.model.data_min
        m.model.to(m.device)
        return m

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print_fit_info(self) -> None:
        cfg = self.config
        ld  = cfg.LATENT_DIM
        R   = cfg.Z_HIGH;  R2 = R / 2.0;  SR = _SQRT3 * R2
        print(f"\nPC-CVAE viscosity version  latent_dim={ld}")
        if ld == 1:
            print(f"  Mode: 1D endpoint probe points (2 boundaries; HMN=0 skipped)")
            print(f"  MCH=0: z ~ U({cfg.Z_LOW:.1f}, {cfg.Z_LOW + cfg.Z_COLLOC_WIDTH:.1f})")
            print(f"  Dec=0: z ~ U({cfg.Z_HIGH - cfg.Z_COLLOC_WIDTH:.1f}, {cfg.Z_HIGH:.1f})")
        elif ld == 2:
            print(f"  Mode: full equilateral triangle (3 boundaries)")
            print(f"  P_MCH=(0,{R:.1f})  P_Dec=({SR:.2f},{-R2:.1f})  "
                  f"P_HMN=({-SR:.2f},{-R2:.1f})")
        else:
            print(f"  Mode: equilateral triangle embedded in first 2 dims; z[2:{ld}] ~ N(0,1)")
        print(f"  phi hidden dims={cfg.PHI_HIDDEN_DIMS}  lambda_cycle={cfg.LAMBDA_CYCLE}"
              f"  N_cycle={cfg.N_CYCLE_POINTS}")
        T_str = str(cfg.CYCLE_T_RANGE) if cfg.CYCLE_T_RANGE else \
                "training range (recommended: set explicitly)"
        P_str = str(cfg.CYCLE_P_RANGE) if cfg.CYCLE_P_RANGE else \
                "training range (recommended: set explicitly)"
        print(f"  cycle loss T range={T_str}  P range={P_str}")

    def _print_progress(self, epoch: int, tr: dict, vl: Optional[dict]) -> None:
        print(f"\nEpoch {epoch + 1}/{self.config.N_EPOCHS}")
        print(f"  Train — total: {tr['total']:.6f}  recon: {tr['recon']:.6f}  "
              f"KL: {tr['kl']:.6f}  cycle: {tr.get('cycle', 0):.6f}")
        print(f"    Colloc: MCH={tr.get('colloc_mch', 0):.6f}  "
              f"Dec={tr.get('colloc_dec', 0):.6f}  HMN={tr.get('colloc_hmn', 0):.6f}")
        if vl:
            print(f"  Val   — total: {vl['total']:.6f}  cycle: {vl.get('cycle', 0):.6f}")


CVAETrainer = CVAEPhysicsModel  # backward compatibility alias


__all__ = [
    'CVAEConfig',
    'LowDimInfo',
    'PhysicsConstrainedCVAE',
    'CVAELoss',
    'CVAEPhysicsModel',
    'CVAETrainer',
    'barycentric_coords',
]


# ──────────────────────────────────────────────
#  Self-test
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("PC-CVAE viscosity version — self-test (phi + L_cycle)")
    print("=" * 70)

    np.random.seed(0);  torch.manual_seed(0)
    N = 80
    X_dummy = np.column_stack([
        np.random.uniform(20,   80,  N),    # T
        np.random.uniform(1e5,  1e8, N),    # P
        np.random.uniform(0,    80,  N),    # MCH
        np.random.uniform(0,    80,  N),    # Dec
    ]).astype(np.float32)
    y_dummy = np.random.uniform(0.5, 6.0, (N, 1)).astype(np.float32)

    for ld in [1, 2, 4]:
        print(f"\n--- latent_dim = {ld} ---")
        cfg = CVAEConfig(
            LATENT_DIM=ld,
            N_EPOCHS=3,
            VERBOSE=False,
            LAMBDA_KL=0.001,
            LAMBDA_COLLOCATION_MCH=1.0,
            LAMBDA_COLLOCATION_DEC=1.0,
            LAMBDA_COLLOCATION_HMN=1.0,
            PHI_HIDDEN_DIMS=[64, 64],
            LAMBDA_CYCLE=1.0,
            N_CYCLE_POINTS=16,
            CYCLE_T_RANGE=(20.0, 80.0),
            CYCLE_P_RANGE=(1e5, 1e8),
        )
        m = CVAEPhysicsModel(config=cfg)
        m.fit(X_dummy, y_dummy, low_dim_list=None)

        # predict()
        y_pred = m.predict(X_dummy[:10])
        assert y_pred.shape == (10, 1), f"Expected (10, 1), got {y_pred.shape}"
        print(f"  predict()   shape: {y_pred.shape} ✓")

        # infer_z()
        z_pred = m.infer_z(X_dummy[:10])
        assert z_pred.shape == (10, ld), f"Expected (10, {ld}), got {z_pred.shape}"
        print(f"  infer_z()   shape: {z_pred.shape} ✓")

        # generate_samples()
        s = m.generate_samples(20, T_range=(20, 80), P_range=(1e5, 1e8))
        assert s.shape == (20, 5), f"Expected (20, 5), got {s.shape}"
        print(f"  generate()  shape: {s.shape} ✓")

        # scan_latent_space()
        scan = m.scan_latent_space(T_val=50.0, P_val=5e7, n_z=10)
        assert set(scan.keys()) == {'edge_mch0', 'edge_dec0', 'edge_hmn0'}
        print(f"  scan()      keys: {set(scan.keys())} ✓")

        # history contains cycle
        assert 'train_cycle' in m.history, "history missing train_cycle"
        print(f"  history     contains train_cycle ✓")

    print("\nSelf-test passed (latent_dim = 1, 2, 4) ✓")