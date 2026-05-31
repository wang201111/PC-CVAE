"""
物理约束条件变分自编码器（CVAE-Physics，粘度三元体系版 v2）

适用于 MCH / cis-Decalin / HMN 三元粘度体系，以温度 T 和压力 P
为条件变量，潜变量空间采用等边三角形几何约束三条二元边界。
依赖 low_dim_model.py 中的 LowDimEnsemble 作为低维体系边界模型接口。

v2 新增（与溶解度版对称）：
  φ（逆流形推断头）
    轻量级 MLP，输入 (T_norm, P_norm, MCH_norm, Dec_norm) 共 4 维，
    输出 z̃ ≈ μ，维度等于 latent_dim。
    推断时以 z̃ 替代随机采样或编码器推断，实现确定性预测。

  L_cycle（先验自采样循环一致性损失）
    从与 latent_dim 对应的几何先验采样 z_rand：
      latent_dim=1 : z ~ U(Z_LOW, Z_HIGH)
      latent_dim=2 : Dirichlet(1,1,1) 均匀覆盖等边三角形
      latent_dim>2 : 前 2 维来自 Dirichlet，其余 ~ N(0,1)
    覆盖全温压区（含高温外推区），经梯度截断的 Decoder 生成
    (MCH_fake, Dec_fake)，再由 φ 逆推 z̃，最小化 ||z̃ - z_rand||²。
    Decoder 梯度完全截断，不影响配点约束。

新增公共接口：
  predict(X)  — 确定性推断，z = φ(T, P, MCH, Dec)，替代原编码器方案

潜变量维度与消融实验：
  latent_dim = 1
    使用 Z_LOW/Z_HIGH 端点约束 MCH=0 和 Dec=0 两条边界；
    HMN=0 边界在 1D 中无自然位置，训练时跳过。
    φ 输入仍为 4 维（T/P/MCH/Dec），输出 1 维 z。

  latent_dim = 2（参考设置）
    完整等边三角形，三条边各对应一条二元边界。
    φ 输入 4 维，输出 2 维 z。

  latent_dim > 2（消融对照）
    三角形嵌入前 2 个 z 维度，额外维度 z[2:] 自由浮动。
    φ 输出 latent_dim 维；L_cycle 也约束全部维度。

等边三角形顶点（R = Z_HIGH = 2.0）：
  P_MCH = (0,    R)  = ( 0.000,  2.000)  → 纯 MCH
  P_Dec = (+√3, -1)  = ( 1.732, -1.000)  → 纯 cis-Decalin
  P_HMN = (-√3, -1)  = (-1.732, -1.000)  → 纯 HMN
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
    """将笛卡尔坐标 (z1, z2) 转换为等边三角形的重心坐标 (λ_MCH, λ_Dec, λ_HMN)。"""
    R2      = R / 2.0
    lam_mch = (z2 + R2) / (3.0 * R2)
    lam_dec = (R - z2 + _SQRT3 * z1) / (3.0 * R)
    lam_hmn = (R - z2 - _SQRT3 * z1) / (3.0 * R)
    return lam_mch, lam_dec, lam_hmn


# ──────────────────────────────────────────────
#  配置
# ──────────────────────────────────────────────

@dataclass
class CVAEConfig:
    """CVAE 超参数配置（粘度三元体系，支持消融实验）。"""

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

    # φ 逆流形推断头
    PHI_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [64, 64])
    LAMBDA_CYCLE: float = 1.0
    N_CYCLE_POINTS: int = 64
    # 循环损失温度/压力采样范围，应显式设置为完整物理区间（含高温/高压外推区）
    # 例如 CYCLE_T_RANGE=(20.0, 80.0), CYCLE_P_RANGE=(1e5, 1e8)
    # None 时退化为训练数据范围（φ 外推能力将受限）
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
#  低维模型描述符
# ──────────────────────────────────────────────

@dataclass
class LowDimInfo:
    """低维体系模型描述符。

    Args:
        model: LowDimEnsemble 实例，需实现 predict_torch(X, return_std) 接口。
        name: 体系名称标识符（用于日志）。
        boundary_type: 约束边界类型：
            'mch_zero' — x_MCH=0 边界（cis_Dec-HMN），模型输入 [T, P, x_Dec]
            'dec_zero' — x_Dec=0 边界（MCH-HMN），模型输入 [T, P, x_MCH]
            'hmn_zero' — x_HMN=0 边界（MCH-cis_Dec），模型输入 [T, P, x_MCH]
    """
    model: LowDimEnsemble
    name: str
    boundary_type: str


# ──────────────────────────────────────────────
#  网络主体
# ──────────────────────────────────────────────

class PhysicsConstrainedCVAE(nn.Module):
    """物理约束条件变分自编码器网络（含逆流形推断头 φ）。

    编码器：[T, P, MCH, Dec, Visc] → z(latent_dim)
    解码器：[z, T, P] → (MCH, Dec, Visc)
    φ 头：  [T_norm, P_norm, MCH_norm, Dec_norm] → z̃ ≈ μ

    Args:
        input_dim:       编码器输入维度，固定为 5 [T, P, MCH, Dec, Visc]。
        condition_dim:   解码器条件维度，固定为 2 [T, P]。
        latent_dim:      潜变量维度。
        hidden_dims:     编码器/解码器隐藏层宽度列表。
        phi_hidden_dims: φ 头隐藏层宽度列表。
        dropout:         Dropout 概率。
    """

    # φ 输入维度 = input_dim - 1（排除 Visc）= 4 [T, P, MCH, Dec]
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

        # 编码器
        self.encoder   = self._build_mlp(input_dim, hidden_dims, dropout)
        self.fc_mean   = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # 解码器：输出 3 维 (MCH, Dec, Visc)
        self.decoder = self._build_mlp(
            latent_dim + condition_dim, list(reversed(hidden_dims)), dropout
        )
        self.fc_out = nn.Linear(hidden_dims[0], 3)

        # φ：逆流形推断头，(T_norm, P_norm, MCH_norm, Dec_norm) → z̃
        self.phi     = self._build_mlp(self.PHI_INPUT_DIM, phi_hidden_dims, dropout)
        self.phi_out = nn.Linear(phi_hidden_dims[-1], latent_dim)

        # 由 CVAEPhysicsModel.fit() 写入，供配点损失和循环损失使用
        self.data_min = self.data_max = self.data_range = None

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ELU(), nn.Dropout(dropout)])
            prev = h
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回后验分布参数 (mu, log_var)。"""
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def decode(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """(z, [T_norm, P_norm]) → (MCH, Dec, Visc)，形状 (B, 3)。"""
        h = self.decoder(torch.cat([z, conditions], dim=-1))
        return self.fc_out(h)

    def infer_z(self, phi_input: torch.Tensor) -> torch.Tensor:
        """φ 推断：(T_norm, P_norm, MCH_norm, Dec_norm) → z̃，形状 (B, latent_dim)。"""
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
#  损失函数
# ──────────────────────────────────────────────

class CVAELoss:
    """CVAE 复合损失函数（v2）。

    总损失：
        L = L_recon + λ_KL · L_KL
            + λ_MCH · L_colloc,MCH
            + λ_Dec · L_colloc,Dec
            + λ_HMN · L_colloc,HMN
            + λ_cycle · L_cycle
    """

    def __init__(self, model: PhysicsConstrainedCVAE, config: CVAEConfig):
        self.model  = model
        self.config = config
        self.device = next(model.parameters()).device

    def _sample_z_prior(self, n: int, device: torch.device) -> torch.Tensor:
        """从与 latent_dim 几何对应的先验中采样 z_rand。

        latent_dim=1 : z ~ U(Z_LOW, Z_HIGH)，形状 (n, 1)
        latent_dim=2 : Dirichlet(1,1,1) 均匀覆盖等边三角形，形状 (n, 2)
        latent_dim>2 : 前 2 维来自 Dirichlet，其余 ~ N(0,1)，形状 (n, latent_dim)

        与 generate_samples 的采样逻辑完全对称，确保 L_cycle 的覆盖
        与实际使用的流形几何一致。
        """
        ld     = self.model.latent_dim
        Z_LOW  = self.config.Z_LOW
        Z_HIGH = self.config.Z_HIGH
        R      = Z_HIGH;  R2 = R / 2.0;  SR = _SQRT3 * R2

        if ld == 1:
            return (torch.rand(n, 1, device=device) * (Z_HIGH - Z_LOW) + Z_LOW)

        # Dirichlet 均匀采样等边三角形
        # torch.distributions.Dirichlet 在 GPU 上可用
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
        """将 2D 三角形配点坐标补全至 latent_dim。（与 v1 完全一致）"""
        ld = self.model.latent_dim
        if ld == 2:
            return z_2d
        if ld == 1:
            return z_2d[:, 0:1]
        extra = torch.randn(z_2d.shape[0], ld - 2, device=z_2d.device)
        return torch.cat([z_2d, extra], dim=1)

    def _cycle_loss(self) -> torch.Tensor:
        """先验自采样循环一致性损失。

        流程：
          1. 从几何先验采样 z_rand（覆盖全潜空间，含三角形边界）。
          2. 在完整物理温压区（CYCLE_T_RANGE, CYCLE_P_RANGE）随机采样 T_rand, P_rand。
          3. Decoder(z_rand, [T_norm, P_norm]) [梯度截断] → (MCH_fake, Dec_fake, Visc_fake)。
          4. 反归一化 MCH_fake, Dec_fake → 重归一化为 MCH_fake_norm, Dec_fake_norm。
          5. φ(T_norm, P_norm, MCH_fake_norm, Dec_fake_norm) → z̃。
          6. L_cycle = MSE(z̃, z_rand.detach())。

        Decoder 梯度通过 no_grad 完全截断，不影响配点约束和重建损失。
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

        # Step 1: 采样 z_rand
        z_rand = self._sample_z_prior(n, device)

        # Step 2: 采样 T_rand, P_rand
        T_raw  = torch.rand(n, 1, device=device) * (Tc_max - Tc_min) + Tc_min
        P_raw  = torch.rand(n, 1, device=device) * (Pc_max - Pc_min) + Pc_min
        T_norm = (T_raw - T_min) / T_rng
        P_norm = (P_raw - P_min) / P_rng
        cond   = torch.cat([T_norm, P_norm], dim=1)

        # Step 3: Decoder 前向（梯度截断）
        with torch.no_grad():
            out_fake     = self.model.decode(z_rand, cond)    # (n, 3)
            MCH_fake_raw = out_fake[:, 0:1] * MCH_rng + MCH_min
            Dec_fake_raw = out_fake[:, 1:2] * Dec_rng + Dec_min

        # Step 4: 反归一化 → 重归一化（φ 输入归一化域与训练数据一致）
        MCH_fake_norm = (MCH_fake_raw - MCH_min) / MCH_rng
        Dec_fake_norm = (Dec_fake_raw - Dec_min) / Dec_rng

        # Step 5: φ 推断 z̃
        phi_input = torch.cat([T_norm, P_norm, MCH_fake_norm, Dec_fake_norm], dim=1)
        z_tilde   = self.model.infer_z(phi_input)

        # Step 6: 循环一致性误差
        return F.mse_loss(z_tilde, z_rand.detach())

    def compute_boundary_loss(
        self,
        conditions: torch.Tensor,
        low_dim_list: List[LowDimInfo],
    ) -> Tuple[dict, dict]:
        """计算三条边界的配点约束损失。

        ★ v3 修复：全程在归一化 [0,1] 空间计算损失，彻底消除各变量量级差异的影响。

        核心原则：
          1. decoder 输出本身就是归一化空间的值（不再反归一化）。
          2. 组成目标在归一化 [0,1] 空间均匀采样，仅在传给教师模型时反归一化。
          3. 教师模型输出（原始尺度）用 (Vt - V_min) / V_rng 归一化后再做损失。
          4. "comp=0" 约束目标 = (0 - comp_min) / comp_rng（MinMax 归一化下物理零点的坐标）。
             对粘度（comp_min=0）此值恰为 0，对 TC（comp_min>0）为小负数，两者均正确。

        如此，所有损失项量级均在 [0,1]² 范围，与 L_recon / L_KL 完全可比，
        不受 T/P 单位（Pa vs K）或组成范围（mol% vs 摩尔分数）的影响。
        """
        if self.model.data_min is None or not low_dim_list:
            z = torch.tensor(0.0, device=self.device)
            return {'colloc_mch': z, 'colloc_dec': z, 'colloc_hmn': z}, {}

        device = conditions.device
        ld     = self.model.latent_dim

        # ── scaler 参数（原始尺度）──────────────────────────────────────
        T_min,  T_rng  = self.model.data_min[0], self.model.data_range[0]
        P_min,  P_rng  = self.model.data_min[1], self.model.data_range[1]
        C1_min, C1_rng = self.model.data_min[2], self.model.data_range[2]  # MCH / CaCl₂
        C2_min, C2_rng = self.model.data_min[3], self.model.data_range[3]  # Dec / NaCl
        V_min,  V_rng  = self.model.data_min[4], self.model.data_range[4]  # Visc / TC

        # ── 物理零点在归一化空间中的坐标 ───────────────────────────────
        # MinMaxScaler: norm = (x - x_min) / x_rng
        # 物理 comp=0 → norm_zero = (0 - x_min) / x_rng = -x_min / x_rng
        # 粘度 MCH_min=Dec_min=0  → norm_zero = 0  （与旧版完全等价）
        # TC   CaCl₂_min=0.0017  → norm_zero ≈ -0.155（正确反映盐浓度下限）
        C1_zero_norm = -C1_min / C1_rng
        C2_zero_norm = -C2_min / C2_rng

        # ── 配点采样范围 ────────────────────────────────────────────────
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

        # 总组成（mol% → 100，摩尔分数 → 1.0；用于 hmn_zero 约束）
        # 由 C1_rng 量级自动判断：mol% 的 range 约为 80，摩尔分数约为 0.05
        C_total = 100.0 if float(C1_rng) > 10.0 else 1.0

        losses = {k: torch.tensor(0.0, device=device)
                  for k in ('colloc_mch', 'colloc_dec', 'colloc_hmn')}
        diag = {}

        for ti in low_dim_list:
            T_c    = torch.rand(n, 1, device=device) * (Tc_max - Tc_min) + Tc_min
            P_c    = torch.rand(n, 1, device=device) * (Pc_max - Pc_min) + Pc_min
            cond_c = torch.cat([(T_c - T_min) / T_rng,
                                 (P_c - P_min) / P_rng], dim=1)

            def _vt_norm(comp_orig: torch.Tensor) -> torch.Tensor:
                """教师模型接受原始尺度输入，输出归一化到 [0,1]。"""
                with torch.no_grad():
                    Vt, _, _ = ti.model.predict_torch(
                        torch.cat([T_c, P_c, comp_orig], dim=1), return_std=False)
                return (Vt - V_min) / V_rng

            # ── latent_dim == 1：端点配点 ──────────────────────────────
            if ld == 1:
                delta = torch.rand(n, 1, device=device) * dW

                if ti.boundary_type == 'mch_zero':
                    z_c = self.config.Z_LOW + delta
                    # C2 目标：归一化空间均匀采样 → 反归一化传给教师模型
                    C2_tgt_n = torch.rand(n, 1, device=device)
                    Vt_n     = _vt_norm(C2_tgt_n * C2_rng + C2_min)
                    out      = self.model.decode(z_c, cond_c)   # 归一化空间
                    losses['colloc_mch'] = (
                        (out[:, 0:1] - C1_zero_norm).pow(2).mean()   # C1 → 0
                        + (out[:, 1:2] - C2_tgt_n).pow(2).mean()     # C2 匹配目标
                        + (out[:, 2:3] - Vt_n).pow(2).mean()         # 性质匹配教师
                    )

                elif ti.boundary_type == 'dec_zero':
                    z_c      = self.config.Z_HIGH - delta
                    C1_tgt_n = torch.rand(n, 1, device=device)
                    Vt_n     = _vt_norm(C1_tgt_n * C1_rng + C1_min)
                    out      = self.model.decode(z_c, cond_c)
                    losses['colloc_dec'] = (
                        (out[:, 1:2] - C2_zero_norm).pow(2).mean()   # C2 → 0
                        + (out[:, 0:1] - C1_tgt_n).pow(2).mean()     # C1 匹配目标
                        + (out[:, 2:3] - Vt_n).pow(2).mean()         # 性质匹配教师
                    )
                # HMN=0 在 1D 中无自然位置，跳过
                continue

            # ── latent_dim >= 2：等边三角形配点 ───────────────────────
            t     = torch.rand(n, 1, device=device)
            delta = torch.rand(n, 1, device=device) * dW

            if ti.boundary_type == 'mch_zero':
                z1_e = SR * (1.0 - 2.0 * t)
                z2_e = torch.full((n, 1), -R2, device=device)
                z_c  = self._pad_z(torch.cat([z1_e, z2_e + delta], dim=1))
                # 沿 mch_zero 边：C2 从最大（t=0）到最小（t=1），归一化空间即 (1-t) → 0
                C2_tgt_n = (1.0 - t)
                Vt_n     = _vt_norm(C2_tgt_n * C2_rng + C2_min)
                out      = self.model.decode(z_c, cond_c)
                losses['colloc_mch'] = (
                    (out[:, 0:1] - C1_zero_norm).pow(2).mean()   # C1 → 0
                    + (out[:, 1:2] - C2_tgt_n).pow(2).mean()     # C2 匹配目标
                    + (out[:, 2:3] - Vt_n).pow(2).mean()         # 性质匹配教师
                )
                diag[f'{ti.name}_C1_orig_mean'] = float(
                    out[:, 0:1].mean().item() * float(C1_rng) + float(C1_min))

            elif ti.boundary_type == 'dec_zero':
                z1_e = -SR * t
                z2_e = R - 3.0 * R2 * t
                z_c  = self._pad_z(torch.cat([
                    z1_e + delta * (_SQRT3 / 2.0),
                    z2_e + delta * (-0.5),
                ], dim=1))
                C1_tgt_n = (1.0 - t)
                Vt_n     = _vt_norm(C1_tgt_n * C1_rng + C1_min)
                out      = self.model.decode(z_c, cond_c)
                losses['colloc_dec'] = (
                    (out[:, 1:2] - C2_zero_norm).pow(2).mean()   # C2 → 0
                    + (out[:, 0:1] - C1_tgt_n).pow(2).mean()     # C1 匹配目标
                    + (out[:, 2:3] - Vt_n).pow(2).mean()         # 性质匹配教师
                )
                diag[f'{ti.name}_C2_orig_mean'] = float(
                    out[:, 1:2].mean().item() * float(C2_rng) + float(C2_min))

            elif ti.boundary_type == 'hmn_zero':
                z1_e = SR * t
                z2_e = R - 3.0 * R2 * t
                z_c  = self._pad_z(torch.cat([
                    z1_e + delta * (-_SQRT3 / 2.0),
                    z2_e + delta * (-0.5),
                ], dim=1))
                C1_tgt_n = (1.0 - t)
                C2_tgt_n = t
                Vt_n     = _vt_norm(C1_tgt_n * C1_rng + C1_min)
                out      = self.model.decode(z_c, cond_c)
                # C3（HMN/H₂O）原始值 = C_total - C1 - C2，用 C_total 归一化
                C1_orig  = out[:, 0:1] * C1_rng + C1_min
                C2_orig  = out[:, 1:2] * C2_rng + C2_min
                C3_orig  = C_total - C1_orig - C2_orig
                losses['colloc_hmn'] = (
                    (C3_orig / C_total).pow(2).mean()             # C3 → 0，C_total 归一化
                    + (out[:, 0:1] - C1_tgt_n).pow(2).mean()     # C1 匹配目标
                    + (out[:, 1:2] - C2_tgt_n).pow(2).mean()     # C2 匹配目标
                    + (out[:, 2:3] - Vt_n).pow(2).mean()         # 性质匹配教师
                )
                diag[f'{ti.name}_C3_orig_mean'] = float(C3_orig.mean())

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
        """计算总损失并返回各分项。"""
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
#  训练 & 推理接口
# ──────────────────────────────────────────────

class CVAEPhysicsModel:
    """CVAE-Physics 粘度版 v2 训练与推理接口。

    封装数据预处理、训练循环、学习率调度、早停、采样及模型持久化。
    输入维度固定为 5：[T, P, MCH, Dec, Visc]；
    条件维度为 2：[T, P]；输出维度为 3：[MCH, Dec, Visc]。
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
        """拟合模型。

        Args:
            X: 训练输入，形状 (N, 4)，列为 [T, P, MCH, Dec]。
            y: 训练目标，形状 (N, 1)，为粘度。
            low_dim_list: 低维体系模型列表；为 None 时跳过配点损失。
            X_val: 验证集输入，可选。
            y_val: 验证集目标，可选。

        Returns:
            训练历史字典。
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
                                print(f"\n早停于第 {epoch + 1} 轮")
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
        """确定性推断：z = φ(T, P, MCH, Dec) → Visc。

        Args:
            X:          输入特征，形状 (N, 4)，列为 [T, P, MCH, Dec]。
            return_std: 为 True 时返回 (y_pred, None)，接口与 LowDimEnsemble 一致。

        Returns:
            粘度预测值，形状 (N, 1)。
        """
        if not self.is_scaler_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

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
        """公开 φ 推断接口，返回潜变量坐标 z̃。

        Args:
            X: (N, 4) [T, P, MCH, Dec]。

        Returns:
            (N, latent_dim) z̃ 数组。
        """
        if not self.is_scaler_fitted:
            raise RuntimeError("请先调用 fit()。")

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
        """从学习的粘度流形中采样。（逻辑与 v1 完全一致）

        Args:
            n_samples: 采样数量。
            T_range: 温度范围 (T_min, T_max)。
            P_range: 压力范围 (P_min, P_max)。

        Returns:
            形状 (n_samples, 5) 的数组，列为 [T, P, MCH, Dec, Visc]。
        """
        if not self.is_scaler_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

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
        """诊断工具：沿等边三角形三条边扫描潜变量空间。（与 v1 完全一致）"""
        if not self.is_scaler_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

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
        """消融实验：潜空间有效维度分析。（与 v1 完全一致）"""
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

        print(f"\n  潜空间有效维度分析  latent_dim={ld}")
        print(f"  {'维度':<6} {'协方差特征值':>14} {'KL散度':>10} 状态")
        print("  " + "-" * 50)
        for i, (ev, kl) in enumerate(zip(eigenvalues, kl_per_dim)):
            status = "激活" if ev > THRESHOLD else "退化（冗余）"
            print(f"  z[{i}]   {ev:>14.4f}   {kl:>10.4f}   {status}")
        print(f"\n  激活维度: {active_dims}  |  理论期望: 2（三元组成空间自由度）")

        return {
            'eigenvalues': eigenvalues,
            'active_dims': active_dims,
            'kl_per_dim':  kl_per_dim,
            'z_mean':      z_mean,
        }

    def save(self, path) -> None:
        """保存模型（φ 权重已包含在 model.state_dict() 中）。"""
        torch.save({
            'config':           self.config,
            'model_state_dict': self.model.state_dict(),
            'scaler':           self.scaler,
            'is_scaler_fitted': self.is_scaler_fitted,
            'history':          self.history,
        }, path)

    @classmethod
    def load(cls, path) -> 'CVAEPhysicsModel':
        """从文件恢复模型。"""
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
    # 内部辅助
    # ------------------------------------------------------------------

    def _print_fit_info(self) -> None:
        cfg = self.config
        ld  = cfg.LATENT_DIM
        R   = cfg.Z_HIGH;  R2 = R / 2.0;  SR = _SQRT3 * R2
        print(f"\nCVAE 粘度版 v2  latent_dim={ld}")
        if ld == 1:
            print(f"  模式: 1D 端点配点（2 条边界，HMN=0 跳过）")
            print(f"  MCH=0: z ~ U({cfg.Z_LOW:.1f}, {cfg.Z_LOW + cfg.Z_COLLOC_WIDTH:.1f})")
            print(f"  Dec=0: z ~ U({cfg.Z_HIGH - cfg.Z_COLLOC_WIDTH:.1f}, {cfg.Z_HIGH:.1f})")
        elif ld == 2:
            print(f"  模式: 完整等边三角形（3 条边界）")
            print(f"  P_MCH=(0,{R:.1f})  P_Dec=({SR:.2f},{-R2:.1f})  "
                  f"P_HMN=({-SR:.2f},{-R2:.1f})")
        else:
            print(f"  模式: 等边三角形嵌入前 2 维，z[2:{ld}] ~ N(0,1)")
        print(f"  φ 隐藏层={cfg.PHI_HIDDEN_DIMS}  λ_cycle={cfg.LAMBDA_CYCLE}"
              f"  N_cycle={cfg.N_CYCLE_POINTS}")
        T_str = str(cfg.CYCLE_T_RANGE) if cfg.CYCLE_T_RANGE else "训练范围（建议显式设置）"
        P_str = str(cfg.CYCLE_P_RANGE) if cfg.CYCLE_P_RANGE else "训练范围（建议显式设置）"
        print(f"  循环损失温度范围={T_str}  压力范围={P_str}")

    def _print_progress(self, epoch: int, tr: dict, vl: Optional[dict]) -> None:
        print(f"\nEpoch {epoch + 1}/{self.config.N_EPOCHS}")
        print(f"  训练 — 总损失: {tr['total']:.6f}  重建: {tr['recon']:.6f}  "
              f"KL: {tr['kl']:.6f}  循环: {tr.get('cycle', 0):.6f}")
        print(f"    配点: MCH={tr.get('colloc_mch', 0):.6f}  "
              f"Dec={tr.get('colloc_dec', 0):.6f}  HMN={tr.get('colloc_hmn', 0):.6f}")
        if vl:
            print(f"  验证 — 总损失: {vl['total']:.6f}  循环: {vl.get('cycle', 0):.6f}")


CVAETrainer = CVAEPhysicsModel  # 兼容旧名


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
#  自检
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("CVAE-Physics 粘度版 v2 自检（φ + L_cycle）")
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
        assert y_pred.shape == (10, 1), f"期望 (10, 1)，得到 {y_pred.shape}"
        print(f"  predict()   形状: {y_pred.shape} ✓")

        # infer_z()
        z_pred = m.infer_z(X_dummy[:10])
        assert z_pred.shape == (10, ld), f"期望 (10, {ld})，得到 {z_pred.shape}"
        print(f"  infer_z()   形状: {z_pred.shape} ✓")

        # generate_samples()
        s = m.generate_samples(20, T_range=(20, 80), P_range=(1e5, 1e8))
        assert s.shape == (20, 5), f"期望 (20, 5)，得到 {s.shape}"
        print(f"  generate()  形状: {s.shape} ✓")

        # scan_latent_space()
        scan = m.scan_latent_space(T_val=50.0, P_val=5e7, n_z=10)
        assert set(scan.keys()) == {'edge_mch0', 'edge_dec0', 'edge_hmn0'}
        print(f"  scan()      键集: {set(scan.keys())} ✓")

        # history 包含 cycle
        assert 'train_cycle' in m.history, "history 缺少 train_cycle"
        print(f"  history     包含 train_cycle ✓")

    print("\n自检通过（latent_dim = 1, 2, 4）")