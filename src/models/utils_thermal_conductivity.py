"""
NaCl–CaCl₂–H₂O 热导率三元体系物理一致性评估工具

提供边界一致性评估（NaCl=0、CaCl₂=0 两条有效边界，H₂O=0 跳过）、
热力学平滑性评估（4D 拉普拉斯 + P99 分位数）及双支柱综合评估框架。
依赖 low_dim_model.py 中的 LowDimEnsemble 作为低维体系边界模型接口。

与粘度版的核心差异：
  1. 两条有效边界（nacl_zero | cacl2_zero），h2o_zero 无物理意义，跳过。
  2. 二元模型输入接口（新结构，盐组分作第三输入）：
     - nacl_zero:  model_cacl2_h2o，切片 [:, [0,1,2]]（T, P, CaCl₂）
     - cacl2_zero: model_nacl_h2o， 切片 [:, [0,1,3]]（T, P, NaCl）← 取第4列
  3. 4D 平滑性网格轴：[T, P, CaCl₂, NaCl]，X[:,2]=CaCl₂，X[:,3]=NaCl。
  4. DNN/TSTREvaluator 输入维度固定为 4：[T, P, CaCl₂, NaCl]。
  5. 边界误差汇总为两项之和（非三项），衰减系数不变。
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
# 日志
# ==============================================================================

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """返回已配置的 Logger 实例。"""
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
# 评估工具函数
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
# DNN 基线网络
# ==============================================================================

class DNN(nn.Module):
    """全连接基线 DNN（热导率版，input_dim=4）。

    Args:
        input_dim:  输入特征维度，热导率体系固定为 4 [T, P, CaCl₂, NaCl]。
        layer_dim:  隐藏层数。
        node_dim:   每层节点数。
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
    """TSTREvaluator 超参数配置（与粘度版接口完全一致）。

    Checkpoint 准则：val MSE（原始尺度）下降即保存，从第 1 轮开始，无轮次门槛。

    Args:
        tstr_epochs:   训练轮数。
        tstr_lr:       学习率。
        dnn_layer_dim: DNN 隐藏层数。
        dnn_node_dim:  每层节点数。
        tstr_device:   计算设备（'auto'/'cuda'/'cpu'）。
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
    """热导率体系标准训练/评估器（与粘度版接口完全一致）。

    训练策略
    --------
    - 训练满 ``config.tstr_epochs`` 轮（默认 1000）。
    - 每轮结束后在原始尺度上计算验证集 MSE；
      若低于历史最低则在内存中保存 state_dict（从第 1 轮开始记录）。
    - 训练结束后加载最佳 checkpoint，再对 train / val / test 做最终评估。

    Args:
        X_val:   验证集特征，(N, 4) [T, P, CaCl₂, NaCl]。
        y_val:   验证集目标，(N,) TC。
        X_test:  测试集特征，(N, 4)。
        y_test:  测试集目标，(N,)。
        X_train: 训练集特征，(N, 4) [T, P, CaCl₂, NaCl]。
        y_train: 训练集目标，(N,)。
        config:  PhysicsConfig 超参数配置。
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
        """在合成数据（或真实数据）上训练 DNN，在验证集/测试集上评估。

        Checkpoint 准则：val MSE（原始尺度）下降即保存，从第 1 轮开始，无轮次门槛。

        Args:
            X_syn:   训练用特征（可以是真实数据，也可以是合成数据），(N, 4) [T, P, CaCl₂, NaCl]。
            y_syn:   训练用目标，(N,) TC。
            epochs:  训练轮数。
            verbose: 是否打印训练进度。

        Returns:
            包含以下键的字典（与粘度版完全对齐）：
                'metrics':      dict，含 train_*/val_*/test_* 全套指标。
                'history':      dict，含 train_r2/val_r2/val_loss/test_r2 逐轮曲线。
                'model':        加载最佳权重后的 DNN。
                'x_scaler':     特征 StandardScaler。
                'y_scaler':     目标 StandardScaler。
                'predictions':  dict，含 'train'/'val'/'test' 预测数组 (N,)。
                'true_values':  dict，含 'train'/'val'/'test' 真实值数组 (N,)。
                'inputs':       dict，含 'train'/'val'/'test' 输入特征数组。
                'n_synthetic':  训练样本数。
                'epochs':       训练轮数。
                'best_epoch':   最佳 checkpoint 所在轮次（1-indexed）。
                'best_val_mse': 最佳验证集 MSE（原始尺度）。
        """
        # ── 标准化 ──────────────────────────────────────────────────────
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_syn_sc  = x_scaler.fit_transform(X_syn)
        y_syn_sc  = y_scaler.fit_transform(y_syn.reshape(-1, 1))   # (N, 1)
        X_val_sc  = x_scaler.transform(self.X_val)
        X_test_sc = x_scaler.transform(self.X_test)

        # ── 构建模型 ────────────────────────────────────────────────────
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
                torch.FloatTensor(y_syn_sc).to(self.device),    # (N, 1)
            ),
            batch_size=64,
            shuffle=True,
        )

        # ── Checkpoint 状态（val MSE 下降，从第 1 轮开始）─────────────
        best_val_mse     = float('inf')
        best_model_state = None
        best_epoch       = 0

        history: Dict[str, List[float]] = {
            'train_r2': [], 'val_r2': [], 'val_loss': [], 'test_r2': []
        }

        log_every = max(1, epochs // 10)

        # ── 训练循环 ────────────────────────────────────────────────────
        for ep in range(epochs):
            # train
            model.train()
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)   # yb: (batch, 1)
                loss.backward()
                optimizer.step()

            # eval（原始尺度）
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

            # checkpoint：val MSE 下降即保存，从第 1 轮开始
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

        # ── 恢复最佳权重（guaranteed，因第 1 轮必然触发）────────────────
        model.load_state_dict(best_model_state)
        model.eval()
        self.logger.info(
            f"加载最优模型：第 {best_epoch} 轮（验证 MSE={best_val_mse:.6f}）"
        )

        # ── 最终推断（使用最佳权重）──────────────────────────────────────
        with torch.no_grad():
            def _final_pred(X_sc: np.ndarray) -> np.ndarray:
                out = model(
                    torch.FloatTensor(X_sc).to(self.device)
                ).cpu().numpy()
                return y_scaler.inverse_transform(out).flatten()

            y_train_final = _final_pred(X_syn_sc)
            y_val_final   = _final_pred(X_val_sc)
            y_test_final  = _final_pred(X_test_sc)

        # ── 计算最终指标 ─────────────────────────────────────────────────
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
                f"  [最终] train_r²={metrics['train_r2']:.4f}  "
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
# 边界一致性评估器
# ==============================================================================

class TCBoundaryEvaluator:
    """NaCl–CaCl₂–H₂O 热导率体系边界一致性评估器（两条有效边界）。

    有效边界：
      nacl_zero  — x_NaCl=0，对应 CaCl₂–H₂O 二元系，使用 model_cacl2_h2o
      cacl2_zero — x_CaCl₂=0，对应 NaCl–H₂O 二元系，使用 model_nacl_h2o
      h2o_zero   — x_H₂O=0，水盐体系无物理意义，跳过

    三元模型输入 X = [T, P, CaCl₂, NaCl]（新结构，盐组分为第3/4列）：
      nacl_zero:  X = [T, P, CaCl₂_varied, NaCl=0]，
                  model_cacl2_h2o 输入 = X[:, [0,1,2]]（T, P, CaCl₂）
      cacl2_zero: X = [T, P, CaCl₂=0, NaCl_varied]，
                  model_nacl_h2o  输入 = X[:, [0,1,3]]（T, P, NaCl）← 取第4列

    Args:
        model_cacl2_h2o: CaCl₂–H₂O 二元体系模型，输入 [T, P, CaCl₂]（3 列）。
        model_nacl_h2o:  NaCl–H₂O  二元体系模型，输入 [T, P, NaCl]（3 列）。
        temp_range:      温度范围 (T_min, T_max)，单位 K。
        pressure_range:  压力范围 (P_min, P_max)，单位 Pa。
        cacl2_range:     CaCl₂ 组分范围 (min, max)，用于 nacl_zero 边界扫描。
        nacl_range:      NaCl  组分范围 (min, max)，用于 cacl2_zero 边界扫描。
        composition_total: H₂O + NaCl + CaCl₂ 的总和（默认 1.0）。
        decay_lambda:    边界评分衰减系数。
        n_samples:       每条边界的采样点数（T/P 网格 × 组分采样）。
        log_level:       日志级别。
    """

    def __init__(
        self,
        model_cacl2_h2o:   LowDimEnsemble,
        model_nacl_h2o:    LowDimEnsemble,
        temp_range:        Tuple[float, float] = (290.0, 570.0),
        pressure_range:    Tuple[float, float] = (5e6, 5e7),
        cacl2_range:       Tuple[float, float] = (0.0, 0.05),
        nacl_range:        Tuple[float, float] = (0.0, 0.05),
        composition_total: float = 1.0,
        decay_lambda:      float = 5.0,
        n_samples:         int   = 10,
        log_level:         int   = logging.INFO,
    ):
        self.model_cacl2_h2o   = model_cacl2_h2o
        self.model_nacl_h2o    = model_nacl_h2o
        self.temp_range        = temp_range
        self.pressure_range    = pressure_range
        self.cacl2_range       = cacl2_range
        self.nacl_range        = nacl_range
        self.composition_total = composition_total
        self.decay_lambda      = decay_lambda
        self.n_samples         = n_samples
        self.logger            = get_logger(self.__class__.__name__, log_level)

        self._generate_boundary_test_points()

    def _generate_boundary_test_points(self) -> None:
        """预生成两条有效边界的测试点及二元体系模型真实值。

        三元模型 X = [T, P, CaCl₂, NaCl]（新结构）：
          nacl_zero:  X = [T, P, CaCl₂_varied, NaCl=0]
                      model_cacl2_h2o 输入 = X[:, [0,1,2]] = [T, P, CaCl₂]
          cacl2_zero: X = [T, P, CaCl₂=0, NaCl_varied]
                      model_nacl_h2o  输入 = X[:, [0,1,3]] = [T, P, NaCl]  ← 第4列

        注：h2o_zero（底边）水盐体系无物理意义，不生成测试点。
        """
        self.logger.info(f"生成边界测试点（每条边界 {self.n_samples} 个点）...")

        T_test = np.linspace(*self.temp_range, self.n_samples)
        P_test = np.linspace(*self.pressure_range, self.n_samples)
        T_grid, P_grid = np.meshgrid(T_test, P_test)
        T_flat = T_grid.flatten()
        P_flat = P_grid.flatten()
        n_tp   = len(T_flat)

        CaCl2_samples = np.linspace(*self.cacl2_range, self.n_samples)
        NaCl_samples  = np.linspace(*self.nacl_range,  self.n_samples)

        # ── nacl_zero 边界（NaCl=0，CaCl₂–H₂O 二元系）─────────────────
        # X = [T, P, CaCl₂_varied, 0]；model_cacl2_h2o 取 X[:,[0,1,2]]
        rows = []
        for i in range(n_tp):
            for cacl2 in CaCl2_samples:
                rows.append([T_flat[i], P_flat[i], cacl2, 0.0])   # NaCl=0
        self.boundary_nacl_zero_X      = np.array(rows, dtype=np.float32)
        self.boundary_nacl_zero_y_true = self.model_cacl2_h2o.predict(
            self.boundary_nacl_zero_X[:, [0, 1, 2]]               # [T, P, CaCl₂]
        ).flatten()

        # ── cacl2_zero 边界（CaCl₂=0，NaCl–H₂O 二元系）────────────────
        # X = [T, P, 0, NaCl_varied]；model_nacl_h2o 取 X[:,[0,1,3]]  ← 第4列
        rows = []
        for i in range(n_tp):
            for nacl in NaCl_samples:
                rows.append([T_flat[i], P_flat[i], 0.0, nacl])    # CaCl₂=0
        self.boundary_cacl2_zero_X      = np.array(rows, dtype=np.float32)
        self.boundary_cacl2_zero_y_true = self.model_nacl_h2o.predict(
            self.boundary_cacl2_zero_X[:, [0, 1, 3]]              # [T, P, NaCl]
        ).flatten()

        self.logger.info(
            f"边界测试点生成完成  "
            f"nacl_zero={len(self.boundary_nacl_zero_X)}  "
            f"cacl2_zero={len(self.boundary_cacl2_zero_X)}"
        )

    def evaluate_parl_boundary(self, trainer: Any) -> Dict[str, Any]:
        """评估模型在两条有效边界上的一致性（h2o_zero 跳过）。

        Args:
            trainer: 具有 predict(X, return_original_scale) 方法的对象。
                     X 形状 (N, 4) [T, P, H₂O, NaCl]。
        """
        self.logger.info("边界一致性评估开始（nacl_zero | cacl2_zero，h2o_zero 跳过）")
        results = {}

        def _eval_one(name: str, X: np.ndarray, y_true: np.ndarray) -> Tuple[float, Dict]:
            y_pred = trainer.predict(X, return_original_scale=True).flatten()
            r2     = float(r2_score(y_true, y_pred))
            rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae    = float(mean_absolute_error(y_true, y_pred))
            nrmse  = calculate_boundary_nrmse(y_pred, y_true, float(np.max(y_true)))
            self.logger.info(f"  {name}  R²={r2:.4f}  RMSE={rmse:.4f}  NRMSE={nrmse:.6f}")
            return nrmse, {
                'r2': r2, 'rmse': rmse, 'mae': mae,
                'y_true': y_true.copy(), 'y_pred': y_pred.copy(),
                'X': X.copy(),
            }

        nrmse_1, results['nacl_zero_boundary']  = _eval_one(
            'NaCl=0', self.boundary_nacl_zero_X, self.boundary_nacl_zero_y_true
        )
        nrmse_2, results['cacl2_zero_boundary'] = _eval_one(
            'CaCl₂=0', self.boundary_cacl2_zero_X, self.boundary_cacl2_zero_y_true
        )

        # 两条边界的误差之和（对应粘度版三条边界之和，衰减系数不变）
        total_error    = nrmse_1 + nrmse_2
        boundary_score = exponential_decay_score(total_error, self.decay_lambda)

        results['combined'] = {
            'nrmse_nacl_zero':          nrmse_1,
            'nrmse_cacl2_zero':         nrmse_2,
            'total_error':              total_error,
            'boundary_score':           boundary_score,
            'physical_max_nacl_zero':   float(np.max(self.boundary_nacl_zero_y_true)),
            'physical_max_cacl2_zero':  float(np.max(self.boundary_cacl2_zero_y_true)),
            'decay_lambda':             self.decay_lambda,
            'n_boundaries_evaluated':   2,
            'h2o_zero_skipped':         True,
        }

        self.logger.info(
            f"边界综合得分: {boundary_score:.6f}  "
            f"（总误差={nrmse_1:.4f}+{nrmse_2:.4f}={total_error:.4f}）"
        )
        return results


# ==============================================================================
# 平滑性评估器
# ==============================================================================

class TCSmoothnessEvaluator:
    """NaCl–CaCl₂–H₂O 热导率体系热力学平滑性评估器（4D 拉普拉斯 + P99 分位数）。

    4D 网格轴：[T, P, CaCl₂, NaCl]（与粘度版 [T, P, MCH, Dec] 完全对称）。

    Args:
        temp_range:     温度范围 (T_min, T_max)，单位 K。
        pressure_range: 压力范围 (P_min, P_max)，单位 Pa。
        cacl2_range:    CaCl₂ 组分范围 (min, max)，X[:,2] 轴。
        nacl_range:     NaCl  组分范围 (min, max)，X[:,3] 轴。
        grid_resolution: 4D 网格分辨率 (n_T, n_P, n_CaCl₂, n_NaCl)。
        smoothness_decay_lambda: 平滑性评分衰减系数。
        log_level:      日志级别。
    """

    def __init__(
        self,
        temp_range:     Tuple[float, float] = (290.0, 570.0),
        pressure_range: Tuple[float, float] = (5e6, 5e7),
        cacl2_range:    Tuple[float, float] = (0.0, 0.05),
        nacl_range:     Tuple[float, float] = (0.0, 0.05),
        grid_resolution: Tuple[int, int, int, int] = (20, 20, 20, 20),
        smoothness_decay_lambda: float = 15.0,
        log_level: int = logging.INFO,
    ):
        self.temp_range      = temp_range
        self.pressure_range  = pressure_range
        self.cacl2_range     = cacl2_range
        self.nacl_range      = nacl_range
        self.grid_resolution = grid_resolution
        self.decay_lambda    = smoothness_decay_lambda
        self.logger          = get_logger(self.__class__.__name__, log_level)

    def generate_regular_grid(self) -> np.ndarray:
        """生成规则 4D 网格，返回 (N, 4) [T, P, CaCl₂, NaCl]。"""
        n_T, n_P, n_CaCl2, n_NaCl = self.grid_resolution
        T_s     = np.linspace(*self.temp_range,     n_T)
        P_s     = np.linspace(*self.pressure_range, n_P)
        CaCl2_s = np.linspace(*self.cacl2_range,    n_CaCl2)
        NaCl_s  = np.linspace(*self.nacl_range,     n_NaCl)
        T_g, P_g, CaCl2_g, NaCl_g = np.meshgrid(T_s, P_s, CaCl2_s, NaCl_s, indexing='ij')
        X_grid = np.column_stack([
            T_g.flatten(), P_g.flatten(),
            CaCl2_g.flatten(), NaCl_g.flatten(),
        ])
        self.logger.info(
            f"4D 规则网格生成完成：分辨率={self.grid_resolution}，总点数={len(X_grid):,}"
        )
        return X_grid

    def evaluate_smoothness(self, trainer: Any) -> Tuple[float, Dict[str, Any]]:
        """用 4D 拉普拉斯法评估热力学平滑性。"""
        self.logger.info("热力学平滑性评估开始（4D 拉普拉斯 + P99 分位数）")

        X_grid  = self.generate_regular_grid()
        self.logger.info("模型预测中...")
        TC_pred = trainer.predict(X_grid, return_original_scale=True).flatten()
        TC_4d   = TC_pred.reshape(self.grid_resolution)
        self.logger.info(f"预测完成，4D 张量形状: {TC_4d.shape}")

        self.logger.info("计算 4D 拉普拉斯算子...")
        d2_dT2    = np.gradient(np.gradient(TC_4d, axis=0), axis=0)
        d2_dP2    = np.gradient(np.gradient(TC_4d, axis=1), axis=1)
        d2_dCaCl2 = np.gradient(np.gradient(TC_4d, axis=2), axis=2)
        d2_dNaCl2 = np.gradient(np.gradient(TC_4d, axis=3), axis=3)

        grad_T    = np.gradient(TC_4d, axis=0)
        grad_P    = np.gradient(TC_4d, axis=1)
        grad_CaCl2 = np.gradient(TC_4d, axis=2)
        grad_NaCl = np.gradient(TC_4d, axis=3)

        laplacian_4d  = d2_dT2 + d2_dP2 + d2_dCaCl2 + d2_dNaCl2
        laplacian_abs = np.abs(laplacian_4d)

        l1_norm   = float(np.mean(laplacian_abs))
        l2_norm   = float(np.sqrt(np.mean(laplacian_4d ** 2)))
        l4_norm   = float(np.power(np.mean(laplacian_abs ** 4), 0.25))
        linf_norm = float(np.max(laplacian_abs))

        p50  = float(np.percentile(laplacian_abs, 50))
        p90  = float(np.percentile(laplacian_abs, 90))
        p95  = float(np.percentile(laplacian_abs, 95))
        p99  = float(np.percentile(laplacian_abs, 99))

        data_range = float(np.ptp(TC_4d)) + 1e-8
        eta_p99    = p99 / data_range

        smoothness_score = float(np.exp(-self.decay_lambda * eta_p99))
        score_l2  = float(np.exp(-self.decay_lambda * l2_norm  / data_range))
        score_l4  = float(np.exp(-self.decay_lambda * l4_norm  / data_range))
        score_p95 = float(np.exp(-self.decay_lambda * p95      / data_range))

        self.logger.info(f"平滑性得分: {smoothness_score:.6f}  η(P99)={eta_p99:.6f}")

        grad_magnitude = np.sqrt(grad_T**2 + grad_P**2 + grad_CaCl2**2 + grad_NaCl**2)
        tail_thickness = float((p99 - p95) / (p95 + 1e-8))
        quantile_ok    = (p50 <= p90 <= p95 <= p99 <= linf_norm)
        holder_ok      = (l2_norm <= l4_norm + 1e-6) and (l4_norm <= linf_norm + 1e-6)

        if smoothness_score >= 0.99:
            quality, description = '优秀', f'热力学曲面高度平滑（η={eta_p99:.6f}）'
        elif smoothness_score >= 0.95:
            quality, description = '良好', f'曲面平滑，有轻微波动（η={eta_p99:.6f}）'
        elif smoothness_score >= 0.90:
            quality, description = '可接受', f'平滑性尚可，有可见噪声（η={eta_p99:.6f}）'
        elif smoothness_score >= 0.80:
            quality, description = '较差', f'检测到明显粗糙度（η={eta_p99:.6f}）'
        else:
            quality, description = '不合格', f'严重热力学不一致（η={eta_p99:.6f}）'

        if tail_thickness > 0.5:
            tail_interp = '重尾（存在严重异常点）'
        elif tail_thickness > 0.2:
            tail_interp = '中等尾部'
        else:
            tail_interp = '轻尾（行为良好）'

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
            'data_min':   float(np.min(TC_4d)),
            'data_max':   float(np.max(TC_4d)),
            'data_mean':  float(np.mean(TC_4d)),
            'data_std':   float(np.std(TC_4d)),
            'laplacian_mean':           float(np.mean(laplacian_4d)),
            'laplacian_std':            float(np.std(laplacian_4d)),
            'laplacian_abs_mean':       float(np.mean(laplacian_abs)),
            'laplacian_abs_std':        float(np.std(laplacian_abs)),
            'laplacian_positive_ratio': float(np.sum(laplacian_4d > 0) / laplacian_4d.size),
            'tail_thickness':      tail_thickness,
            'tail_interpretation': tail_interp,
            'p99_to_max_ratio':    float(p99 / (linf_norm + 1e-10)),
            'gradient_magnitude_mean':  float(np.mean(grad_magnitude)),
            'gradient_magnitude_max':   float(np.max(grad_magnitude)),
            'gradient_T_rms':           float(np.sqrt(np.mean(grad_T    ** 2))),
            'gradient_P_rms':           float(np.sqrt(np.mean(grad_P    ** 2))),
            'gradient_CaCl2_rms':        float(np.sqrt(np.mean(grad_CaCl2 ** 2))),
            'gradient_NaCl_rms':        float(np.sqrt(np.mean(grad_NaCl ** 2))),
            'quantile_order_satisfied':    bool(quantile_ok),
            'holder_inequality_satisfied': bool(holder_ok),
            'theory_consistency':          '通过' if (quantile_ok and holder_ok) else '警告',
            'lambda':          self.decay_lambda,
            'grid_resolution': list(self.grid_resolution),
            'actual_shape':    list(TC_4d.shape),
            'total_elements':  int(TC_4d.size),
            'method':          'P99 分位数（CVaR 代理）',
        }

        self.logger.info(f"平滑性评估完成  得分={smoothness_score:.6f}  等级={quality}")
        return smoothness_score, details


# ==============================================================================
# 综合物理评估器
# ==============================================================================

class ThermalConductivityPhysicsEvaluator:
    """NaCl–CaCl₂–H₂O 热导率体系综合物理一致性评估器
    （边界一致性 + 热力学平滑性双支柱框架）。

    与粘度版 ViscosityPhysicsEvaluator 接口完全对称，核心差异：
      - teacher_models 为 2 元组（非 3 元组）
      - mch_range/dec_range → cacl2_range/nacl_range
      - 内部使用 TCBoundaryEvaluator 和 TCSmoothnessEvaluator

    Args:
        teacher_models:  (model_cacl2_h2o, model_nacl_h2o)。
                         model_cacl2_h2o 输入 [T,P,CaCl₂]，model_nacl_h2o 输入 [T,P,NaCl]。
        temp_range:      温度范围 (T_min, T_max)，单位 K。
        pressure_range:  压力范围 (P_min, P_max)，单位 Pa。
        cacl2_range:     CaCl₂ 组分范围 (min, max)，用于 nacl_zero 边界扫描。
        nacl_range:      NaCl 组分范围 (min, max)，用于平滑性网格。
        composition_total: H₂O + NaCl + CaCl₂ 总和，默认 1.0。
        boundary_decay_lambda:   边界评分衰减系数。
        smoothness_decay_lambda: 平滑性评分衰减系数。
        n_boundary_samples:      每条边界采样点数。
        grid_resolution:         平滑性评估 4D 网格分辨率 (n_T, n_P, n_H₂O, n_NaCl)。
        log_level:               日志级别。
    """

    def __init__(
        self,
        teacher_models: Tuple[LowDimEnsemble, LowDimEnsemble],
        temp_range:     Tuple[float, float] = (290.0, 570.0),
        pressure_range: Tuple[float, float] = (5e6, 5e7),
        cacl2_range:    Tuple[float, float] = (0.0, 0.05),
        nacl_range:     Tuple[float, float] = (0.0, 0.05),
        composition_total:       float = 1.0,
        boundary_decay_lambda:   float = 5.0,
        smoothness_decay_lambda: float = 15.0,
        n_boundary_samples:      int   = 100,
        grid_resolution: Tuple[int, int, int, int] = (20, 20, 20, 20),
        log_level: int = logging.INFO,
    ):
        self.logger = get_logger(self.__class__.__name__, log_level)
        model_cacl2_h2o, model_nacl_h2o = teacher_models

        self.boundary_evaluator = TCBoundaryEvaluator(
            model_cacl2_h2o=model_cacl2_h2o,
            model_nacl_h2o=model_nacl_h2o,
            temp_range=temp_range,
            pressure_range=pressure_range,
            cacl2_range=cacl2_range,
            nacl_range=nacl_range,
            composition_total=composition_total,
            decay_lambda=boundary_decay_lambda,
            n_samples=n_boundary_samples,
            log_level=log_level,
        )

        self.smoothness_evaluator = TCSmoothnessEvaluator(
            temp_range=temp_range,
            pressure_range=pressure_range,
            cacl2_range=cacl2_range,
            nacl_range=nacl_range,
            grid_resolution=grid_resolution,
            smoothness_decay_lambda=smoothness_decay_lambda,
            log_level=log_level,
        )

        self.logger.info(
            "ThermalConductivityPhysicsEvaluator 初始化完成（双支柱框架，2 条有效边界）"
        )

    def evaluate_full(self, trainer: Any) -> Tuple[float, Dict[str, Any]]:
        """完整物理一致性评估。overall_score = 0.5 * boundary + 0.5 * smoothness。"""
        self.logger.info("综合物理一致性评估开始（双支柱框架）")

        self.logger.info("支柱一：边界一致性评估（nacl_zero | cacl2_zero）")
        boundary_results = self.boundary_evaluator.evaluate_parl_boundary(trainer)
        boundary_score   = boundary_results['combined']['boundary_score']

        self.logger.info("支柱二：热力学平滑性评估")
        smoothness_score, smoothness_details = self.smoothness_evaluator.evaluate_smoothness(trainer)

        overall_score = 0.5 * boundary_score + 0.5 * smoothness_score

        self.logger.info(
            f"综合评估结果  边界={boundary_score:.6f}  "
            f"平滑性={smoothness_score:.6f}  综合={overall_score:.6f}"
        )

        return overall_score, {
            'boundary':         boundary_results,
            'smoothness':       smoothness_details,
            'overall_score':    float(overall_score),
            'boundary_score':   float(boundary_score),
            'smoothness_score': float(smoothness_score),
        }

    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """生成格式化评估报告字符串。"""
        bd = results['boundary']['combined']
        sm = results['smoothness']
        lines = [
            '=' * 70,
            'NaCl–CaCl₂–H₂O 热导率体系物理一致性评估报告',
            '=' * 70,
            '',
            f"综合得分: {results['overall_score']:.6f}",
            '',
            '支柱一：边界一致性（2 条有效边界，h2o_zero 跳过）',
            '-' * 70,
            f"  边界得分:              {bd['boundary_score']:.6f}",
            f"  NaCl=0  边界 NRMSE:   {bd['nrmse_nacl_zero']:.6f}",
            f"  CaCl₂=0 边界 NRMSE:   {bd['nrmse_cacl2_zero']:.6f}",
            f"  总误差:                {bd['total_error']:.6f}",
            '',
            '支柱二：热力学平滑性',
            '-' * 70,
            f"  平滑性得分:          {sm['smoothness_score']:.6f}",
            f"  归一化粗糙度 η(P99): {sm['normalized_roughness']:.6f}",
            f"  质量等级:            {sm['quality']}",
            f"  描述:                {sm['description']}",
            '',
            '=' * 70,
        ]
        return '\n'.join(lines)


# ==============================================================================
# 公开接口
# ==============================================================================

__all__ = [
    'calculate_boundary_nrmse',
    'exponential_decay_score',
    'get_logger',
    'DNN',
    'PhysicsConfig',
    'TSTREvaluator',
    'TCBoundaryEvaluator',
    'TCSmoothnessEvaluator',
    'ThermalConductivityPhysicsEvaluator',
]