"""
================================================================================
PC-CVAE K-Fold 消融实验 — NaCl–CaCl₂–H₂O 热导率
================================================================================

数据列：[T, P, CaCl₂, NaCl, TC]
  X[:,2] = CaCl₂ ≡ MCH（第一组分）
  X[:,3] = NaCl  ≡ Dec（第二组分）
  H₂O   ≡ HMN（隐式）

边界类型映射：
  NaCl-H₂O  → 'mch_zero'（CaCl₂=0 边界）
  CaCl₂-H₂O → 'dec_zero'（NaCl=0  边界）
  H₂O=0     → 不添加（LAMBDA_COLLOCATION_HMN=0，非物理）

最佳模型机制：
  X_val/y_val 传给 cvae.fit()，USE_EARLY_STOPPING=False 保证跑满 N_EPOCHS，
  pc_cvae_viscosity.py 的 fit() 始终保存最优 val 权重并在训练结束后恢复。
  因此评估用的是最优验证 checkpoint，而非最后 epoch。
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

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

from pc_cvae_thermal_conductivity import CVAEConfig, CVAEPhysicsModel, LowDimInfo
from low_dim_model import LowDimEnsemble
from utils_thermal_conductivity import ThermalConductivityPhysicsEvaluator

warnings.filterwarnings('ignore')

# ── 可视化 T/P 网格 ─────────────────────────────────────────────────
VIZ_T_LIST     = [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15]
VIZ_P_LIST     = [5e6, 1e7, 2e7, 3e7, 5e7]
VIZ_GRID_N     = 40
VIZ_BOUNDARY_N = 80
T_TOL_REL      = 0.04
P_TOL_REL      = 0.20


# ==============================================================================
# 工具
# ==============================================================================

def get_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger

logger = get_logger(__name__)


def load_tc_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    df = pd.read_excel(filepath, engine='openpyxl')
    if df.shape[1] < 5:
        raise ValueError(f"期望 5 列 [T,P,CaCl₂,NaCl,TC]，实际: {df.shape[1]}")
    return df.iloc[:, :4].values.astype(np.float32), df.iloc[:, 4].values.astype(np.float32)


def move_to_device(m, d):
    m.to(d); m.device = d; return m


def create_k_folds(X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [{'fold_idx': fi, 'X_train': X[tr], 'y_train': y[tr],
             'X_val': X[va], 'y_val': y[va]}
            for fi, (tr, va) in enumerate(kf.split(X))]


# ==============================================================================
# CVAE 适配器
# ==============================================================================

class _CVAEWrapper:
    """trainer.predict(X, return_original_scale) 接口适配器。"""
    def __init__(self, cvae): self._cvae = cvae
    def predict(self, X, return_original_scale=True): return self._cvae.predict(X)


# ==============================================================================
# 可视化：每个 (T,P) 一张独立 3D HTML（与基线版完全对称）
# ==============================================================================

class FoldVisualizer:
    """
    每张图：CaCl₂（x）× NaCl（y）× TC（z），轴从 0 开始使边界线可见。
      ① CVAE 预测曲面
      ② nacl_zero 边界（NaCl=0，y=0 面）：实线=CVAE，虚线=教师 CaCl₂-H₂O
      ③ cacl2_zero 边界（CaCl₂=0，x=0 面）：实线=CVAE，虚线=教师 NaCl-H₂O
      ④ 真实数据散点（train/near/far 分色）
    """

    def __init__(self, viz_dir, fold_idx, cacl2_range, nacl_range,
                 t_list=None, p_list=None,
                 grid_n=VIZ_GRID_N, boundary_n=VIZ_BOUNDARY_N,
                 log_level=logging.INFO):
        self.viz_dir     = Path(viz_dir)
        self.fold_idx    = fold_idx
        self.cacl2_range = cacl2_range
        self.nacl_range  = nacl_range
        self.t_list      = t_list or VIZ_T_LIST
        self.p_list      = p_list or VIZ_P_LIST
        self.grid_n      = grid_n
        self.bn          = boundary_n
        self.logger      = get_logger(self.__class__.__name__, log_level)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, wrapper, X_train, y_train, X_near, y_near, X_far, y_far,
                     model_cacl2_h2o=None, model_nacl_h2o=None):
        if not PLOTLY_OK:
            self.logger.warning("plotly 未安装，跳过可视化"); return

        all_X  = np.vstack([X_train, X_near, X_far])
        all_y  = np.concatenate([y_train, y_near, y_far])
        splits = (['train']*len(X_train) + ['near']*len(X_near) + ['far']*len(X_far))

        n = len(self.t_list) * len(self.p_list)
        self.logger.info(f"生成 Fold {self.fold_idx} 可视化：{n} 张图")

        for T_val in self.t_list:
            for P_val in self.p_list:
                tag = f"T{T_val:.0f}K_P{P_val:.0e}Pa"
                try:
                    fig = self._one_fig(wrapper, all_X, all_y, splits,
                                        T_val, P_val, model_cacl2_h2o, model_nacl_h2o)
                    out = self.viz_dir / f'surface_{tag}.html'
                    fig.write_html(str(out))
                    self.logger.info(f"  ✓ {out.name}")
                except Exception as e:
                    self.logger.warning(f"  ✗ {tag}: {e}")

    def _one_fig(self, wrapper, all_X, all_y, splits,
                 T_val, P_val, m_cacl2, m_nacl):
        # 轴从 0 开始，使 x=0 和 y=0 的边界线落在曲面边缘可见
        cacl2_g = np.linspace(0, self.cacl2_range[1], self.grid_n)
        nacl_g  = np.linspace(0, self.nacl_range[1],  self.grid_n)
        C2, N   = np.meshgrid(cacl2_g, nacl_g)

        X_surf = np.column_stack([
            np.full(C2.size, T_val), np.full(C2.size, P_val),
            C2.flatten(), N.flatten(),
        ]).astype(np.float32)
        TC_surf = wrapper.predict(X_surf).flatten().reshape(self.grid_n, self.grid_n)

        fig = go.Figure()

        # ── ① 预测曲面 ─────────────────────────────────────────────
        fig.add_trace(go.Surface(
            x=C2, y=N, z=TC_surf,
            colorscale='Viridis', opacity=0.70, name='CVAE 预测曲面',
            showscale=True,
            colorbar=dict(title='TC (W/m·K)', thickness=16, len=0.6),
            hovertemplate='CaCl₂=%{x:.5f}  NaCl=%{y:.5f}<br>TC=%{z:.5f}'
                          '<extra>CVAE</extra>',
        ))

        # ── ② nacl_zero 边界（NaCl=0，y=0 面）─────────────────────
        cacl2_scan = np.linspace(0, self.cacl2_range[1], self.bn).astype(np.float32)
        X_b1 = np.column_stack([
            np.full(self.bn, T_val), np.full(self.bn, P_val),
            cacl2_scan, np.zeros(self.bn),   # X[:,2]=CaCl₂, X[:,3]=NaCl=0
        ]).astype(np.float32)
        tc_cvae_b1 = wrapper.predict(X_b1).flatten()

        fig.add_trace(go.Scatter3d(
            x=cacl2_scan, y=np.zeros(self.bn), z=tc_cvae_b1,
            mode='lines', line=dict(color='deepskyblue', width=7),
            name='CVAE: NaCl=0',
            hovertemplate='CaCl₂=%{x:.5f}  NaCl=0<br>TC=%{z:.5f}'
                          '<extra>CVAE NaCl=0</extra>',
        ))
        if m_cacl2 is not None:
            # 教师模型取 [T,P,CaCl₂] = X_b1[:,[0,1,2]]
            tc_ref_b1 = m_cacl2.predict(X_b1[:, [0, 1, 2]]).flatten()
            fig.add_trace(go.Scatter3d(
                x=cacl2_scan, y=np.zeros(self.bn), z=tc_ref_b1,
                mode='lines', line=dict(color='red', width=5, dash='dash'),
                name='教师: NaCl=0 (CaCl₂-H₂O)',
                hovertemplate='CaCl₂=%{x:.5f}  NaCl=0<br>TC_ref=%{z:.5f}'
                              '<extra>教师 NaCl=0</extra>',
            ))

        # ── ③ cacl2_zero 边界（CaCl₂=0，x=0 面）───────────────────
        nacl_scan = np.linspace(0, self.nacl_range[1], self.bn).astype(np.float32)
        X_b2 = np.column_stack([
            np.full(self.bn, T_val), np.full(self.bn, P_val),
            np.zeros(self.bn), nacl_scan,    # X[:,2]=CaCl₂=0, X[:,3]=NaCl
        ]).astype(np.float32)
        tc_cvae_b2 = wrapper.predict(X_b2).flatten()

        fig.add_trace(go.Scatter3d(
            x=np.zeros(self.bn), y=nacl_scan, z=tc_cvae_b2,
            mode='lines', line=dict(color='lime', width=7),
            name='CVAE: CaCl₂=0',
            hovertemplate='CaCl₂=0  NaCl=%{y:.5f}<br>TC=%{z:.5f}'
                          '<extra>CVAE CaCl₂=0</extra>',
        ))
        if m_nacl is not None:
            # 教师模型取 [T,P,NaCl] = X_b2[:,[0,1,3]]
            tc_ref_b2 = m_nacl.predict(X_b2[:, [0, 1, 3]]).flatten()
            fig.add_trace(go.Scatter3d(
                x=np.zeros(self.bn), y=nacl_scan, z=tc_ref_b2,
                mode='lines', line=dict(color='orange', width=5, dash='dash'),
                name='教师: CaCl₂=0 (NaCl-H₂O)',
                hovertemplate='CaCl₂=0  NaCl=%{y:.5f}<br>TC_ref=%{z:.5f}'
                              '<extra>教师 CaCl₂=0</extra>',
            ))

        # ── ④ 真实数据散点 ─────────────────────────────────────────
        cmap = {'train': '#1f77b4', 'near': '#ff7f0e', 'far': '#d62728'}
        smap = {'train': 'circle',  'near': 'square',  'far': 'diamond'}
        lmap = {'train': '训练集',   'near': '近外推',   'far': '远外推'}
        t_tol = max(T_val * T_TOL_REL, 5.0)
        p_tol = P_val * P_TOL_REL

        for sp in ('train', 'near', 'far'):
            idx  = [i for i, s in enumerate(splits) if s == sp]
            if not idx: continue
            Xs, ys = all_X[idx], all_y[idx]
            mask = (np.abs(Xs[:,0]-T_val)<t_tol) & (np.abs(Xs[:,1]-P_val)<p_tol)
            if mask.sum() == 0: continue
            yp = wrapper.predict(Xs[mask]).flatten()
            fig.add_trace(go.Scatter3d(
                x=Xs[mask,2], y=Xs[mask,3], z=ys[mask],
                mode='markers',
                marker=dict(size=6, color=cmap[sp], symbol=smap[sp],
                            line=dict(width=0.8, color='white')),
                name=lmap[sp],
                customdata=np.column_stack([Xs[mask,0], Xs[mask,1], yp, ys[mask]-yp]),
                hovertemplate=('CaCl₂=%{x:.5f}  NaCl=%{y:.5f}<br>TC_true=%{z:.5f}<br>'
                               'T=%{customdata[0]:.1f}K  P=%{customdata[1]:.2e}Pa<br>'
                               'TC_pred=%{customdata[2]:.5f}  res=%{customdata[3]:+.5f}'
                               f'<extra>{lmap[sp]}</extra>'),
            ))

        # ── R² 计算 ──────────────────────────────────────────────────
        all_mask = ((np.abs(all_X[:,0]-T_val)<t_tol) &
                    (np.abs(all_X[:,1]-P_val)<p_tol))
        r2_loc = float('nan')
        if all_mask.sum() >= 2:
            r2_loc = float(r2_score(all_y[all_mask],
                                    wrapper.predict(all_X[all_mask]).flatten()))

        def _r2(pred, ref):
            try: return float(r2_score(ref, pred))
            except: return float('nan')

        r2_b1 = r2_b2 = float('nan')
        if m_cacl2 is not None:
            r2_b1 = _r2(tc_cvae_b1, m_cacl2.predict(X_b1[:,[0,1,2]]).flatten())
        if m_nacl is not None:
            r2_b2 = _r2(tc_cvae_b2, m_nacl.predict(X_b2[:,[0,1,3]]).flatten())

        fig.update_layout(
            title=dict(
                text=(f'Fold {self.fold_idx} — PC-CVAE  '
                      f'T={T_val:.1f}K  P={P_val:.2e}Pa<br>'
                      f'<sup>局部R²={r2_loc:.4f} | '
                      f'NaCl=0边界R²={r2_b1:.3f} | '
                      f'CaCl₂=0边界R²={r2_b2:.3f} | '
                      f'实线=CVAE  虚线=教师</sup>'),
                x=0.5),
            scene=dict(
                xaxis=dict(title='CaCl₂ (mole frac)',
                           range=[0, self.cacl2_range[1]]),
                yaxis=dict(title='NaCl (mole frac)',
                           range=[0, self.nacl_range[1]]),
                zaxis=dict(title='TC (W/m·K)'),
                aspectmode='manual',
                aspectratio=dict(x=1.4, y=1.0, z=0.8),
            ),
            legend=dict(x=0.01, y=0.99,
                        bgcolor='rgba(255,255,255,0.7)',
                        bordercolor='lightgray', borderwidth=1),
            width=1000, height=720,
            margin=dict(l=0, r=0, t=80, b=0),
        )
        return fig


# ==============================================================================
# 配置
# ==============================================================================

@dataclass
class CVAEExperimentConfig:

    data_dir: Path = PROJECT_ROOT / 'data' / 'Thermal_conductivity' / 'split_by_temperature'
    train_data_file:  str = 'interpolation domain.xlsx'
    near_extrap_file: str = 'near-range extrapolation.xlsx'
    far_extrap_file:  str = 'far-range extrapolation.xlsx'

    models_dir: Path = PROJECT_ROOT / 'models' / 'Low_dim_model' / 'Thermal_conductivity'
    cacl2_h2o_model_file: str = 'CaCl2-H2O.pth'
    nacl_h2o_model_file:  str = 'NaCl-H2O.pth'

    cvae_config: CVAEConfig = field(default_factory=lambda: CVAEConfig(
        LATENT_DIM=2,
        HIDDEN_DIMS=[128, 256, 256, 128],
        DROPOUT=0.1,
        LEARNING_RATE=1e-3,
        BATCH_SIZE=64,
        N_EPOCHS=500,
        WEIGHT_DECAY=1e-5,
        LAMBDA_KL=0.001,
        LAMBDA_COLLOCATION_MCH=0.5,
        LAMBDA_COLLOCATION_DEC=0.5,
        LAMBDA_COLLOCATION_HMN=0.0,
        N_COLLOCATION_POINTS=64,
        COLLOCATION_T_RANGE=(290.0, 570.0),
        COLLOCATION_P_RANGE=(5e6, 5e7),
        Z_LOW=-2.0, Z_HIGH=2.0, Z_COLLOC_WIDTH=0.5,
        PHI_HIDDEN_DIMS=[64, 64],
        LAMBDA_CYCLE=1.0, N_CYCLE_POINTS=64,
        CYCLE_T_RANGE=(290.0, 570.0),
        CYCLE_P_RANGE=(5e6, 5e7),
        USE_EARLY_STOPPING=False,   # 跑满 epoch，但恢复最优 val 权重
        USE_LR_SCHEDULER=True,
        LR_SCHEDULER_TYPE='cosine',
        LR_MIN=1e-5,
        DEVICE='auto', VERBOSE=False,
    ))

    t_min: float = 290.0;  t_max: float = 570.0
    p_min: float = 5e6;    p_max: float = 5e7
    # ★ 三元实测盐组分摩尔分数范围
    cacl2_range: Tuple[float, float] = (0.0017, 0.0127)
    nacl_range:  Tuple[float, float] = (0.0063, 0.0482)

    k_folds: int = 5
    kfold_random_state: int = 42

    output_dir: Path = PROJECT_ROOT / 'results' / 'tc' / 'ablation' / 'CVAE_results'
    save_predictions:  bool = True
    save_metrics:      bool = True
    save_cvae_history: bool = True
    excel_prefix:      str  = 'cvae_'

    # 可视化
    viz_t_list:   List[float] = field(default_factory=lambda: VIZ_T_LIST)
    viz_p_list:   List[float] = field(default_factory=lambda: VIZ_P_LIST)
    generate_viz: bool = True

    device:    str = 'auto'
    log_level: int = logging.INFO

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cvae_config.DEVICE = self.device
        self.train_data_path  = self.data_dir   / self.train_data_file
        self.near_extrap_path = self.data_dir   / self.near_extrap_file
        self.far_extrap_path  = self.data_dir   / self.far_extrap_file
        self.cacl2_h2o_path   = self.models_dir / self.cacl2_h2o_model_file
        self.nacl_h2o_path    = self.models_dir / self.nacl_h2o_model_file


# ==============================================================================
# 单折执行器
# ==============================================================================

class SingleFoldRunner:

    def __init__(self, config, model_cacl2_h2o, model_nacl_h2o):
        self.config          = config
        self.logger          = get_logger(self.__class__.__name__, config.log_level)
        self.device          = torch.device(config.device)
        self.model_cacl2_h2o = model_cacl2_h2o
        self.model_nacl_h2o  = model_nacl_h2o

    def run(self, fold_idx, X_train, y_train, X_val, y_val,
            X_near, y_near, X_far, y_far, fold_dir):
        self.logger.info("=" * 70)
        self.logger.info(f"Fold {fold_idx}")
        self.logger.info("=" * 70)

        # Step 1: 训练
        # ★ X_val/y_val 传入 fit()，触发 best checkpoint 机制
        # ★ USE_EARLY_STOPPING=False → 跑满 N_EPOCHS 但恢复最优 val 权重
        self.logger.info("[Step 1] 训练 PC-CVAE（热导率版，固定 epoch）")
        cvae, history = self._train_cvae(X_train, y_train, X_val, y_val)

        # Step 2: 评估（使用最优 val checkpoint）
        self.logger.info("[Step 2] 经 φ 头直接评估")
        metrics = self._evaluate_cvae(cvae, X_train, y_train, X_val, y_val,
                                      X_near, y_near, X_far, y_far)

        # Step 3: 物理一致性
        self.logger.info("[Step 3] 物理一致性评估（双支柱框架）")
        phys = self._compute_physics_metrics(cvae)

        # Step 4: 可视化
        if self.config.generate_viz:
            self.logger.info("[Step 4] 生成 3D 可视化（CaCl₂×NaCl 轴）")
            wrapper = _CVAEWrapper(cvae)
            viz_dir = fold_dir / 'visualizations'
            viz = FoldVisualizer(
                viz_dir=viz_dir,
                fold_idx=fold_idx,
                cacl2_range=self.config.cacl2_range,
                nacl_range=self.config.nacl_range,
                t_list=self.config.viz_t_list,
                p_list=self.config.viz_p_list,
                log_level=self.config.log_level,
            )
            viz.generate_all(
                wrapper, X_train, y_train, X_near, y_near, X_far, y_far,
                model_cacl2_h2o=self.model_cacl2_h2o,
                model_nacl_h2o=self.model_nacl_h2o,
            )

        self._save_fold_results(fold_idx, fold_dir, metrics, phys, history, cvae)

        return {
            'fold_idx':           fold_idx,
            'metrics':            metrics,
            'physics_score':      phys.get('physics_score',      None),
            'physics_boundary':   phys.get('physics_boundary',   None),
            'physics_smoothness': phys.get('physics_smoothness', None),
        }

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    def _train_cvae(self, X_train, y_train, X_val, y_val):
        low_dim_list = None
        if all(m is not None for m in (self.model_cacl2_h2o, self.model_nacl_h2o)):
            low_dim_list = [
                LowDimInfo(model=self.model_nacl_h2o,  name='NaCl-H2O',
                           boundary_type='mch_zero'),   # CaCl₂=0
                LowDimInfo(model=self.model_cacl2_h2o, name='CaCl2-H2O',
                           boundary_type='dec_zero'),   # NaCl=0
            ]
        else:
            self.logger.warning("边界模型缺失 — 配点约束已禁用")

        cvae = CVAEPhysicsModel(config=self.config.cvae_config)

        # ★ 关键：传入 X_val/y_val 触发 best val checkpoint
        #    USE_EARLY_STOPPING=False → 跑满 epoch，训练结束自动恢复最优权重
        history = cvae.fit(
            X=X_train,
            y=y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train,
            low_dim_list=low_dim_list,
            X_val=X_val,
            y_val=y_val.reshape(-1, 1) if y_val is not None and y_val.ndim == 1 else y_val,
        )

        # 打印损失量级
        tr = history.get('train_loss', [])
        if tr:
            self.logger.info("=" * 80)
            self.logger.info("损失分项量级分析（各关键轮次）")
            self.logger.info("=" * 80)
            check = sorted(set([0, len(tr)//4, len(tr)//2, 3*len(tr)//4, len(tr)-1]))
            cfg   = self.config.cvae_config
            self.logger.info(
                f"{'Epoch':>7}  {'total':>9}  {'recon':>9}  {'kl*λ':>9}  "
                f"{'cycle':>9}  {'nacl×λ':>9}  {'cacl2×λ':>10}  "
                f"{'raw_nacl':>10}  {'raw_cacl2':>10}")
            self.logger.info("-" * 90)
            for ep in check:
                def _g(k): v=history.get(k,[]); return v[ep] if ep<len(v) else float('nan')
                rm = _g('train_colloc_mch');  rd = _g('train_colloc_dec')
                self.logger.info(
                    f"{ep+1:>7}  {tr[ep]:>9.5f}  {_g('train_recon'):>9.5f}  "
                    f"{_g('train_kl')*cfg.LAMBDA_KL:>9.5f}  {_g('train_cycle'):>9.5f}  "
                    f"{rm*cfg.LAMBDA_COLLOCATION_MCH:>9.5f}  "
                    f"{rd*cfg.LAMBDA_COLLOCATION_DEC:>10.5f}  "
                    f"{rm:>10.5f}  {rd:>10.5f}")
            self.logger.info("-" * 90)

            ep = len(tr)-1
            total = tr[ep]
            def _g(k): v=history.get(k,[]); return v[ep] if ep<len(v) else 0
            items = [
                ('recon',   _g('train_recon')),
                ('nacl×λ',  _g('train_colloc_mch') * cfg.LAMBDA_COLLOCATION_MCH),
                ('cacl2×λ', _g('train_colloc_dec') * cfg.LAMBDA_COLLOCATION_DEC),
                ('cycle',   _g('train_cycle')),
                ('kl×λ',    _g('train_kl') * cfg.LAMBDA_KL),
            ]
            self.logger.info("末轮各分项占总损失百分比：")
            for name, val in items:
                pct = val / max(abs(total), 1e-10) * 100
                self.logger.info(f"  {name:<10}: {val:.5f}  ({pct:5.1f}%)  "
                                 f"{'█'*int(pct/2)}")
            self.logger.info("-" * 80)
            self.logger.info(
                f"  训练完成 total={tr[-1]:.5f}  "
                f"recon={_g('train_recon'):.5f}  "
                f"kl={_g('train_kl'):.5f}  "
                f"cycle={_g('train_cycle'):.5f}  "
                f"nacl={_g('train_colloc_mch'):.5f}  "
                f"cacl2={_g('train_colloc_dec'):.5f}")
        return cvae, history

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------

    def _evaluate_cvae(self, cvae, X_train, y_train, X_val, y_val,
                       X_near, y_near, X_far, y_far):
        splits = {'train':(X_train,y_train), 'val':(X_val,y_val),
                  'near':(X_near,y_near),    'far':(X_far,y_far)}
        preds = {s: cvae.predict(X).flatten() for s,(X,_) in splits.items()}
        trues = {s: y.flatten() for s,(_,y) in splits.items()}
        met: Dict[str, float] = {}
        for s in ('train','val','near','far'):
            met[f'{s}_r2']   = float(r2_score(trues[s], preds[s]))
            met[f'{s}_rmse'] = float(np.sqrt(mean_squared_error(trues[s], preds[s])))
            met[f'{s}_mae']  = float(mean_absolute_error(trues[s], preds[s]))
        self.logger.info("-"*60)
        self.logger.info("预测性能：")
        for s in ('train','val','near','far'):
            self.logger.info(f"  {s:<5}  R²= {met[f'{s}_r2']:.4f}  "
                             f"RMSE= {met[f'{s}_rmse']:.5f}  "
                             f"MAE= {met[f'{s}_mae']:.5f}")
        self.logger.info(f"  near vs train 衰减: {met['near_r2']-met['train_r2']:+.4f}")
        self.logger.info(f"  far  vs train 衰减: {met['far_r2'] -met['train_r2']:+.4f}")
        self.logger.info("-"*60)
        return {'metrics': met, 'predictions': preds, 'true_values': trues}

    # ------------------------------------------------------------------
    # Step 3
    # ------------------------------------------------------------------

    def _compute_physics_metrics(self, cvae):
        if any(m is None for m in (self.model_cacl2_h2o, self.model_nacl_h2o)):
            self.logger.warning("边界模型缺失 — 跳过物理评估"); return {}
        try:
            ev = ThermalConductivityPhysicsEvaluator(
                teacher_models=(self.model_cacl2_h2o, self.model_nacl_h2o),
                temp_range=(self.config.t_min, self.config.t_max),
                pressure_range=(self.config.p_min, self.config.p_max),
                cacl2_range=self.config.cacl2_range,
                nacl_range=self.config.nacl_range,
            )
            overall, res = ev.evaluate_full(_CVAEWrapper(cvae))
            bs = res.get('boundary_score',   float('nan'))
            ss = res.get('smoothness_score', float('nan'))
            self.logger.info(f"  physics={overall:.4f}  "
                             f"boundary={bs:.4f}  smoothness={ss:.4f}")
            return {'physics_score': float(overall),
                    'physics_boundary': float(bs),
                    'physics_smoothness': float(ss)}
        except Exception as e:
            import traceback
            self.logger.error(f"物理评估失败: {e}\n{traceback.format_exc()}")
            return {}

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------

    def _save_fold_results(self, fold_idx, fold_dir, metrics, phys, history, cvae):
        fold_dir.mkdir(parents=True, exist_ok=True)
        md = fold_dir/'model'; md.mkdir(exist_ok=True)
        cvae.save(str(md/'cvae.pth'))

        if self.config.save_metrics:
            inner = metrics.get('metrics', metrics)
            ps = phys.get('physics_score',      float('nan'))
            bs = phys.get('physics_boundary',   float('nan'))
            ss = phys.get('physics_smoothness', float('nan'))
            rows = [
                ['Train R²',  inner.get('train_r2',  float('nan'))],
                ['Train RMSE',inner.get('train_rmse',float('nan'))],
                ['Train MAE', inner.get('train_mae', float('nan'))],
                ['Val R²',    inner.get('val_r2',    float('nan'))],
                ['Val RMSE',  inner.get('val_rmse',  float('nan'))],
                ['Val MAE',   inner.get('val_mae',   float('nan'))],
                ['Near R²',   inner.get('near_r2',   float('nan'))],
                ['Near RMSE', inner.get('near_rmse', float('nan'))],
                ['Near MAE',  inner.get('near_mae',  float('nan'))],
                ['Far R²',    inner.get('far_r2',    float('nan'))],
                ['Far RMSE',  inner.get('far_rmse',  float('nan'))],
                ['Far MAE',   inner.get('far_mae',   float('nan'))],
                ['Physics Score',            ps],
                ['Boundary Consistency',     bs],
                ['Thermodynamic Smoothness', ss],
            ]
            pd.DataFrame(rows, columns=['Metric','Value'])\
              .to_excel(fold_dir/f'{self.config.excel_prefix}metrics.xlsx',
                        index=False, engine='openpyxl')

        if self.config.save_predictions:
            for sp in ('train','val','near','far'):
                yt = metrics['true_values'][sp].flatten()
                yp = metrics['predictions'][sp].flatten()
                pd.DataFrame({'y_true':yt,'y_pred':yp,'residual':yt-yp})\
                  .to_excel(fold_dir/f'{self.config.excel_prefix}{sp}_predictions.xlsx',
                            index=False, engine='openpyxl')

        if self.config.save_cvae_history:
            tl = history.get('train_loss', [])
            if tl:
                rows = []
                for ep, total in enumerate(tl):
                    def _g(k): v=history.get(k,[]); return v[ep] if ep<len(v) else float('nan')
                    rows.append({'epoch':ep,'train_total':total,
                                 'train_recon':_g('train_recon'),
                                 'train_kl':_g('train_kl'),
                                 'train_cycle':_g('train_cycle'),
                                 'train_colloc_nacl':_g('train_colloc_mch'),
                                 'train_colloc_cacl2':_g('train_colloc_dec')})
                df = pd.DataFrame(rows)
                if history.get('val_loss'):
                    df['val_total'] = pd.Series(history['val_loss'])
                df.to_excel(fold_dir/f'{self.config.excel_prefix}cvae_history.xlsx',
                            index=False, engine='openpyxl')


# ==============================================================================
# K-Fold 管理器
# ==============================================================================

class KFoldExperimentManager:

    def __init__(self, config: CVAEExperimentConfig, n_folds: int = 5):
        if n_folds < 2: raise ValueError("n_folds >= 2")
        self.config      = config
        self.n_folds     = n_folds
        self.logger      = get_logger(self.__class__.__name__, config.log_level)
        self.output_dir  = config.output_dir
        self.all_results: List[Dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cacl2_h2o = None
        self.model_nacl_h2o  = None
        self._load_boundary_models()

    def _load_boundary_models(self):
        try:
            dev   = torch.device(self.config.device)
            paths = {'cacl2_h2o': self.config.cacl2_h2o_path,
                     'nacl_h2o':  self.config.nacl_h2o_path}
            miss  = [k for k,p in paths.items() if not p.exists()]
            if miss:
                self.logger.warning(f"边界模型缺失: {miss}"); return
            self.model_cacl2_h2o = move_to_device(
                LowDimEnsemble.load(str(paths['cacl2_h2o'])), dev)
            self.model_nacl_h2o  = move_to_device(
                LowDimEnsemble.load(str(paths['nacl_h2o'])),  dev)
            for name, m in [('CaCl2-H2O', self.model_cacl2_h2o),
                             ('NaCl-H2O',  self.model_nacl_h2o)]:
                if m.is_scaler_fitted:
                    xm = m.x_mean.cpu().numpy().flatten()
                    self.logger.info(f"  {name} x_mean[2]={xm[2]:.5f}  "
                                     f"（盐浓度均值，应为 ~0.01，非 H₂O ~0.97）")
            self.logger.info(
                f"两个边界模型已加载 → {dev}  "
                f"(CaCl₂-H₂O: dec_zero/nacl_zero | NaCl-H₂O: mch_zero/cacl2_zero)")
        except Exception as e:
            self.logger.error(f"加载边界模型失败: {e}")

    def run_all_folds(self):
        self.logger.info("\n" + "█"*70)
        self.logger.info("PC-CVAE K-Fold 实验（NaCl–CaCl₂–H₂O 热导率）")
        self.logger.info("█"*70)
        self.logger.info(f"时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  K={self.n_folds}")
        cfg = self.config.cvae_config
        self.logger.info("\n" + "-"*70 + "\n超参数\n" + "-"*70)
        self.logger.info(f"latent={cfg.LATENT_DIM} epoch={cfg.N_EPOCHS} lr={cfg.LEARNING_RATE}")
        self.logger.info(f"λ_KL={cfg.LAMBDA_KL} λ_nacl={cfg.LAMBDA_COLLOCATION_MCH} "
                         f"λ_cacl2={cfg.LAMBDA_COLLOCATION_DEC} λ_cycle={cfg.LAMBDA_CYCLE}")
        self.logger.info(f"Colloc T={cfg.COLLOCATION_T_RANGE}  P={cfg.COLLOCATION_P_RANGE}")
        self.logger.info(f"Cycle  T={cfg.CYCLE_T_RANGE}  P={cfg.CYCLE_P_RANGE}")
        self.logger.info(f"最佳模型: X_val 传入 fit()，USE_EARLY_STOPPING=False，"
                         f"训练结束恢复最优 val checkpoint")

        ts = time.time()
        X_tp, y_tp = load_tc_data(self.config.train_data_path)
        X_nr, y_nr = load_tc_data(self.config.near_extrap_path)
        X_fr, y_fr = load_tc_data(self.config.far_extrap_path)
        self.logger.info(f"训练池:{len(X_tp)}  近域:{len(X_nr)}  远域:{len(X_fr)}")
        self._near_data = (X_nr, y_nr)
        self._far_data  = (X_fr, y_fr)

        folds = create_k_folds(X_tp, y_tp, self.n_folds, self.config.kfold_random_state)
        for f in folds:
            self.logger.info(f"  Fold {f['fold_idx']}: train={len(f['X_train'])} "
                             f"val={len(f['X_val'])}")

        for fd in folds:
            fi = fd['fold_idx']
            self.logger.info(f"\n{'█'*70}\nFold {fi+1}/{self.n_folds}\n{'█'*70}")
            fold_dir = self.output_dir / f'fold_{fi}'
            fold_dir.mkdir(exist_ok=True)

            runner = SingleFoldRunner(self.config, self.model_cacl2_h2o, self.model_nacl_h2o)
            result = runner.run(
                fi,
                fd['X_train'], fd['y_train'],
                fd['X_val'],   fd['y_val'],
                X_nr, y_nr, X_fr, y_fr, fold_dir,
            )
            self.all_results.append(result)
            self.logger.info(f"Fold {fi} 完成")

        self._summary()
        self._best_eval()
        self.logger.info(
            f"\n总耗时 {timedelta(seconds=int(time.time()-ts))}  "
            f"结果: {self.output_dir}")

    def _summary(self):
        sd = self.output_dir/'summary'; sd.mkdir(exist_ok=True)
        keys  = ['train_r2','train_rmse','train_mae','val_r2','val_rmse','val_mae',
                 'near_r2','near_rmse','near_mae','far_r2','far_rmse','far_mae']
        names = ['Train R²','Train RMSE','Train MAE','Val R²','Val RMSE','Val MAE',
                 'Near R²','Near RMSE','Near MAE','Far R²','Far RMSE','Far MAE']
        rows = []
        for k,n in zip(keys,names):
            vs = [r['metrics']['metrics'][k] for r in self.all_results]
            rows.append({'Metric':n,'Mean Value':float(np.mean(vs)),
                         'Std':float(np.std(vs,ddof=1)) if len(vs)>1 else 0.0})
        for attr,label in [('physics_score','Physics Score'),
                            ('physics_boundary','Boundary Consistency'),
                            ('physics_smoothness','Thermodynamic Smoothness')]:
            vs = [r.get(attr) for r in self.all_results
                  if r.get(attr) is not None
                  and not np.isnan(float(r.get(attr,float('nan'))))]
            rows.append({'Metric':label,
                         'Mean Value':float(np.mean(vs)) if vs else float('nan'),
                         'Std':float(np.std(vs,ddof=1)) if len(vs)>1 else 0.0})
        pd.DataFrame(rows).to_excel(sd/'summary_metrics.xlsx',index=False,engine='openpyxl')

        cfg = self.config.cvae_config
        sep = '-'*70
        lines = ['='*70,'PC-CVAE K-Fold 汇总 — NaCl–CaCl₂–H₂O 热导率','='*70,
                 f"时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  K={self.n_folds}",
                 '',sep,'超参数',sep,
                 f"latent={cfg.LATENT_DIM} epoch={cfg.N_EPOCHS} lr={cfg.LEARNING_RATE}",
                 f"λ_KL={cfg.LAMBDA_KL} λ_nacl={cfg.LAMBDA_COLLOCATION_MCH} "
                 f"λ_cacl2={cfg.LAMBDA_COLLOCATION_DEC} λ_cycle={cfg.LAMBDA_CYCLE}",
                 f"最佳模型: X_val 传入 fit()，恢复最优 val checkpoint",
                 '',sep,'汇总统计',sep]
        for k,n in zip(keys,names):
            vs = [r['metrics']['metrics'][k] for r in self.all_results]
            lines.append(f"{n:22s}: {np.mean(vs):.6f} ± {np.std(vs,ddof=1):.6f}")
        lines.append('\n物理评估:')
        for attr,n in [('physics_score','Physics Score'),
                       ('physics_boundary','Boundary Consistency'),
                       ('physics_smoothness','Thermodynamic Smoothness')]:
            vs = [r.get(attr) for r in self.all_results
                  if r.get(attr) is not None
                  and not np.isnan(float(r.get(attr,float('nan'))))]
            if vs: lines.append(f"{n:22s}: {np.mean(vs):.6f} ± {np.std(vs,ddof=1):.6f}")
            else:  lines.append(f"{n:22s}: N/A")
        val_r2 = [r['metrics']['metrics']['val_r2'] for r in self.all_results]
        best   = self.all_results[int(np.argmax(val_r2))]
        lines += ['',sep,'最优折（按 Val R²）',sep,
                  f"Fold {best['fold_idx']}  "
                  f"Val={best['metrics']['metrics']['val_r2']:.6f}  "
                  f"Near={best['metrics']['metrics']['near_r2']:.6f}  "
                  f"Far={best['metrics']['metrics']['far_r2']:.6f}",
                  '','='*70]
        report = '\n'.join(lines)
        (sd/'summary_report.txt').write_text(report, encoding='utf-8')
        self.logger.info(f'\n{report}')

    def _best_eval(self):
        val_r2 = [r['metrics']['metrics']['val_r2'] for r in self.all_results]
        fi  = self.all_results[int(np.argmax(val_r2))]['fold_idx']
        mp  = self.output_dir/f'fold_{fi}'/'model'/'cvae.pth'
        if not mp.exists(): return
        cvae = CVAEPhysicsModel.load(str(mp))
        bd   = self.output_dir/'best_model'; bd.mkdir(exist_ok=True)
        self.logger.info(f"\n最佳折: Fold {fi}  Val R²={val_r2[int(np.argmax(val_r2))]:.6f}")
        rows = [['Best Fold',fi],['Val R²',val_r2[int(np.argmax(val_r2))]]]
        for tag, X, ya in [('near',*self._near_data),('far',*self._far_data)]:
            yp = cvae.predict(X).flatten(); yt = ya.flatten()
            r2 = float(r2_score(yt,yp))
            rmse = float(np.sqrt(mean_squared_error(yt,yp)))
            mae  = float(mean_absolute_error(yt,yp))
            label = 'Near-Range' if tag=='near' else 'Far-Range'
            self.logger.info(f"  best {label}: r²={r2:.4f} rmse={rmse:.5f} mae={mae:.5f}")
            rows += [[f'{label} R²',r2],[f'{label} RMSE',rmse],[f'{label} MAE',mae]]
            pd.DataFrame({'y_true':yt,'y_pred':yp,'residual':yt-yp})\
              .to_excel(bd/f'best_{tag}_predictions.xlsx',index=False,engine='openpyxl')
        pd.DataFrame(rows,columns=['Metric','Value'])\
          .to_excel(bd/'best_model_metrics.xlsx',index=False,engine='openpyxl')


# ==============================================================================
# Main
# ==============================================================================

def main():
    config = CVAEExperimentConfig()
    config.k_folds            = 5
    config.kfold_random_state = 42
    config.t_min              = 290.0
    config.t_max              = 570.0
    config.p_min              = 5e6
    config.p_max              = 5e7
    config.cacl2_range        = (0.0017, 0.0127)   # ★ 实测 CaCl₂ 摩尔分数范围
    config.nacl_range         = (0.0063, 0.0482)   # ★ 实测 NaCl  摩尔分数范围
    config.save_predictions   = True
    config.save_metrics       = True
    config.save_cvae_history  = True
    config.generate_viz       = True
    config.log_level          = logging.INFO
    config.viz_t_list = [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15]
    config.viz_p_list = [5e6, 1e7, 2e7, 3e7, 5e7]

    config.cvae_config.LATENT_DIM             = 2
    config.cvae_config.N_EPOCHS               = 500
    config.cvae_config.LEARNING_RATE          = 1e-3
    config.cvae_config.LAMBDA_KL              = 0.001
    config.cvae_config.LAMBDA_COLLOCATION_MCH = 0.5   # cacl2_zero
    config.cvae_config.LAMBDA_COLLOCATION_DEC = 0.5   # nacl_zero
    config.cvae_config.LAMBDA_COLLOCATION_HMN = 0.0
    config.cvae_config.N_COLLOCATION_POINTS   = 64
    config.cvae_config.COLLOCATION_T_RANGE    = (290.0, 570.0)
    config.cvae_config.COLLOCATION_P_RANGE    = (5e6, 5e7)
    config.cvae_config.Z_LOW                  = -2.0
    config.cvae_config.Z_HIGH                 = 2.0
    config.cvae_config.Z_COLLOC_WIDTH         = 0.5
    config.cvae_config.PHI_HIDDEN_DIMS        = [64, 64]
    config.cvae_config.LAMBDA_CYCLE           = 1.0
    config.cvae_config.N_CYCLE_POINTS         = 64
    config.cvae_config.CYCLE_T_RANGE          = (290.0, 570.0)
    config.cvae_config.CYCLE_P_RANGE          = (5e6, 5e7)
    config.cvae_config.USE_EARLY_STOPPING     = False   # 跑满但恢复最优 val 权重
    config.cvae_config.DEVICE                 = config.device

    KFoldExperimentManager(config, n_folds=config.k_folds).run_all_folds()


if __name__ == '__main__':
    main()