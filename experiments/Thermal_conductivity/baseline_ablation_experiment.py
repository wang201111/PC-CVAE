"""
================================================================================
Baseline Ablation Experiment - Thermal Conductivity (NaCl–CaCl₂–H₂O)
================================================================================

数据结构：[T, P, CaCl₂, NaCl, TC]
  X[:,2] = CaCl₂  （第一盐组分，三元实测范围 0.0017–0.0127）
  X[:,3] = NaCl   （第二盐组分，三元实测范围 0.0063–0.0482）

边界约束：
  nacl_zero  (NaCl=0):  X=[T,P,CaCl₂_varied,0]，教师模型取 X[:,[0,1,2]]=[T,P,CaCl₂]
  cacl2_zero (CaCl₂=0): X=[T,P,0,NaCl_varied]， 教师模型取 X[:,[0,1,3]]=[T,P,NaCl]

可视化：每个 (T,P) 一张独立 3D 图，以 CaCl₂ × NaCl 为 x/y 轴。
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from low_dim_model import LowDimEnsemble
from utils_thermal_conductivity import (
    DNN, PhysicsConfig, TSTREvaluator,
    ThermalConductivityPhysicsEvaluator, get_logger,
)

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ── 可视化 T/P 网格（可按需修改）────────────────────────────────────────
VIZ_T_LIST     = [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15]
VIZ_P_LIST     = [5e6, 1e7, 2e7, 3e7, 5e7]
VIZ_GRID_N     = 40    # 曲面网格密度（每轴）
VIZ_BOUNDARY_N = 80    # 边界线采样点数
T_TOL_REL      = 0.04  # 散点筛选：T ± 4%
P_TOL_REL      = 0.20  # 散点筛选：P ± 20%


# ==============================================================================
# 数据加载
# ==============================================================================

def load_tc_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not filepath.exists():
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    data = pd.read_excel(filepath, engine='openpyxl')
    if data.shape[1] < 5:
        raise ValueError(f"期望 5 列 [T,P,CaCl₂,NaCl,TC]，实际: {data.shape[1]}")
    return (data.iloc[:, :4].values.astype(np.float32),
            data.iloc[:,  4].values.astype(np.float32))


def move_to_device(m, d):
    m.to(d); m.device = d; return m


def create_k_folds(X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [{'fold_idx': fi, 'X_train': X[tr], 'y_train': y[tr],
             'X_val': X[va], 'y_val': y[va]}
            for fi, (tr, va) in enumerate(kf.split(X))]


# ==============================================================================
# DNN 适配器
# ==============================================================================

class _DNNWrapper:
    def __init__(self, model, x_scaler, y_scaler, device):
        self._m = model; self._xs = x_scaler
        self._ys = y_scaler; self._dev = device

    def predict(self, X, return_original_scale=True):
        self._m.eval()
        X_sc = self._xs.transform(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            out = self._m(torch.FloatTensor(X_sc).to(self._dev)).cpu().numpy()
        return self._ys.inverse_transform(out)


# ==============================================================================
# 可视化：每个 (T, P) 一张独立 3D 图
# ==============================================================================

class BaselineVisualizer:
    """
    每张图以 CaCl₂（x 轴）× NaCl（y 轴）为组成空间，包含：
      ① 预测曲面（DNN）
      ② nacl_zero  边界线：X=[T,P,CaCl₂_scan,0]，y=0 平面
         教师模型 CaCl₂-H₂O 取 X[:,[0,1,2]]=[T,P,CaCl₂]
      ③ cacl2_zero 边界线：X=[T,P,0,NaCl_scan]，x=0 平面
         教师模型 NaCl-H₂O  取 X[:,[0,1,3]]=[T,P,NaCl]
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
        if not PLOTLY_AVAILABLE:
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
        # ── 构造 CaCl₂ × NaCl 网格 ──────────────────────────────────
        cacl2_g = np.linspace(0, self.cacl2_range[1], self.grid_n)
        nacl_g  = np.linspace(0, self.nacl_range[1],  self.grid_n)
        C2, N   = np.meshgrid(cacl2_g, nacl_g)   # (grid_n × grid_n)

        X_surf = np.column_stack([
            np.full(C2.size, T_val), np.full(C2.size, P_val),
            C2.flatten(), N.flatten(),
        ]).astype(np.float32)
        TC_surf = wrapper.predict(X_surf).flatten().reshape(self.grid_n, self.grid_n)

        fig = go.Figure()

        # ── ① 预测曲面 ──────────────────────────────────────────────
        fig.add_trace(go.Surface(
            x=C2, y=N, z=TC_surf,
            colorscale='Viridis', opacity=0.70,
            name='DNN 预测曲面', showscale=True,
            colorbar=dict(title='TC (W/m·K)', thickness=16, len=0.6),
            hovertemplate='CaCl₂=%{x:.5f}  NaCl=%{y:.5f}<br>TC=%{z:.5f}'
                          '<extra>DNN</extra>',
        ))

        # ── ② nacl_zero 边界（NaCl=0，沿 CaCl₂ 扫描，y=0 平面）────
        cacl2_scan = np.linspace(*self.cacl2_range, self.bn).astype(np.float32)
        X_b1 = np.column_stack([
            np.full(self.bn, T_val), np.full(self.bn, P_val),
            cacl2_scan, np.zeros(self.bn),       # X[:,2]=CaCl₂, X[:,3]=NaCl=0
        ]).astype(np.float32)
        tc_dnn_b1 = wrapper.predict(X_b1).flatten()

        fig.add_trace(go.Scatter3d(
            x=cacl2_scan, y=np.zeros(self.bn), z=tc_dnn_b1,
            mode='lines', line=dict(color='deepskyblue', width=7),
            name='DNN: NaCl=0 边界',
            hovertemplate='CaCl₂=%{x:.5f}  NaCl=0<br>TC=%{z:.5f}<extra>DNN NaCl=0</extra>',
        ))
        if m_cacl2 is not None:
            # 教师模型取 [T, P, CaCl₂] = X_b1[:,[0,1,2]]
            tc_ref_b1 = m_cacl2.predict(X_b1[:, [0, 1, 2]]).flatten()
            fig.add_trace(go.Scatter3d(
                x=cacl2_scan, y=np.zeros(self.bn), z=tc_ref_b1,
                mode='lines', line=dict(color='red', width=5, dash='dash'),
                name='教师: NaCl=0 (CaCl₂-H₂O)',
                hovertemplate='CaCl₂=%{x:.5f}  NaCl=0<br>TC_ref=%{z:.5f}'
                              '<extra>教师 NaCl=0</extra>',
            ))

        # ── ③ cacl2_zero 边界（CaCl₂=0，沿 NaCl 扫描，x=0 平面）──
        nacl_scan = np.linspace(*self.nacl_range, self.bn).astype(np.float32)
        X_b2 = np.column_stack([
            np.full(self.bn, T_val), np.full(self.bn, P_val),
            np.zeros(self.bn), nacl_scan,         # X[:,2]=CaCl₂=0, X[:,3]=NaCl
        ]).astype(np.float32)
        tc_dnn_b2 = wrapper.predict(X_b2).flatten()

        fig.add_trace(go.Scatter3d(
            x=np.zeros(self.bn), y=nacl_scan, z=tc_dnn_b2,
            mode='lines', line=dict(color='lime', width=7),
            name='DNN: CaCl₂=0 边界',
            hovertemplate='CaCl₂=0  NaCl=%{y:.5f}<br>TC=%{z:.5f}'
                          '<extra>DNN CaCl₂=0</extra>',
        ))
        if m_nacl is not None:
            # 教师模型取 [T, P, NaCl] = X_b2[:,[0,1,3]]
            tc_ref_b2 = m_nacl.predict(X_b2[:, [0, 1, 3]]).flatten()
            fig.add_trace(go.Scatter3d(
                x=np.zeros(self.bn), y=nacl_scan, z=tc_ref_b2,
                mode='lines', line=dict(color='orange', width=5, dash='dash'),
                name='教师: CaCl₂=0 (NaCl-H₂O)',
                hovertemplate='CaCl₂=0  NaCl=%{y:.5f}<br>TC_ref=%{z:.5f}'
                              '<extra>教师 CaCl₂=0</extra>',
            ))

        # ── ④ 真实数据散点 ───────────────────────────────────────────
        cmap = {'train': '#1f77b4', 'near': '#ff7f0e', 'far': '#d62728'}
        smap = {'train': 'circle',  'near': 'square',  'far': 'diamond'}
        lmap = {'train': '训练集',   'near': '近外推',   'far': '远外推'}
        t_tol = max(T_val * T_TOL_REL, 5.0)
        p_tol = P_val * P_TOL_REL

        for sp in ('train', 'near', 'far'):
            idx  = [i for i, s in enumerate(splits) if s == sp]
            if not idx: continue
            Xs, ys = all_X[idx], all_y[idx]
            mask = (np.abs(Xs[:,0]-T_val) < t_tol) & (np.abs(Xs[:,1]-P_val) < p_tol)
            if mask.sum() == 0: continue
            yp = wrapper.predict(Xs[mask]).flatten()
            fig.add_trace(go.Scatter3d(
                x=Xs[mask, 2], y=Xs[mask, 3], z=ys[mask],
                mode='markers',
                marker=dict(size=6, color=cmap[sp], symbol=smap[sp],
                            line=dict(width=0.8, color='white')),
                name=f'{lmap[sp]}',
                customdata=np.column_stack([Xs[mask,0], Xs[mask,1], yp, ys[mask]-yp]),
                hovertemplate=('CaCl₂=%{x:.5f}  NaCl=%{y:.5f}<br>TC_true=%{z:.5f}<br>'
                               'T=%{customdata[0]:.1f}K  P=%{customdata[1]:.2e}Pa<br>'
                               'TC_pred=%{customdata[2]:.5f}  res=%{customdata[3]:+.5f}'
                               f'<extra>{lmap[sp]}</extra>'),
            ))

        # ── 局部 R² + 边界 R² ────────────────────────────────────────
        all_mask = (np.abs(all_X[:,0]-T_val)<t_tol) & (np.abs(all_X[:,1]-P_val)<p_tol)
        r2_loc = float('nan')
        if all_mask.sum() >= 2:
            r2_loc = float(r2_score(all_y[all_mask],
                                    wrapper.predict(all_X[all_mask]).flatten()))

        def _r2_boundary(dnn_pred, ref_pred):
            try: return float(r2_score(ref_pred, dnn_pred))
            except: return float('nan')

        r2_b1 = r2_b2 = float('nan')
        if m_cacl2 is not None:
            r2_b1 = _r2_boundary(tc_dnn_b1, m_cacl2.predict(X_b1[:,[0,1,2]]).flatten())
        if m_nacl is not None:
            r2_b2 = _r2_boundary(tc_dnn_b2, m_nacl.predict(X_b2[:,[0,1,3]]).flatten())

        fig.update_layout(
            title=dict(
                text=(f'Fold {self.fold_idx} — DNN 基线  '
                      f'T={T_val:.1f}K  P={P_val:.2e}Pa<br>'
                      f'<sup>局部R²={r2_loc:.4f} | '
                      f'NaCl=0边界R²={r2_b1:.3f} | '
                      f'CaCl₂=0边界R²={r2_b2:.3f} | '
                      f'实线=DNN  虚线=教师</sup>'),
                x=0.5),
            scene=dict(
                xaxis=dict(title='CaCl₂ (mole frac)', range=[0, self.cacl2_range[1]]),
                yaxis=dict(title='NaCl (mole frac)',  range=[0, self.nacl_range[1]]),
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
class BaselineConfig:
    data_dir: Path = PROJECT_ROOT / 'data' / 'Thermal_conductivity' / 'split_by_temperature'
    train_data_file:  str = 'interpolation domain.xlsx'
    near_extrap_file: str = 'near-range extrapolation.xlsx'
    far_extrap_file:  str = 'far-range extrapolation.xlsx'

    models_dir: Path = PROJECT_ROOT / 'models' / 'Low_dim_model' / 'Thermal_conductivity'
    cacl2_h2o_model_file: str = 'CaCl2-H2O.pth'
    nacl_h2o_model_file:  str = 'NaCl-H2O.pth'

    dnn_epochs:        int   = 1000
    dnn_learning_rate: float = 0.00831
    dnn_layer_dim:     int   = 4
    dnn_node_dim:      int   = 128

    t_min: float = 290.0;  t_max: float = 570.0
    p_min: float = 5e6;    p_max: float = 5e7

    # ★ 实测盐组分摩尔分数范围（三元数据中的真实值）
    cacl2_range: Tuple[float, float] = (0.0017, 0.0127)
    nacl_range:  Tuple[float, float] = (0.0063, 0.0482)

    k_folds:            int = 5
    kfold_random_state: int = 42

    output_dir: Path = PROJECT_ROOT / 'results' / 'tc' / 'ablation' / 'baseline_results'
    save_predictions: bool = True
    save_metrics:     bool = True
    excel_prefix:     str  = 'baseline_'

    viz_t_list:   List[float] = field(default_factory=lambda: VIZ_T_LIST)
    viz_p_list:   List[float] = field(default_factory=lambda: VIZ_P_LIST)
    generate_viz: bool = True

    device:    str = 'auto'
    log_level: int = logging.INFO

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_data_path  = self.data_dir   / self.train_data_file
        self.near_extrap_path = self.data_dir   / self.near_extrap_file
        self.far_extrap_path  = self.data_dir   / self.far_extrap_file
        self.cacl2_h2o_path   = self.models_dir / self.cacl2_h2o_model_file
        self.nacl_h2o_path    = self.models_dir / self.nacl_h2o_model_file


# ==============================================================================
# 单折实验
# ==============================================================================

class BaselineAblationStudy:
    def __init__(self, config, model_cacl2_h2o, model_nacl_h2o):
        self.config          = config
        self.model_cacl2_h2o = model_cacl2_h2o
        self.model_nacl_h2o  = model_nacl_h2o
        self.results: Dict[str, Any] = {}
        self.logger = get_logger(self.__class__.__name__, config.log_level)

    def run_experiment(self, X_train, y_train, X_val, y_val,
                       X_near, y_near, X_far, y_far):
        self.logger.info("=" * 70)
        self.logger.info("Baseline DNN — Thermal Conductivity (NaCl–CaCl₂–H₂O)")
        self.logger.info(f"  CaCl₂ 范围: {self.config.cacl2_range}  "
                         f"NaCl 范围: {self.config.nacl_range}")
        self.logger.info("=" * 70)

        # ── Step 1: TSTREvaluator ──────────────────────────────────────
        self.logger.info("[Step 1] TSTREvaluator — 训练 DNN")
        ev = TSTREvaluator(
            X_val=X_val, y_val=y_val,
            X_test=X_near, y_test=y_near,
            X_train=X_train, y_train=y_train,
            config=PhysicsConfig(
                tstr_epochs=self.config.dnn_epochs,
                tstr_lr=self.config.dnn_learning_rate,
                dnn_layer_dim=self.config.dnn_layer_dim,
                dnn_node_dim=self.config.dnn_node_dim,
                tstr_device=self.config.device,
            ),
        )
        result  = ev.evaluate(X_syn=X_train, y_syn=y_train,
                               epochs=self.config.dnn_epochs, verbose=True)
        md      = result['metrics']
        model   = result['model']
        xs, ys  = result['x_scaler'], result['y_scaler']
        preds   = result['predictions']

        for stat in ('r2', 'rmse', 'mae'):
            md[f'near_{stat}'] = md.pop(f'test_{stat}', float('nan'))
        preds['near'] = preds.pop('test', np.array([]))

        device = torch.device(self.config.device);  model.eval()

        def _pred(X):
            X_sc = xs.transform(np.asarray(X, dtype=np.float32))
            with torch.no_grad():
                out = model(torch.FloatTensor(X_sc).to(device)).cpu().numpy()
            return ys.inverse_transform(out).flatten()

        for tag, Xq, yq in [('far', X_far, y_far), ('train', X_train, y_train)]:
            yp = _pred(Xq)
            md[f'{tag}_r2']   = float(r2_score(yq, yp))
            md[f'{tag}_rmse'] = float(np.sqrt(mean_squared_error(yq, yp)))
            md[f'{tag}_mae']  = float(mean_absolute_error(yq, yp))
            preds[tag] = yp

        true_v = {'train': y_train, 'val': y_val, 'near': y_near, 'far': y_far}
        full   = {'metrics': md, 'predictions': preds, 'true_values': true_v}
        self.results.update({'metrics': full, 'model': model,
                              'x_scaler': xs, 'y_scaler': ys})

        self.logger.info(
            f"  train={md['train_r2']:.4f}  val={md['val_r2']:.4f}  "
            f"near={md['near_r2']:.4f}  far={md['far_r2']:.4f}"
        )

        # ── Step 2: 物理评估（含调试打印）──────────────────────────────
        wrapper = _DNNWrapper(model, xs, ys, device)
        if all(m is not None for m in (self.model_cacl2_h2o, self.model_nacl_h2o)):
            self.logger.info("[Step 2] 物理一致性评估")
            self._run_physical_evaluation(wrapper)
        else:
            self.logger.warning("[Step 2] 边界模型缺失 — 跳过")

        # ── Step 3: 可视化 ─────────────────────────────────────────────
        if self.config.generate_viz:
            self.logger.info("[Step 3] 生成 3D 可视化（CaCl₂×NaCl 轴）")
            viz_dir = self.config.output_dir / 'visualizations'
            viz = BaselineVisualizer(
                viz_dir=viz_dir,
                fold_idx=getattr(self, '_fold_idx', 0),
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

        # ── Step 4: 保存 ──────────────────────────────────────────────
        self.logger.info("[Step 4] 保存结果")
        self._save(full)

    def _run_physical_evaluation(self, wrapper):
        try:
            # ── 调试：打印边界点的组成范围 ──────────────────────────────
            self.logger.info(
                f"  边界评估组成范围: CaCl₂={self.config.cacl2_range}  "
                f"NaCl={self.config.nacl_range}"
            )
            self.logger.info(
                f"  教师模型 CaCl₂-H₂O 取 X[:,[0,1,2]]=[T,P,CaCl₂]  "
                f"NaCl-H₂O 取 X[:,[0,1,3]]=[T,P,NaCl]"
            )

            ev = ThermalConductivityPhysicsEvaluator(
                teacher_models=(self.model_cacl2_h2o, self.model_nacl_h2o),
                temp_range=(self.config.t_min, self.config.t_max),
                pressure_range=(self.config.p_min, self.config.p_max),
                cacl2_range=self.config.cacl2_range,   # ★ 实测 CaCl₂ 范围
                nacl_range=self.config.nacl_range,     # ★ 实测 NaCl  范围
            )

            # ── 调试：抽样对比教师模型与 DNN 在边界上的输出 ────────────
            T_mid = (self.config.t_min + self.config.t_max) / 2
            P_mid = (self.config.p_min + self.config.p_max) / 2
            n_dbg = 5

            self.logger.info("  ── 边界调试采样（各取5点）──")
            # nacl_zero: X=[T,P,CaCl₂,0]
            cacl2_dbg = np.linspace(*self.config.cacl2_range, n_dbg).astype(np.float32)
            X_dbg1 = np.column_stack([np.full(n_dbg,T_mid), np.full(n_dbg,P_mid),
                                       cacl2_dbg, np.zeros(n_dbg)]).astype(np.float32)
            tc_dnn1  = wrapper.predict(X_dbg1).flatten()
            tc_ref1  = self.model_cacl2_h2o.predict(X_dbg1[:,[0,1,2]]).flatten()
            self.logger.info(f"  nacl_zero (NaCl=0, T={T_mid:.0f}K, P={P_mid:.0e}Pa):")
            for i in range(n_dbg):
                self.logger.info(
                    f"    CaCl₂={cacl2_dbg[i]:.5f}  "
                    f"DNN={tc_dnn1[i]:.5f}  teacher={tc_ref1[i]:.5f}  "
                    f"diff={tc_dnn1[i]-tc_ref1[i]:+.5f}"
                )

            # cacl2_zero: X=[T,P,0,NaCl]
            nacl_dbg = np.linspace(*self.config.nacl_range, n_dbg).astype(np.float32)
            X_dbg2 = np.column_stack([np.full(n_dbg,T_mid), np.full(n_dbg,P_mid),
                                       np.zeros(n_dbg), nacl_dbg]).astype(np.float32)
            tc_dnn2 = wrapper.predict(X_dbg2).flatten()
            tc_ref2 = self.model_nacl_h2o.predict(X_dbg2[:,[0,1,3]]).flatten()
            self.logger.info(f"  cacl2_zero (CaCl₂=0, T={T_mid:.0f}K, P={P_mid:.0e}Pa):")
            for i in range(n_dbg):
                self.logger.info(
                    f"    NaCl={nacl_dbg[i]:.5f}  "
                    f"DNN={tc_dnn2[i]:.5f}  teacher={tc_ref2[i]:.5f}  "
                    f"diff={tc_dnn2[i]-tc_ref2[i]:+.5f}"
                )

            overall, res = ev.evaluate_full(wrapper)
            self.logger.info(
                f"  physics={overall:.4f}  "
                f"boundary={res.get('boundary_score', float('nan')):.4f}  "
                f"smoothness={res.get('smoothness_score', float('nan')):.4f}"
            )
            report = ev.generate_evaluation_report(res)
            rd = self.config.output_dir / 'data'
            rd.mkdir(parents=True, exist_ok=True)
            (rd / 'physical_evaluation_report.txt').write_text(report, encoding='utf-8')
            self.results['physical_evaluation'] = res
            self.results['physics_score'] = overall
        except Exception as e:
            import traceback
            self.logger.error(f"物理评估失败: {e}\n{traceback.format_exc()}")

    def _save(self, result):
        ed = self.config.output_dir / 'excel';  ed.mkdir(parents=True, exist_ok=True)
        md = self.config.output_dir / 'model';  md.mkdir(exist_ok=True)
        torch.save({'model_state_dict': self.results['model'].state_dict(),
                    'x_scaler': self.results['x_scaler'],
                    'y_scaler': self.results['y_scaler'],
                    'layer_dim': self.config.dnn_layer_dim,
                    'node_dim':  self.config.dnn_node_dim}, md / 'dnn.pth')

        if self.config.save_predictions:
            for sp in ('train', 'val', 'near', 'far'):
                yt = result['true_values'][sp].flatten()
                yp = result['predictions'][sp].flatten()
                pd.DataFrame({'y_true': yt, 'y_pred': yp, 'residual': yt-yp})\
                  .to_excel(ed / f"{self.config.excel_prefix}{sp}_predictions.xlsx",
                            index=False, engine='openpyxl')

        if self.config.save_metrics:
            inner = result['metrics']
            pe    = self.results.get('physical_evaluation')
            ps    = self.results.get('physics_score', float('nan'))
            bs    = pe.get('boundary_score',   float('nan')) if pe else float('nan')
            ss    = pe.get('smoothness_score', float('nan')) if pe else float('nan')
            rows  = [
                ['Train R²', inner.get('train_r2', float('nan'))],
                ['Train RMSE', inner.get('train_rmse', float('nan'))],
                ['Train MAE',  inner.get('train_mae',  float('nan'))],
                ['Val R²',   inner.get('val_r2',   float('nan'))],
                ['Val RMSE', inner.get('val_rmse', float('nan'))],
                ['Val MAE',  inner.get('val_mae',  float('nan'))],
                ['Near R²',  inner.get('near_r2',  float('nan'))],
                ['Near RMSE',inner.get('near_rmse',float('nan'))],
                ['Near MAE', inner.get('near_mae', float('nan'))],
                ['Far R²',   inner.get('far_r2',   float('nan'))],
                ['Far RMSE', inner.get('far_rmse', float('nan'))],
                ['Far MAE',  inner.get('far_mae',  float('nan'))],
                ['Physics Score', ps],
                ['Boundary Consistency', bs],
                ['Thermodynamic Smoothness', ss],
            ]
            pd.DataFrame(rows, columns=['Metric', 'Value'])\
              .to_excel(ed / f'{self.config.excel_prefix}metrics.xlsx',
                        index=False, engine='openpyxl')


# ==============================================================================
# K-Fold 管理器
# ==============================================================================

class KFoldExperimentManager:
    def __init__(self, base_config: BaselineConfig, n_folds: int = 5):
        if n_folds < 2: raise ValueError("n_folds >= 2")
        self.base_config     = base_config
        self.n_folds         = n_folds
        self.all_results:    List[Dict[str, Any]] = []
        self.logger          = get_logger(self.__class__.__name__, base_config.log_level)
        self.main_output_dir = base_config.output_dir
        self.main_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cacl2_h2o = None
        self.model_nacl_h2o  = None
        self._load_boundary_models()

    def _load_boundary_models(self):
        try:
            dev   = torch.device(self.base_config.device)
            paths = {'cacl2_h2o': self.base_config.cacl2_h2o_path,
                     'nacl_h2o':  self.base_config.nacl_h2o_path}
            miss  = [k for k, p in paths.items() if not p.exists()]
            if miss:
                self.logger.warning(f"边界模型缺失: {miss}"); return
            self.model_cacl2_h2o = move_to_device(
                LowDimEnsemble.load(str(paths['cacl2_h2o'])), dev)
            self.model_nacl_h2o  = move_to_device(
                LowDimEnsemble.load(str(paths['nacl_h2o'])),  dev)
            # 打印归一化参数确认
            for name, m in [('CaCl2-H2O', self.model_cacl2_h2o),
                             ('NaCl-H2O',  self.model_nacl_h2o)]:
                if m.is_scaler_fitted:
                    xm = m.x_mean.cpu().numpy().flatten()
                    self.logger.info(f"  {name} x_mean={xm} "
                                     f"(T均值, P均值, 第三组分均值≈{xm[2]:.5f})")
            self.logger.info("两个边界模型已加载  "
                             "CaCl₂-H₂O→nacl_zero  NaCl-H₂O→cacl2_zero")
        except Exception as e:
            self.logger.error(f"加载边界模型失败: {e}")

    def run_all_folds(self):
        self.logger.info("█" * 70)
        self.logger.info("█" + " TC Baseline K-Fold (NaCl–CaCl₂–H₂O) ".center(68) + "█")
        self.logger.info("█" * 70)
        cfg = self.base_config
        self.logger.info(f"CaCl₂范围={cfg.cacl2_range}  NaCl范围={cfg.nacl_range}")

        ts = time.time()
        X_tp, y_tp = load_tc_data(cfg.train_data_path)
        X_nr, y_nr = load_tc_data(cfg.near_extrap_path)
        X_fr, y_fr = load_tc_data(cfg.far_extrap_path)
        self.logger.info(f"训练池:{len(X_tp)}  近域:{len(X_nr)}  远域:{len(X_fr)}")
        self.logger.info(
            f"数据 CaCl₂∈[{X_tp[:,2].min():.5f},{X_tp[:,2].max():.5f}]  "
            f"NaCl∈[{X_tp[:,3].min():.5f},{X_tp[:,3].max():.5f}]"
        )
        self._near_data = (X_nr, y_nr)
        self._far_data  = (X_fr, y_fr)

        folds = create_k_folds(X_tp, y_tp, self.n_folds, cfg.kfold_random_state)

        for fd in folds:
            fi = fd['fold_idx']
            fs = time.time()
            self.logger.info(f"\n{'='*70}\nFold {fi+1}/{self.n_folds}\n{'='*70}")

            fold_dir = self.main_output_dir / f'fold_{fi}'
            for sub in ('excel', 'data', 'visualizations'):
                (fold_dir / sub).mkdir(parents=True, exist_ok=True)

            fc = BaselineConfig()
            fc.__dict__.update(cfg.__dict__)
            fc.output_dir = fold_dir

            study = BaselineAblationStudy(fc, self.model_cacl2_h2o, self.model_nacl_h2o)
            study._fold_idx = fi
            study.run_experiment(fd['X_train'], fd['y_train'],
                                  fd['X_val'],   fd['y_val'],
                                  X_nr, y_nr, X_fr, y_fr)

            fr: Dict[str, Any] = {
                'fold_idx': fi, 'metrics': study.results['metrics'],
                'physics_score': None, 'physics_boundary': None, 'physics_smoothness': None,
            }
            if 'physical_evaluation' in study.results:
                pe = study.results['physical_evaluation']
                fr['physics_score']      = pe.get('overall_score',    None)
                fr['physics_boundary']   = pe.get('boundary_score',   None)
                fr['physics_smoothness'] = pe.get('smoothness_score', None)

            self.all_results.append(fr)
            self.logger.info(f"Fold {fi} 完成 {timedelta(seconds=int(time.time()-fs))}")

        self.logger.info(
            f"\n总耗时 {timedelta(seconds=int(time.time()-ts))}  "
            f"结果: {self.main_output_dir}")
        self._summary()
        self._best_eval()

    def _summary(self):
        sd = self.main_output_dir / 'summary';  sd.mkdir(exist_ok=True)
        keys  = ['train_r2','train_rmse','train_mae','val_r2','val_rmse','val_mae',
                 'near_r2','near_rmse','near_mae','far_r2','far_rmse','far_mae']
        names = ['Train R²','Train RMSE','Train MAE','Val R²','Val RMSE','Val MAE',
                 'Near R²','Near RMSE','Near MAE','Far R²','Far RMSE','Far MAE']
        rows = []
        for k, n in zip(keys, names):
            vs = [r['metrics']['metrics'][k] for r in self.all_results]
            rows.append({'Metric': n, 'Mean Value': float(np.mean(vs)),
                         'Std': float(np.std(vs, ddof=1)) if len(vs)>1 else 0.0})
        for attr, label in [('physics_score','Physics Score'),
                             ('physics_boundary','Boundary Consistency'),
                             ('physics_smoothness','Thermodynamic Smoothness')]:
            vs = [r.get(attr) for r in self.all_results
                  if r.get(attr) is not None
                  and not np.isnan(float(r.get(attr, float('nan'))))]
            rows.append({'Metric': label,
                         'Mean Value': float(np.mean(vs)) if vs else float('nan'),
                         'Std': float(np.std(vs, ddof=1)) if len(vs)>1 else 0.0})
        pd.DataFrame(rows).to_excel(sd/'summary_metrics.xlsx', index=False, engine='openpyxl')

        # 文字报告
        sep  = '-'*70
        cfg  = self.base_config
        lines = ['='*70, 'TC Baseline K-Fold Summary', '='*70,
                 f"CaCl₂范围={cfg.cacl2_range}  NaCl范围={cfg.nacl_range}", '', sep]
        for k, n in zip(keys[:4], names[:4]):
            vs = [r['metrics']['metrics'][k] for r in self.all_results]
            lines.append(f"{n:<12}: {np.mean(vs):.6f} ± {np.std(vs,ddof=1):.6f}")
        (sd/'summary_report.txt').write_text('\n'.join(lines), encoding='utf-8')
        self.logger.info('\n' + '\n'.join(lines))

    def _best_eval(self):
        val_r2 = [r['metrics']['metrics']['val_r2'] for r in self.all_results]
        fi  = self.all_results[int(np.argmax(val_r2))]['fold_idx']
        mp  = self.main_output_dir / f'fold_{fi}' / 'model' / 'dnn.pth'
        if not mp.exists(): return
        dev  = torch.device(self.base_config.device)
        ckpt = torch.load(str(mp), map_location=dev)
        dnn  = DNN(input_dim=4, layer_dim=ckpt['layer_dim'],
                   node_dim=ckpt['node_dim']).to(dev)
        dnn.load_state_dict(ckpt['model_state_dict']); dnn.eval()
        w   = _DNNWrapper(dnn, ckpt['x_scaler'], ckpt['y_scaler'], dev)
        bd  = self.main_output_dir / 'best_model';  bd.mkdir(exist_ok=True)
        rows = [['Best Fold', fi], ['Val R²', val_r2[int(np.argmax(val_r2))]]]
        for tag, X, ya in [('near', *self._near_data), ('far', *self._far_data)]:
            yp = w.predict(X).flatten(); yt = ya.flatten()
            r2 = float(r2_score(yt, yp))
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            mae  = float(mean_absolute_error(yt, yp))
            label = 'Near' if tag == 'near' else 'Far'
            self.logger.info(f"  best {label}: r²={r2:.4f} rmse={rmse:.5f} mae={mae:.5f}")
            rows += [[f'{label} R²',r2],[f'{label} RMSE',rmse],[f'{label} MAE',mae]]
            pd.DataFrame({'y_true':yt,'y_pred':yp,'residual':yt-yp})\
              .to_excel(bd/f'best_{tag}_predictions.xlsx', index=False, engine='openpyxl')
        pd.DataFrame(rows, columns=['Metric','Value'])\
          .to_excel(bd/'best_model_metrics.xlsx', index=False, engine='openpyxl')


# ==============================================================================
# Main
# ==============================================================================

def main():
    config = BaselineConfig()
    config.k_folds            = 5
    config.kfold_random_state = 2
    config.dnn_epochs         = 1000
    config.dnn_learning_rate  = 0.00831
    config.dnn_layer_dim      = 4
    config.dnn_node_dim       = 128
    config.t_min              = 290.0
    config.t_max              = 570.0
    config.p_min              = 5e6
    config.p_max              = 5e7

    # ★ 关键：使用三元数据实测的盐组分范围，而非旧的 H₂O 范围
    config.cacl2_range        = (0.0017, 0.0127)   # ★ 实测 CaCl₂ 摩尔分数范围
    config.nacl_range         = (0.0063, 0.0482)   # ★ 实测 NaCl  摩尔分数范围

    config.save_predictions   = True
    config.save_metrics       = True
    config.generate_viz       = True
    config.log_level          = logging.INFO
    config.viz_t_list = [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15]
    config.viz_p_list = [5e6, 1e7, 2e7, 3e7, 5e7]

    logger.info("Baseline Ablation — Thermal Conductivity (NaCl–CaCl₂–H₂O)")
    KFoldExperimentManager(config, n_folds=config.k_folds).run_all_folds()
    logger.info("Done.")


if __name__ == '__main__':
    main()