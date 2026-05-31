"""
================================================================================
GPR Baseline - K-Fold Cross-Validation（热导率 NaCl–CaCl₂–H₂O 体系）
================================================================================

与 pc_cvae_experiment_tc.py 完全对齐：
  - 相同分层 K-Fold（按 T 分位数，StratifiedKFold）
  - 相同输出格式：fold metrics / predictions / summary
  - 相同物理评估接口：ThermalConductivityPhysicsEvaluator 双支柱框架
  - 相同目录结构：fold_{i}/ + summary/ + best_model/

与粘度版 GPR 的差异：
  1. 数据列：[T, P, CaCl₂, NaCl, TC]（4 输入特征）
  2. 拟合目标：TC 原始值（不做对数变换，TC 无需对数正态假设）
  3. 边界模型：2 个（CaCl₂-H₂O / NaCl-H₂O），替换粘度版的 3 个
  4. 物理评估器：ThermalConductivityPhysicsEvaluator
  5. T/P 范围：290–570 K，5e6–5e7 Pa
  6. 分层 K-Fold：按 T 分位数，消除折间高温点分配不均

GPR 专有适配：
  - _GPRWrapper：将 (gpr, scaler) 适配为 trainer.predict() 接口
  - 模型用 joblib pickle（gpr.pkl）存储，供 best_model 评估加载
================================================================================
"""

import logging
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / 'models'))

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
from low_dim_model import LowDimEnsemble
from utils_thermal_conductivity import ThermalConductivityPhysicsEvaluator

warnings.filterwarnings('ignore')


# ==============================================================================
# 工具函数
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


def load_tc_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """加载热导率数据文件。

    列顺序约定：[T, P, CaCl₂, NaCl, TC]
    X: (N, 4)  y: (N,)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    data = pd.read_excel(filepath, engine='openpyxl')
    if data.shape[1] < 5:
        raise ValueError(f"期望至少 5 列 [T, P, CaCl₂, NaCl, TC]，实际: {data.shape[1]}")
    X = data.iloc[:, :4].values.astype(np.float64)
    y = data.iloc[:,  4].values.astype(np.float64)
    return X, y


def move_to_device(model: LowDimEnsemble, device) -> LowDimEnsemble:
    model.to(device)
    model.device = device
    return model


def create_k_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """按 T 分位数分层的 K-Fold，保证每折训练集覆盖全温度范围。"""
    T_vals = X[:, 0]
    T_bins = pd.qcut(T_vals, q=n_splits, labels=False, duplicates='drop')
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [
        {
            'fold_idx': fi,
            'X_train': X[tr], 'y_train': y[tr],
            'X_val':   X[va], 'y_val':   y[va],
        }
        for fi, (tr, va) in enumerate(skf.split(X, T_bins))
    ]


# ==============================================================================
# GPR 模型构建与预测
# ==============================================================================

def build_gpr(n_restarts_optimizer: int = 10, random_state: int = 42) -> GaussianProcessRegressor:
    """
    核函数：ConstantKernel × Matern(nu=2.5) + WhiteKernel
      - length_scale 4 维，对应 [T, P, CaCl₂, NaCl]
      - Matern(nu=2.5)：TC-T 曲线一阶导数连续
      - WhiteKernel：吸收测量噪声
    """
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=[1.0, 1.0, 1.0, 1.0],
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=True,
        random_state=random_state,
    )


def train_gpr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_restarts_optimizer: int = 10,
    random_state: int = 42,
) -> Tuple[GaussianProcessRegressor, StandardScaler]:
    """特征标准化 + 原始尺度 GPR 拟合（TC 无需对数变换）。"""
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    gpr      = build_gpr(n_restarts_optimizer, random_state)
    gpr.fit(X_scaled, y_train)
    return gpr, scaler


def predict_gpr(
    gpr:    GaussianProcessRegressor,
    scaler: StandardScaler,
    X:      np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 TC 预测值 (N,) 及标准差 (N,)，均为原始尺度（W/m·K）。"""
    X_scaled    = scaler.transform(X)
    pred, std   = gpr.predict(X_scaled, return_std=True)
    return pred, std


# ==============================================================================
# ThermalConductivityPhysicsEvaluator 适配包装器
# ==============================================================================

class _GPRWrapper:
    """将 (gpr, scaler) 适配为 trainer.predict(X, return_original_scale) 接口。"""

    def __init__(self, gpr: GaussianProcessRegressor, scaler: StandardScaler) -> None:
        self._gpr    = gpr
        self._scaler = scaler

    def predict(self, X: np.ndarray, return_original_scale: bool = True) -> np.ndarray:
        """(N, 4) [T, P, CaCl₂, NaCl] → (N, 1) TC（W/m·K）。"""
        y_pred, _ = predict_gpr(self._gpr, self._scaler, X)
        return y_pred.reshape(-1, 1)


# ==============================================================================
# 配置
# ==============================================================================

@dataclass
class GPRExperimentConfig:
    """GPR 热导率基线 K-Fold 实验配置（NaCl–CaCl₂–H₂O 体系）。"""

    # 数据路径
    data_dir: Path = PROJECT_ROOT / 'data' / 'Thermal_conductivity' / 'split_by_temperature'
    train_data_file:  str = 'interpolation domain.xlsx'
    near_extrap_file: str = 'near-range extrapolation.xlsx'
    far_extrap_file:  str = 'far-range extrapolation.xlsx'

    # 边界模型路径（两个二元体系）
    models_dir: Path = PROJECT_ROOT / 'models' / 'Low_dim_model' / 'Thermal_conductivity'
    cacl2_h2o_model_file: str = 'CaCl2-H2O.pth'   # nacl_zero 边界
    nacl_h2o_model_file:  str = 'NaCl-H2O.pth'    # cacl2_zero 边界

    # GPR 超参数
    n_restarts_optimizer: int = 10
    gpr_random_state:     int = 42

    # 物理评估范围
    t_min: float = 290.0
    t_max: float = 570.0
    p_min: float = 5e6
    p_max: float = 5e7
    # 实测盐组分摩尔分数范围
    cacl2_range: Tuple[float, float] = (0.0017, 0.0127)
    nacl_range:  Tuple[float, float] = (0.0063, 0.0482)

    # 交叉验证
    k_folds: int = 5
    kfold_random_state: int = 42

    # 输出
    output_dir: Path = PROJECT_ROOT / 'results' / 'tc' / 'ablation' / 'GPR_results'
    save_predictions: bool = True
    save_metrics:     bool = True
    excel_prefix:     str  = 'gpr_'

    log_level: int = logging.INFO

    def __post_init__(self) -> None:
        self.train_data_path  = self.data_dir   / self.train_data_file
        self.near_extrap_path = self.data_dir   / self.near_extrap_file
        self.far_extrap_path  = self.data_dir   / self.far_extrap_file
        self.cacl2_h2o_path   = self.models_dir / self.cacl2_h2o_model_file
        self.nacl_h2o_path    = self.models_dir / self.nacl_h2o_model_file


# ==============================================================================
# 单折执行器
# ==============================================================================

class SingleFoldRunner:

    def __init__(
        self,
        config:          GPRExperimentConfig,
        model_cacl2_h2o: Optional[LowDimEnsemble],
        model_nacl_h2o:  Optional[LowDimEnsemble],
    ) -> None:
        self.config          = config
        self.logger          = get_logger(self.__class__.__name__, config.log_level)
        self.model_cacl2_h2o = model_cacl2_h2o
        self.model_nacl_h2o  = model_nacl_h2o

    def run(
        self,
        fold_idx: int,
        X_train:  np.ndarray, y_train: np.ndarray,
        X_val:    np.ndarray, y_val:   np.ndarray,
        X_near:   np.ndarray, y_near:  np.ndarray,
        X_far:    np.ndarray, y_far:   np.ndarray,
        fold_dir: Path,
    ) -> Dict[str, Any]:
        self.logger.info("=" * 70)
        self.logger.info(f"Fold {fold_idx}")
        self.logger.info("=" * 70)

        self.logger.info("[Step 1] 训练 GPR（原始尺度，StandardScaler 特征标准化）")
        gpr, scaler = self._train_gpr(X_train, y_train)

        self.logger.info("[Step 2] 评估 train/val/near/far")
        metrics = self._evaluate_gpr(
            gpr, scaler,
            X_train, y_train, X_val, y_val, X_near, y_near, X_far, y_far,
        )

        self.logger.info("[Step 3] 物理一致性评估（双支柱框架）")
        physics_results = self._compute_physics_metrics(gpr, scaler)

        self._save_fold_results(fold_idx, fold_dir, metrics, physics_results, gpr, scaler)

        return {
            'fold_idx':           fold_idx,
            'metrics':            metrics,
            'physics_score':      physics_results.get('physics_score',      None),
            'physics_boundary':   physics_results.get('physics_boundary',   None),
            'physics_smoothness': physics_results.get('physics_smoothness', None),
        }

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    def _train_gpr(self, X_train, y_train):
        gpr, scaler = train_gpr(
            X_train, y_train,
            n_restarts_optimizer=self.config.n_restarts_optimizer,
            random_state=self.config.gpr_random_state,
        )
        self.logger.info(f"  优化后核函数: {gpr.kernel_}")
        self.logger.info(f"  对数边际似然: {gpr.log_marginal_likelihood_value_:.4f}")
        return gpr, scaler

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------

    def _evaluate_gpr(self, gpr, scaler,
                      X_train, y_train, X_val, y_val,
                      X_near, y_near, X_far, y_far):
        splits = {
            'train': (X_train, y_train),
            'val':   (X_val,   y_val),
            'near':  (X_near,  y_near),
            'far':   (X_far,   y_far),
        }
        preds: Dict[str, np.ndarray] = {}
        stds:  Dict[str, np.ndarray] = {}
        for s, (X, _) in splits.items():
            yp, ys   = predict_gpr(gpr, scaler, X)
            preds[s] = yp.flatten()
            stds[s]  = ys.flatten()

        trues  = {s: y.flatten() for s, (_, y) in splits.items()}
        inputs = {s: X          for s, (X, _) in splits.items()}

        met: Dict[str, float] = {}
        for s in ('train', 'val', 'near', 'far'):
            met[f'{s}_r2']   = float(r2_score(trues[s], preds[s]))
            met[f'{s}_rmse'] = float(np.sqrt(mean_squared_error(trues[s], preds[s])))
            met[f'{s}_mae']  = float(mean_absolute_error(trues[s], preds[s]))

        self.logger.info(
            f"  train={met['train_r2']:.4f}  val={met['val_r2']:.4f}  "
            f"near={met['near_r2']:.4f}  far={met['far_r2']:.4f}"
        )

        return {
            'metrics':     met,
            'predictions': preds,
            'true_values': trues,
            'inputs':      inputs,
            'stds':        stds,
        }

    # ------------------------------------------------------------------
    # Step 3
    # ------------------------------------------------------------------

    def _compute_physics_metrics(self, gpr, scaler) -> Dict[str, float]:
        if any(m is None for m in (self.model_cacl2_h2o, self.model_nacl_h2o)):
            self.logger.warning("边界模型缺失 — 跳过物理评估")
            return {}
        try:
            evaluator = ThermalConductivityPhysicsEvaluator(
                teacher_models=(self.model_cacl2_h2o, self.model_nacl_h2o),
                temp_range=(self.config.t_min, self.config.t_max),
                pressure_range=(self.config.p_min, self.config.p_max),
                cacl2_range=self.config.cacl2_range,
                nacl_range=self.config.nacl_range,
            )
            wrapper = _GPRWrapper(gpr, scaler)
            overall_score, results = evaluator.evaluate_full(wrapper)
            bs = results.get('boundary_score',   float('nan'))
            ss = results.get('smoothness_score', float('nan'))
            self.logger.info(
                f"  physics={overall_score:.4f}  boundary={bs:.4f}  smoothness={ss:.4f}"
            )
            return {
                'physics_score':      float(overall_score),
                'physics_boundary':   float(bs),
                'physics_smoothness': float(ss),
            }
        except Exception as e:
            import traceback
            self.logger.error(f"物理评估失败: {e}\n{traceback.format_exc()}")
            return {}

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------

    def _save_fold_results(self, fold_idx, fold_dir, metrics, physics_results, gpr, scaler):
        fold_dir.mkdir(parents=True, exist_ok=True)
        md = fold_dir / 'model';  md.mkdir(exist_ok=True)
        joblib.dump({'gpr': gpr, 'scaler': scaler}, md / 'gpr.pkl')

        if self.config.save_metrics:
            self._save_fold_metrics(metrics, physics_results, fold_dir)
        if self.config.save_predictions:
            self._save_predictions(metrics, fold_dir)

    def _save_fold_metrics(self, metrics, physics_results, fold_dir):
        inner = metrics.get('metrics', metrics)
        pe    = physics_results
        ps    = pe.get('physics_score',      float('nan')) if pe else float('nan')
        bs    = pe.get('physics_boundary',   float('nan')) if pe else float('nan')
        ss    = pe.get('physics_smoothness', float('nan')) if pe else float('nan')
        rows = [
            ['Train R²',                 inner.get('train_r2',   float('nan'))],
            ['Train RMSE',               inner.get('train_rmse', float('nan'))],
            ['Train MAE',                inner.get('train_mae',  float('nan'))],
            ['Val R²',                   inner.get('val_r2',     float('nan'))],
            ['Val RMSE',                 inner.get('val_rmse',   float('nan'))],
            ['Val MAE',                  inner.get('val_mae',    float('nan'))],
            ['Near-Range R²',            inner.get('near_r2',    float('nan'))],
            ['Near-Range RMSE',          inner.get('near_rmse',  float('nan'))],
            ['Near-Range MAE',           inner.get('near_mae',   float('nan'))],
            ['Far-Range R²',             inner.get('far_r2',     float('nan'))],
            ['Far-Range RMSE',           inner.get('far_rmse',   float('nan'))],
            ['Far-Range MAE',            inner.get('far_mae',    float('nan'))],
            ['Physics Score',            ps],
            ['Boundary Consistency',     bs],
            ['Thermodynamic Smoothness', ss],
        ]
        pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
            fold_dir / f'{self.config.excel_prefix}metrics.xlsx',
            index=False, engine='openpyxl',
        )

    def _save_predictions(self, metrics, fold_dir):
        inputs = metrics.get('inputs', {})
        for sp in ('train', 'val', 'near', 'far'):
            yt = metrics['true_values'][sp].flatten()
            yp = metrics['predictions'][sp].flatten()
            X  = inputs.get(sp)
            if X is not None:
                df = pd.DataFrame({
                    'T_K':         X[:, 0],
                    'P_Pa':        X[:, 1],
                    'CaCl2':       X[:, 2],
                    'NaCl':        X[:, 3],
                    'TC_true':     yt,
                    'TC_pred':     yp,
                    'residual':    yt - yp,
                    'abs_error':   np.abs(yt - yp),
                    'rel_error_%': np.abs(yt - yp) / np.maximum(np.abs(yt), 1e-10) * 100,
                })
            else:
                df = pd.DataFrame({'TC_true': yt, 'TC_pred': yp, 'residual': yt - yp})
            df.to_excel(
                fold_dir / f'{self.config.excel_prefix}{sp}_predictions.xlsx',
                index=False, engine='openpyxl',
            )


# ==============================================================================
# K-Fold 实验管理器
# ==============================================================================

class KFoldExperimentManager:

    def __init__(self, config: GPRExperimentConfig, n_folds: int = 5) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds 须 >= 2，实际: {n_folds}")
        self.config      = config
        self.n_folds     = n_folds
        self.logger      = get_logger(self.__class__.__name__, level=config.log_level)
        self.output_dir  = config.output_dir
        self.all_results: List[Dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_cacl2_h2o: Optional[LowDimEnsemble] = None
        self.model_nacl_h2o:  Optional[LowDimEnsemble] = None
        self._load_boundary_models()

    def _load_boundary_models(self) -> None:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            paths  = {
                'cacl2_h2o': self.config.cacl2_h2o_path,
                'nacl_h2o':  self.config.nacl_h2o_path,
            }
            missing = [k for k, p in paths.items() if not p.exists()]
            if missing:
                self.logger.warning(f"边界模型缺失: {missing} — 物理评估将禁用")
                return

            self.model_cacl2_h2o = move_to_device(
                LowDimEnsemble.load(str(paths['cacl2_h2o'])), device)
            self.model_nacl_h2o  = move_to_device(
                LowDimEnsemble.load(str(paths['nacl_h2o'])),  device)
            self.logger.info(f"两个边界模型已加载 → {device}")
        except Exception as e:
            self.logger.error(f"加载边界模型失败: {e}")

    def run_all_folds(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info("GPR 基线实验 — K-Fold CV（NaCl–CaCl₂–H₂O 热导率）")
        self.logger.info("=" * 70)
        self.logger.info(
            f"K-Folds: {self.n_folds}  GPR: Matern(nu=2.5) + WhiteKernel  "
            f"n_restarts={self.config.n_restarts_optimizer}  原始尺度拟合（无对数变换）"
        )
        self.logger.info(
            f"T=[{self.config.t_min},{self.config.t_max}]K  "
            f"P=[{self.config.p_min:.0e},{self.config.p_max:.0e}]Pa  "
            f"CaCl₂={self.config.cacl2_range}  NaCl={self.config.nacl_range}"
        )

        start = time.time()

        X_tp, y_tp = load_tc_data(self.config.train_data_path)
        X_nr, y_nr = load_tc_data(self.config.near_extrap_path)
        X_fr, y_fr = load_tc_data(self.config.far_extrap_path)
        self.logger.info(f"训练池: {len(X_tp)}  近域: {len(X_nr)}  远域: {len(X_fr)}")
        self._near_data = (X_nr, y_nr)
        self._far_data  = (X_fr, y_fr)

        folds = create_k_folds(
            X_tp, y_tp,
            n_splits=self.n_folds,
            random_state=self.config.kfold_random_state,
        )
        for f in folds:
            self.logger.info(
                f"  Fold {f['fold_idx']}: train={len(f['X_train'])}  val={len(f['X_val'])}"
            )

        for fold_data in folds:
            fi = fold_data['fold_idx']
            self.logger.info(f"\n{'█'*70}\nFold {fi+1}/{self.n_folds}\n{'█'*70}")
            fold_dir = self.output_dir / f'fold_{fi}'
            fold_dir.mkdir(exist_ok=True)

            runner = SingleFoldRunner(
                config=self.config,
                model_cacl2_h2o=self.model_cacl2_h2o,
                model_nacl_h2o=self.model_nacl_h2o,
            )
            result = runner.run(
                fold_idx=fi,
                X_train=fold_data['X_train'], y_train=fold_data['y_train'],
                X_val=fold_data['X_val'],     y_val=fold_data['y_val'],
                X_near=X_nr, y_near=y_nr,
                X_far=X_fr,  y_far=y_fr,
                fold_dir=fold_dir,
            )
            self.all_results.append(result)

        self._generate_summary()
        self._best_model_evaluation()
        self.logger.info(
            f"\n总耗时 {timedelta(seconds=int(time.time()-start))}"
            f"  结果: {self.output_dir}"
        )

    def _generate_summary(self) -> None:
        sd = self.output_dir / 'summary';  sd.mkdir(exist_ok=True)
        self._save_summary_metrics(sd)
        self._save_summary_predictions(sd)
        self._generate_text_report(sd)

    def _save_summary_metrics(self, sd: Path) -> None:
        STAT_KEYS  = ['train_r2','train_rmse','train_mae',
                      'val_r2',  'val_rmse',  'val_mae',
                      'near_r2', 'near_rmse', 'near_mae',
                      'far_r2',  'far_rmse',  'far_mae']
        STAT_NAMES = ['Train R²','Train RMSE','Train MAE',
                      'Val R²',  'Val RMSE',  'Val MAE',
                      'Near-Range R²','Near-Range RMSE','Near-Range MAE',
                      'Far-Range R²', 'Far-Range RMSE', 'Far-Range MAE']
        PHYS_ATTRS = [('physics_score',      'Physics Score'),
                      ('physics_boundary',   'Boundary Consistency'),
                      ('physics_smoothness', 'Thermodynamic Smoothness')]

        rows = []
        for k, n in zip(STAT_KEYS, STAT_NAMES):
            vs = [r['metrics']['metrics'][k] for r in self.all_results]
            rows.append({'Metric': n, 'Mean Value': float(np.mean(vs)),
                         'Std': float(np.std(vs, ddof=1)) if len(vs)>1 else 0.0})
        for attr, label in PHYS_ATTRS:
            vs = [r.get(attr) for r in self.all_results
                  if r.get(attr) is not None
                  and not np.isnan(float(r.get(attr, float('nan'))))]
            rows.append({'Metric': label,
                         'Mean Value': float(np.mean(vs)) if vs else float('nan'),
                         'Std': float(np.std(vs, ddof=1)) if len(vs)>1 else 0.0})
        pd.DataFrame(rows, columns=['Metric','Mean Value','Std']).to_excel(
            sd / 'summary_metrics.xlsx', index=False, engine='openpyxl')

    def _save_summary_predictions(self, sd: Path) -> None:
        for tag, attr in [('near','_near_data'),('far','_far_data')]:
            y_true = getattr(self, attr)[1].flatten()
            data   = {'y_true': y_true}
            for r in self.all_results:
                data[f"y_pred_fold{r['fold_idx']}"] = (
                    r['metrics']['predictions'][tag].flatten())
            preds = np.array([r['metrics']['predictions'][tag].flatten()
                               for r in self.all_results])
            data['y_pred_mean']   = np.mean(preds, axis=0)
            data['y_pred_std']    = np.std(preds,  axis=0, ddof=1)
            data['residual_mean'] = y_true - data['y_pred_mean']
            pd.DataFrame(data).to_excel(
                sd / f'{tag}_predictions_summary.xlsx',
                index=False, engine='openpyxl')

    def _generate_text_report(self, sd: Path) -> None:
        sep  = '-' * 70
        cfg  = self.config
        lines = [
            '=' * 70,
            'GPR 基线实验 — K-Fold CV 汇总（NaCl–CaCl₂–H₂O 热导率）',
            '=' * 70,
            f"生成时间:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"K-Folds:     {self.n_folds}",
            '', sep, 'GPR 配置', sep,
            f"核函数:               ConstantKernel × Matern(nu=2.5) + WhiteKernel",
            f"n_restarts_optimizer: {cfg.n_restarts_optimizer}",
            f"特征空间:             StandardScaler 标准化",
            f"拟合目标:             TC 原始值（W/m·K，无对数变换）",
            '', sep, '物理评估范围', sep,
            f"T range:   [{cfg.t_min}, {cfg.t_max}] K",
            f"P range:   [{cfg.p_min:.0e}, {cfg.p_max:.0e}] Pa",
            f"CaCl₂:     {cfg.cacl2_range}",
            f"NaCl:      {cfg.nacl_range}",
            '', sep, '汇总统计', sep,
        ]
        keys  = ['train_r2','train_rmse','train_mae',
                 'val_r2',  'val_rmse',  'val_mae',
                 'near_r2', 'near_rmse', 'near_mae',
                 'far_r2',  'far_rmse',  'far_mae']
        names = ['Train R²','Train RMSE','Train MAE',
                 'Val R²',  'Val RMSE',  'Val MAE',
                 'Near-Range R²','Near-Range RMSE','Near-Range MAE',
                 'Far-Range R²', 'Far-Range RMSE', 'Far-Range MAE']
        for k, n in zip(keys, names):
            vs = [r['metrics']['metrics'][k] for r in self.all_results]
            lines.append(f"{n:22s}: {np.mean(vs):.6f} ± {np.std(vs,ddof=1):.6f}")

        lines.append('\n物理评估:')
        for attr, n in [('physics_score','Physics Score'),
                         ('physics_boundary','Boundary Consistency'),
                         ('physics_smoothness','Thermodynamic Smoothness')]:
            vs = [r.get(attr) for r in self.all_results
                  if r.get(attr) is not None
                  and not np.isnan(float(r.get(attr,float('nan'))))]
            if vs: lines.append(f"{n:22s}: {np.mean(vs):.6f} ± {np.std(vs,ddof=1):.6f}")
            else:  lines.append(f"{n:22s}: N/A")

        val_r2 = [r['metrics']['metrics']['val_r2'] for r in self.all_results]
        best   = self.all_results[int(np.argmax(val_r2))]
        lines += [
            '', sep, '最优折（按 Val R²）', sep,
            f"Fold {best['fold_idx']}",
            f"  Val R²:        {best['metrics']['metrics']['val_r2']:.6f}",
            f"  Near-Range R²: {best['metrics']['metrics']['near_r2']:.6f}",
            f"  Far-Range R²:  {best['metrics']['metrics']['far_r2']:.6f}",
            '', sep, '稳定性 (Val R²)', sep,
            f"Std:  {np.std(val_r2, ddof=1):.6f}",
            f"CV:   {np.std(val_r2,ddof=1)/max(abs(np.mean(val_r2)),1e-8)*100:.2f}%",
            '', '=' * 70,
        ]
        report = '\n'.join(lines)
        (sd / 'summary_report.txt').write_text(report, encoding='utf-8')
        self.logger.info(f'\n{report}')

    def _best_model_evaluation(self) -> None:
        val_r2 = [r['metrics']['metrics']['val_r2'] for r in self.all_results]
        fi     = self.all_results[int(np.argmax(val_r2))]['fold_idx']
        self.logger.info(f"\n最佳折: Fold {fi}  Val R²={val_r2[int(np.argmax(val_r2))]:.6f}")

        mp = self.output_dir / f'fold_{fi}' / 'model' / 'gpr.pkl'
        if not mp.exists():
            self.logger.error(f"模型文件不存在: {mp}"); return

        ckpt   = joblib.load(mp)
        gpr, scaler = ckpt['gpr'], ckpt['scaler']
        bd     = self.output_dir / 'best_model';  bd.mkdir(exist_ok=True)
        rows   = [['Best Fold', fi], ['Val R²', val_r2[int(np.argmax(val_r2))]]]

        for tag, X, ya in [('near',*self._near_data),('far',*self._far_data)]:
            yp = predict_gpr(gpr, scaler, X)[0].flatten()
            yt = ya.flatten()
            r2   = float(r2_score(yt, yp))
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            mae  = float(mean_absolute_error(yt, yp))
            label = 'Near-Range' if tag == 'near' else 'Far-Range'
            self.logger.info(f"  best {label}: r²={r2:.4f} rmse={rmse:.5f} mae={mae:.5f}")
            rows += [[f'{label} R²',r2],[f'{label} RMSE',rmse],[f'{label} MAE',mae]]
            pd.DataFrame({
                'T_K': X[:,0], 'P_Pa': X[:,1], 'CaCl2': X[:,2], 'NaCl': X[:,3],
                'TC_true': yt, 'TC_pred': yp, 'residual': yt-yp,
                'abs_error': np.abs(yt-yp),
                'rel_error_%': np.abs(yt-yp)/np.maximum(np.abs(yt),1e-10)*100,
            }).to_excel(bd/f'best_{tag}_predictions.xlsx', index=False, engine='openpyxl')

        pd.DataFrame(rows, columns=['Metric','Value']).to_excel(
            bd/'best_model_metrics.xlsx', index=False, engine='openpyxl')


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    config = GPRExperimentConfig()
    config.k_folds              = 5
    config.kfold_random_state   = 42
    config.t_min                = 290.0
    config.t_max                = 570.0
    config.p_min                = 5e6
    config.p_max                = 5e7
    config.cacl2_range          = (0.0017, 0.0127)
    config.nacl_range           = (0.0063, 0.0482)
    config.n_restarts_optimizer = 10
    config.save_predictions     = True
    config.save_metrics         = True
    config.log_level            = logging.INFO

    manager = KFoldExperimentManager(config, n_folds=config.k_folds)
    manager.run_all_folds()


if __name__ == '__main__':
    main()