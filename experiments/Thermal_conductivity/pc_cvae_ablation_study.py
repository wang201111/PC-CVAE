"""
================================================================================
PC-CVAE Ablation Study - K-Fold Cross-Validation（热导率 NaCl–CaCl₂–H₂O 体系）
================================================================================

直接复用 pc_cvae_viscosity.py，以下为与粘度版的全部差异：

  1. 数据列：[T, P, CaCl₂, NaCl, TC]，X[:,2]=CaCl₂（≡MCH），X[:,3]=NaCl（≡Dec）
     H₂O = 1 - CaCl₂ - NaCl（隐式，≡HMN）

  2. 两个边界模型（CaCl₂-H₂O / NaCl-H₂O），boundary_type 映射：
       NaCl-H₂O  → 'mch_zero'  （CaCl₂=0 边界，模型取 NaCl ≡ Dec 位置）
       CaCl₂-H₂O → 'dec_zero'  （NaCl=0  边界，模型取 CaCl₂ ≡ MCH 位置）
       H₂O=0 边界非物理 → 不添加（LAMBDA_COLLOCATION_HMN=0）

  3. 物理评估使用 ThermalConductivityPhysicsEvaluator，传入 cacl2_range / nacl_range。

  4. T/P 范围：290–570 K，5e6–5e7 Pa。

  5. history 只含 colloc_mch / colloc_dec（无 colloc_hmn）。

pc_cvae_viscosity.py 零改动。
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

from pc_cvae_thermal_conductivity import CVAEConfig, CVAEPhysicsModel, LowDimInfo
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
      X[:,2] = CaCl₂  ≡ MCH 位置（第一组分）
      X[:,3] = NaCl   ≡ Dec 位置（第二组分）
      H₂O = 1 - CaCl₂ - NaCl（隐式，≡ HMN）
    X: (N, 4) [T, P, CaCl₂, NaCl]
    y: (N, 1) TC
    """
    if not filepath.exists():
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    data = pd.read_excel(filepath, engine='openpyxl')
    if data.shape[1] < 5:
        raise ValueError(f"期望至少 5 列 [T, P, CaCl₂, NaCl, TC]，实际: {data.shape[1]}")
    X = data.iloc[:, :4].values.astype(np.float32)
    y = data.iloc[:, 4].values.astype(np.float32)
    return X, y


def move_to_device(model: LowDimEnsemble, device: torch.device) -> LowDimEnsemble:
    model.to(device)
    model.device = device
    return model


def create_k_folds(
    X: np.ndarray, y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dict[str, np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [
        {
            'fold_idx': fi,
            'X_train': X[tr], 'y_train': y[tr],
            'X_val':   X[va], 'y_val':   y[va],
        }
        for fi, (tr, va) in enumerate(kf.split(X))
    ]


# ==============================================================================
# ThermalConductivityPhysicsEvaluator 适配包装器
# ==============================================================================

class _CVAEWrapper:
    """轻量包装器：将 CVAEPhysicsModel.predict(X) 适配为
    trainer.predict(X, return_original_scale=True) 接口，
    供 ThermalConductivityPhysicsEvaluator.evaluate_full() 调用。
    """

    def __init__(self, cvae: CVAEPhysicsModel) -> None:
        self._cvae = cvae

    def predict(self, X: np.ndarray, return_original_scale: bool = True) -> np.ndarray:
        """(N, 4) [T, P, CaCl₂, NaCl] → (N, 1) TC，始终返回原始尺度。"""
        return self._cvae.predict(X)


# ==============================================================================
# 配置
# ==============================================================================

@dataclass
class CVAEExperimentConfig:
    """PC-CVAE 热导率消融实验配置（NaCl–CaCl₂–H₂O 体系）。"""

    # 数据路径
    data_dir: Path = PROJECT_ROOT / 'data' / 'Thermal_conductivity' / 'split_by_temperature'
    train_data_file:  str = 'data.xlsx'
    near_extrap_file: str = 'near-range extrapolation.xlsx'
    far_extrap_file:  str = 'far-range extrapolation.xlsx'

    # 边界模型路径（两个二元体系）
    models_dir: Path = PROJECT_ROOT / 'models' / 'Low_dim_model' / 'Thermal_conductivity'
    cacl2_h2o_model_file: str = 'CaCl2-H2O.pth'   # dec_zero（nacl_zero 边界）
    nacl_h2o_model_file:  str = 'NaCl-H2O.pth'    # mch_zero（cacl2_zero 边界）

    # PC-CVAE 超参数
    cvae_config: CVAEConfig = field(default_factory=lambda: CVAEConfig(
        LATENT_DIM=2,
        HIDDEN_DIMS=[128, 256, 256, 128],
        DROPOUT=0.1,
        LEARNING_RATE=1e-3,
        BATCH_SIZE=64,
        N_EPOCHS=500,
        WEIGHT_DECAY=1e-5,
        LAMBDA_KL=0.001,
        LAMBDA_COLLOCATION_MCH=0.5,   # 对应 cacl2_zero 约束（'mch_zero'）
        LAMBDA_COLLOCATION_DEC=0.5,   # 对应 nacl_zero  约束（'dec_zero'）
        LAMBDA_COLLOCATION_HMN=0.0,   # h2o_zero 非物理 → 关闭
        N_COLLOCATION_POINTS=64,
        COLLOCATION_T_RANGE=(290.0, 570.0),
        COLLOCATION_P_RANGE=(5e6,   5e7),
        Z_LOW=-2.0,
        Z_HIGH=2.0,
        Z_COLLOC_WIDTH=0.5,
        PHI_HIDDEN_DIMS=[64, 64],
        LAMBDA_CYCLE=1.0,
        N_CYCLE_POINTS=64,
        CYCLE_T_RANGE=(290.0, 570.0),
        CYCLE_P_RANGE=(5e6,   5e7),
        USE_EARLY_STOPPING=False,
        USE_LR_SCHEDULER=True,
        LR_SCHEDULER_TYPE='cosine',
        LR_MIN=1e-5,
        DEVICE='auto',
        VERBOSE=False,
    ))

    # 物理评估范围
    t_min: float = 290.0
    t_max: float = 570.0
    p_min: float = 5e6
    p_max: float = 5e7
    # 组成范围（实测摩尔分数，用于物理评估采样）
    cacl2_range:  Tuple[float, float] = (0.93,   1.0)
    nacl_range: Tuple[float, float] = (0.0063, 0.048)

    # 交叉验证
    k_folds: int = 5
    kfold_random_state: int = 42

    # 输出
    output_dir: Path = PROJECT_ROOT / 'results' / 'tc' / 'ablation' / 'CVAE_results'
    save_predictions:  bool = True
    save_metrics:      bool = True
    save_cvae_history: bool = True
    excel_prefix:      str  = 'cvae_'

    device:    str = 'auto'
    log_level: int = logging.INFO

    def __post_init__(self) -> None:
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
    """对单个 K-fold 切分执行完整 PC-CVAE 流水线。"""

    def __init__(
        self,
        config:          CVAEExperimentConfig,
        model_cacl2_h2o: Optional[LowDimEnsemble],   # dec_zero（nacl_zero 边界）
        model_nacl_h2o:  Optional[LowDimEnsemble],   # mch_zero（cacl2_zero 边界）
    ) -> None:
        self.config          = config
        self.logger          = get_logger(self.__class__.__name__, config.log_level)
        self.device          = torch.device(config.device)
        self.model_cacl2_h2o = model_cacl2_h2o
        self.model_nacl_h2o  = model_nacl_h2o

    def run(
        self,
        fold_idx: int,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        X_near:  np.ndarray, y_near:  np.ndarray,
        X_far:   np.ndarray, y_far:   np.ndarray,
        fold_dir: Path,
    ) -> Dict[str, Any]:
        """三步流水线：训练 → 评估 → 物理一致性。"""
        self.logger.info("=" * 70)
        self.logger.info(f"Fold {fold_idx}")
        self.logger.info("=" * 70)

        self.logger.info("[Step 1] 训练 PC-CVAE（热导率版，固定 epoch）")
        cvae, cvae_history = self._train_cvae(X_train, y_train)

        self.logger.info("[Step 2] 经 φ 头直接评估")
        metrics = self._evaluate_cvae(
            cvae, X_train, y_train, X_val, y_val, X_near, y_near, X_far, y_far,
        )

        self.logger.info("[Step 3] 物理一致性评估（双支柱框架）")
        physics_results = self._compute_physics_metrics(cvae)

        self._save_fold_results(fold_idx, fold_dir, metrics, physics_results, cvae_history, cvae)

        return {
            'fold_idx':           fold_idx,
            'metrics':            metrics,
            'physics_score':      physics_results.get('physics_score',      None),
            'physics_boundary':   physics_results.get('physics_boundary',   None),
            'physics_smoothness': physics_results.get('physics_smoothness', None),
        }

    # ------------------------------------------------------------------
    # Step 1：训练
    # ------------------------------------------------------------------

    def _train_cvae(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[CVAEPhysicsModel, dict]:
        """构建 LowDimInfo 列表并训练 CVAEPhysicsModel。

        边界类型映射（复用粘度版 boundary_type，不改模型代码）：
          NaCl-H₂O  → 'mch_zero'
            CaCl₂=0 边界（X[:,2]=CaCl₂ 为 0），模型取第二组分 NaCl（≡ Dec 位置）
          CaCl₂-H₂O → 'dec_zero'
            NaCl=0  边界（X[:,3]=NaCl  为 0），模型取第一组分 CaCl₂（≡ MCH 位置）
          H₂O=0 → 不加（非物理，LAMBDA_COLLOCATION_HMN=0）
        """
        low_dim_list: Optional[List[LowDimInfo]] = None

        if all(m is not None for m in (self.model_cacl2_h2o, self.model_nacl_h2o)):
            low_dim_list = [
                LowDimInfo(
                    model=self.model_nacl_h2o,
                    name='NaCl-H2O',
                    boundary_type='mch_zero',   # CaCl₂=0，模型取 NaCl
                ),
                LowDimInfo(
                    model=self.model_cacl2_h2o,
                    name='CaCl2-H2O',
                    boundary_type='dec_zero',   # NaCl=0，模型取 CaCl₂
                ),
                # H₂O=0（'hmn_zero'）非物理 → 不加
            ]
        else:
            self.logger.warning("边界模型缺失 — 配点约束已禁用")

        cvae = CVAEPhysicsModel(config=self.config.cvae_config)

        history = cvae.fit(
            X=X_train,
            y=y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train,
            low_dim_list=low_dim_list,
            X_val=None,
            y_val=None,
        )

        tr = history.get('train_loss', [])
        if tr:
            self.logger.info("=" * 80)
            self.logger.info("损失分项量级分析（各关键轮次）")
            self.logger.info("=" * 80)
            check_epochs = [0,
                            len(tr) // 4,
                            len(tr) // 2,
                            3 * len(tr) // 4,
                            len(tr) - 1]
            check_epochs = sorted(set(check_epochs))
            header = (f"{'Epoch':>7}  {'total':>9}  {'recon':>9}  {'kl*λ':>9}  "
                      f"{'cycle':>9}  {'nacl×λ':>9}  {'cacl2×λ':>10}  "
                      f"{'raw_nacl':>10}  {'raw_cacl2':>10}")
            self.logger.info(header)
            self.logger.info("-" * 90)
            cfg = self.config.cvae_config
            for ep in check_epochs:
                raw_mch = history['train_colloc_mch'][ep] if ep < len(history.get('train_colloc_mch', [])) else float('nan')
                raw_dec = history['train_colloc_dec'][ep] if ep < len(history.get('train_colloc_dec', [])) else float('nan')
                weighted_mch = raw_mch * cfg.LAMBDA_COLLOCATION_MCH
                weighted_dec = raw_dec * cfg.LAMBDA_COLLOCATION_DEC
                self.logger.info(
                    f"{ep+1:>7}  "
                    f"{tr[ep]:>9.5f}  "
                    f"{history['train_recon'][ep]:>9.5f}  "
                    f"{history['train_kl'][ep] * cfg.LAMBDA_KL:>9.5f}  "
                    f"{history['train_cycle'][ep]:>9.5f}  "
                    f"{weighted_mch:>9.5f}  "
                    f"{weighted_dec:>10.5f}  "
                    f"{raw_mch:>10.5f}  "
                    f"{raw_dec:>10.5f}"
                )
            self.logger.info("-" * 90)

            # 末轮分项百分比
            ep = len(tr) - 1
            total_last = tr[ep]
            raw_mch = history['train_colloc_mch'][ep] if ep < len(history.get('train_colloc_mch', [])) else 0
            raw_dec = history['train_colloc_dec'][ep] if ep < len(history.get('train_colloc_dec', [])) else 0
            items = [
                ('recon',   history['train_recon'][ep]),
                ('nacl×λ',  raw_mch * cfg.LAMBDA_COLLOCATION_MCH),
                ('cacl2×λ', raw_dec * cfg.LAMBDA_COLLOCATION_DEC),
                ('cycle',   history['train_cycle'][ep]),
                ('kl×λ',    history['train_kl'][ep] * cfg.LAMBDA_KL),
            ]
            self.logger.info("末轮各分项占总损失百分比：")
            for name, val in items:
                pct = val / max(abs(total_last), 1e-10) * 100
                bar = '█' * int(pct / 2)
                self.logger.info(f"  {name:<10}: {val:.5f}  ({pct:5.1f}%)  {bar}")
            self.logger.info("-" * 80)

            self.logger.info(
                f"  训练完成 total={tr[-1]:.5f}  "
                f"recon={history['train_recon'][-1]:.5f}  "
                f"kl={history['train_kl'][-1]:.5f}  "
                f"cycle={history['train_cycle'][-1]:.5f}  "
                f"nacl={history['train_colloc_mch'][-1]:.5f}  "
                f"cacl2={history['train_colloc_dec'][-1]:.5f}"
            )
        return cvae, history

    # ------------------------------------------------------------------
    # Step 2：评估
    # ------------------------------------------------------------------

    def _evaluate_cvae(
        self,
        cvae: CVAEPhysicsModel,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        X_near:  np.ndarray, y_near:  np.ndarray,
        X_far:   np.ndarray, y_far:   np.ndarray,
    ) -> Dict[str, Any]:
        """通过 cvae.predict(X) 计算 train/val/near/far 四集 R²/RMSE/MAE。

        cvae.predict(X) 接受 (N, 4) [T, P, CaCl₂, NaCl]，返回 (N, 1) TC。
        """
        splits = {
            'train': (X_train, y_train),
            'val':   (X_val,   y_val),
            'near':  (X_near,  y_near),
            'far':   (X_far,   y_far),
        }

        preds = {s: cvae.predict(X).flatten() for s, (X, _) in splits.items()}
        trues = {s: y.flatten() for s, (_, y) in splits.items()}

        metrics_result: Dict[str, float] = {}
        for s in ('train', 'val', 'near', 'far'):
            metrics_result[f'{s}_r2']   = float(r2_score(trues[s], preds[s]))
            metrics_result[f'{s}_rmse'] = float(np.sqrt(mean_squared_error(trues[s], preds[s])))
            metrics_result[f'{s}_mae']  = float(mean_absolute_error(trues[s], preds[s]))

        self.logger.info("-" * 60)
        self.logger.info("预测性能：")
        for s, label in [('train','train'), ('val','val'), ('near','near'), ('far','far')]:
            self.logger.info(
                f"  {label:<5}  R²= {metrics_result[f'{s}_r2']:.4f}  "
                f"RMSE= {metrics_result[f'{s}_rmse']:.5f}  "
                f"MAE= {metrics_result[f'{s}_mae']:.5f}"
            )
        near_decay = metrics_result['near_r2'] - metrics_result['train_r2']
        far_decay  = metrics_result['far_r2']  - metrics_result['train_r2']
        self.logger.info(f"  near vs train 衰减: {near_decay:+.4f}")
        self.logger.info(f"  far  vs train 衰减: {far_decay:+.4f}")
        self.logger.info("-" * 60)

        return {
            'metrics':     metrics_result,
            'predictions': preds,
            'true_values': trues,
        }

    # ------------------------------------------------------------------
    # Step 3：物理一致性
    # ------------------------------------------------------------------

    def _compute_physics_metrics(self, cvae: CVAEPhysicsModel) -> Dict[str, float]:
        """通过 ThermalConductivityPhysicsEvaluator 双支柱框架评估物理一致性。"""
        if any(m is None for m in (self.model_cacl2_h2o, self.model_nacl_h2o)):
            self.logger.warning("边界模型缺失 — 跳过物理评估")
            return {}

        try:
            evaluator = ThermalConductivityPhysicsEvaluator(
                teacher_models=(
                    self.model_cacl2_h2o,   # dec_zero / nacl_zero 边界
                    self.model_nacl_h2o,    # mch_zero / cacl2_zero 边界
                ),
                temp_range=(self.config.t_min, self.config.t_max),
                pressure_range=(self.config.p_min, self.config.p_max),
                cacl2_range=self.config.cacl2_range,
                nacl_range=self.config.nacl_range,
            )

            wrapper = _CVAEWrapper(cvae)
            overall_score, results = evaluator.evaluate_full(wrapper)

            boundary_score   = results.get('boundary_score',   float('nan'))
            smoothness_score = results.get('smoothness_score', float('nan'))

            self.logger.info(
                f"  physics={overall_score:.4f}  "
                f"boundary={boundary_score:.4f}  "
                f"smoothness={smoothness_score:.4f}"
            )
            return {
                'physics_score':      float(overall_score),
                'physics_boundary':   float(boundary_score),
                'physics_smoothness': float(smoothness_score),
            }

        except Exception as e:
            import traceback
            self.logger.error(f"物理评估失败: {e}\n{traceback.format_exc()}")
            return {}

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------

    def _save_fold_results(
        self,
        fold_idx:        int,
        fold_dir:        Path,
        metrics:         Dict,
        physics_results: Dict,
        cvae_history:    dict,
        cvae:            CVAEPhysicsModel,
    ) -> None:
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_dir = fold_dir / 'model'
        model_dir.mkdir(exist_ok=True)
        cvae.save(str(model_dir / 'cvae.pth'))

        if self.config.save_metrics:
            self._save_fold_metrics(metrics, physics_results, fold_dir)
        if self.config.save_predictions:
            self._save_predictions(metrics, fold_dir)
        if self.config.save_cvae_history:
            self._save_cvae_history(cvae_history, fold_dir)

    def _save_fold_metrics(
        self, metrics: Dict, physics_results: Dict, fold_dir: Path
    ) -> None:
        inner = metrics.get('metrics', metrics)
        pe    = physics_results

        physics_score    = pe.get('physics_score',      float('nan')) if pe else float('nan')
        boundary_score   = pe.get('physics_boundary',   float('nan')) if pe else float('nan')
        smoothness_score = pe.get('physics_smoothness', float('nan')) if pe else float('nan')

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
            ['Physics Score',            physics_score],
            ['Boundary Consistency',     boundary_score],
            ['Thermodynamic Smoothness', smoothness_score],
        ]
        pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
            fold_dir / f'{self.config.excel_prefix}metrics.xlsx',
            index=False, engine='openpyxl',
        )

    def _save_predictions(self, metrics: Dict, fold_dir: Path) -> None:
        predictions = metrics.get('predictions', {})
        true_values = metrics.get('true_values', {})
        for split in ('train', 'val', 'near', 'far'):
            if split not in predictions:
                continue
            y_true = true_values[split].flatten()
            y_pred = predictions[split].flatten()
            pd.DataFrame({
                'y_true': y_true, 'y_pred': y_pred, 'residual': y_true - y_pred,
            }).to_excel(
                fold_dir / f'{self.config.excel_prefix}{split}_predictions.xlsx',
                index=False, engine='openpyxl',
            )

    def _save_cvae_history(self, history: dict, fold_dir: Path) -> None:
        """保存每轮损失明细。

        列：epoch, train_total, train_recon, train_kl,
            train_cycle, train_colloc_nacl, train_colloc_cacl2
        （无 colloc_hmn，H₂O=0 边界已禁用）
        """
        train_loss = history.get('train_loss', [])
        if not train_loss:
            return

        rows = []
        for ep, total in enumerate(train_loss):
            def _get(key):
                v = history.get(key, [])
                return v[ep] if ep < len(v) else float('nan')
            rows.append({
                'epoch':               ep,
                'train_total':         total,
                'train_recon':         _get('train_recon'),
                'train_kl':            _get('train_kl'),
                'train_cycle':         _get('train_cycle'),
                'train_colloc_nacl':   _get('train_colloc_mch'),   # mch_zero → cacl2_zero
                'train_colloc_cacl2':  _get('train_colloc_dec'),   # dec_zero → nacl_zero
            })

        df = pd.DataFrame(rows)
        val_loss = history.get('val_loss', [])
        if val_loss:
            df['val_total'] = pd.Series(val_loss)

        df.to_excel(
            fold_dir / f'{self.config.excel_prefix}cvae_history.xlsx',
            index=False, engine='openpyxl',
        )


# ==============================================================================
# K-Fold 实验管理器
# ==============================================================================

class KFoldExperimentManager:
    """统筹所有折的训练、评估与汇总报告生成。"""

    def __init__(self, config: CVAEExperimentConfig, n_folds: int = 5) -> None:
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
        """加载两个低维二元体系模型（所有折共享）。"""
        try:
            device = torch.device(self.config.device)
            paths = {
                'cacl2_h2o': self.config.cacl2_h2o_path,
                'nacl_h2o':  self.config.nacl_h2o_path,
            }
            missing = [k for k, p in paths.items() if not p.exists()]
            if missing:
                self.logger.warning(
                    f"以下边界模型文件缺失: {missing} — 配点约束和物理评估将禁用"
                )
                return

            self.model_cacl2_h2o = move_to_device(
                LowDimEnsemble.load(str(paths['cacl2_h2o'])), device
            )
            self.model_nacl_h2o = move_to_device(
                LowDimEnsemble.load(str(paths['nacl_h2o'])), device
            )
            self.logger.info(
                f"两个边界模型已加载 → {device}  "
                f"(CaCl₂-H₂O: dec_zero/nacl_zero | NaCl-H₂O: mch_zero/cacl2_zero)"
            )

        except Exception as e:
            self.logger.error(f"加载边界模型失败: {e}")

    def run_all_folds(self) -> None:
        self.logger.info("\n" + "█" * 70)
        self.logger.info(
            "PC-CVAE K-Fold 实验（NaCl–CaCl₂–H₂O 热导率）"
        )
        self.logger.info("█" * 70)
        self.logger.info(f"时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  K={self.n_folds}")

        cfg = self.config.cvae_config
        self.logger.info("\n" + "-" * 70)
        self.logger.info("超参数")
        self.logger.info("-" * 70)
        self.logger.info(
            f"latent={cfg.LATENT_DIM} epoch={cfg.N_EPOCHS} lr={cfg.LEARNING_RATE}"
        )
        self.logger.info(
            f"λ_KL={cfg.LAMBDA_KL} "
            f"λ_nacl={cfg.LAMBDA_COLLOCATION_MCH} "   # mch_zero → cacl2_zero
            f"λ_cacl2={cfg.LAMBDA_COLLOCATION_DEC} "  # dec_zero → nacl_zero
            f"λ_cycle={cfg.LAMBDA_CYCLE}"
        )
        self.logger.info(
            f"Colloc T={cfg.COLLOCATION_T_RANGE}  P={cfg.COLLOCATION_P_RANGE}"
        )
        self.logger.info(
            f"Cycle  T={cfg.CYCLE_T_RANGE}  P={cfg.CYCLE_P_RANGE}"
        )

        start = time.time()

        X_train_pool, y_train_pool = load_tc_data(self.config.train_data_path)
        X_near,       y_near       = load_tc_data(self.config.near_extrap_path)
        X_far,        y_far        = load_tc_data(self.config.far_extrap_path)
        self.logger.info(
            f"训练池: {len(X_train_pool)}  近域: {len(X_near)}  远域: {len(X_far)}"
        )
        self._near_data = (X_near, y_near)
        self._far_data  = (X_far,  y_far)

        folds = create_k_folds(
            X_train_pool, y_train_pool,
            n_splits=self.n_folds,
            random_state=self.config.kfold_random_state,
        )
        for f in folds:
            self.logger.info(
                f"  Fold {f['fold_idx']}: train={len(f['X_train'])}  val={len(f['X_val'])}"
            )

        for fold_data in folds:
            fold_idx = fold_data['fold_idx']
            self.logger.info(f"\n{'█' * 70}\nFold {fold_idx + 1}/{self.n_folds}\n{'█' * 70}")
            fold_dir = self.output_dir / f'fold_{fold_idx}'
            fold_dir.mkdir(exist_ok=True)

            X_near, y_near = self._near_data
            X_far,  y_far  = self._far_data
            runner = SingleFoldRunner(
                config=self.config,
                model_cacl2_h2o=self.model_cacl2_h2o,
                model_nacl_h2o=self.model_nacl_h2o,
            )
            result = runner.run(
                fold_idx=fold_idx,
                X_train=fold_data['X_train'], y_train=fold_data['y_train'],
                X_val=fold_data['X_val'],     y_val=fold_data['y_val'],
                X_near=X_near,                y_near=y_near,
                X_far=X_far,                  y_far=y_far,
                fold_dir=fold_dir,
            )
            self.all_results.append(result)

        self._generate_summary()
        self._best_model_evaluation()
        self.logger.info(
            f"\n所有折完成，耗时 {timedelta(seconds=int(time.time() - start))}"
            f"  结果路径: {self.output_dir}"
        )

    def _generate_summary(self) -> None:
        summary_dir = self.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)
        self._save_summary_metrics(summary_dir)
        self._save_summary_predictions(summary_dir)
        self._generate_text_report(summary_dir)

    def _save_summary_metrics(self, summary_dir: Path) -> None:
        STAT_KEYS  = ['train_r2',  'train_rmse', 'train_mae',
                      'val_r2',    'val_rmse',   'val_mae',
                      'near_r2',   'near_rmse',  'near_mae',
                      'far_r2',    'far_rmse',   'far_mae']
        STAT_NAMES = ['Train R²',  'Train RMSE', 'Train MAE',
                      'Val R²',    'Val RMSE',   'Val MAE',
                      'Near-Range R²',   'Near-Range RMSE', 'Near-Range MAE',
                      'Far-Range R²',    'Far-Range RMSE',  'Far-Range MAE']
        PHYS_ATTRS = [
            ('physics_score',      'Physics Score'),
            ('physics_boundary',   'Boundary Consistency'),
            ('physics_smoothness', 'Thermodynamic Smoothness'),
        ]

        rows: List[Dict[str, Any]] = []

        for key, name in zip(STAT_KEYS, STAT_NAMES):
            vals = [r['metrics']['metrics'][key] for r in self.all_results]
            rows.append({
                'Metric':     name,
                'Mean Value': float(np.mean(vals)),
                'Std':        float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })

        for attr, label in PHYS_ATTRS:
            vals = [
                r.get(attr) for r in self.all_results
                if r.get(attr) is not None
                and not np.isnan(float(r.get(attr, float('nan'))))
            ]
            rows.append({
                'Metric':     label,
                'Mean Value': float(np.mean(vals)) if vals else float('nan'),
                'Std':        float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })

        pd.DataFrame(rows, columns=['Metric', 'Mean Value', 'Std']).to_excel(
            summary_dir / 'summary_metrics.xlsx', index=False, engine='openpyxl'
        )

    def _save_summary_predictions(self, summary_dir: Path) -> None:
        for tag, data_attr in [('near', '_near_data'), ('far', '_far_data')]:
            y_true = getattr(self, data_attr)[1].flatten()
            data   = {'y_true': y_true}
            for r in self.all_results:
                data[f"y_pred_fold{r['fold_idx']}"] = (
                    r['metrics']['predictions'][tag].flatten()
                )
            preds = np.array([
                r['metrics']['predictions'][tag].flatten() for r in self.all_results
            ])
            data['y_pred_mean']   = np.mean(preds, axis=0)
            data['y_pred_std']    = np.std(preds,  axis=0, ddof=1)
            data['residual_mean'] = y_true - data['y_pred_mean']
            pd.DataFrame(data).to_excel(
                summary_dir / f'{tag}_predictions_summary.xlsx',
                index=False, engine='openpyxl',
            )

    def _generate_text_report(self, summary_dir: Path) -> None:
        cfg = self.config.cvae_config
        sep = '-' * 70
        lines = [
            '=' * 70,
            'PC-CVAE 消融实验 — K-Fold CV 汇总（NaCl–CaCl₂–H₂O 热导率，直接 φ 预测）',
            '=' * 70,
            f"生成时间:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"K-Folds:     {self.n_folds}",
            f"Device:      {self.config.device}",
            '', sep, 'PC-CVAE 配置', sep,
            f"Latent dim:           {cfg.LATENT_DIM}",
            f"Hidden dims:          {cfg.HIDDEN_DIMS}",
            f"N_EPOCHS:             {cfg.N_EPOCHS}  (固定，无早停)",
            f"λ_KL:                 {cfg.LAMBDA_KL}",
            f"λ_nacl(mch_zero):     {cfg.LAMBDA_COLLOCATION_MCH}  → CaCl₂=0 边界",
            f"λ_cacl2(dec_zero):    {cfg.LAMBDA_COLLOCATION_DEC}  → NaCl=0  边界",
            f"λ_h2o(hmn_zero):      {cfg.LAMBDA_COLLOCATION_HMN}  → 禁用（非物理）",
            f"N_collocation:        {cfg.N_COLLOCATION_POINTS}",
            f"Collocation T:        {cfg.COLLOCATION_T_RANGE}",
            f"Collocation P:        {cfg.COLLOCATION_P_RANGE}",
            f"Z range:              [{cfg.Z_LOW}, {cfg.Z_HIGH}]",
            f"Z colloc width:       {cfg.Z_COLLOC_WIDTH}",
            '', sep, 'φ 头（逆流形推断头）', sep,
            f"φ hidden dims:        {cfg.PHI_HIDDEN_DIMS}",
            f"λ_cycle:              {cfg.LAMBDA_CYCLE}",
            f"N_cycle_points:       {cfg.N_CYCLE_POINTS}",
            f"Cycle T range:        {cfg.CYCLE_T_RANGE}",
            f"Cycle P range:        {cfg.CYCLE_P_RANGE}",
            '', sep, '物理评估范围', sep,
            f"T range:              [{self.config.t_min}, {self.config.t_max}] K",
            f"P range:              [{self.config.p_min:.0e}, {self.config.p_max:.0e}] Pa",
            f"H₂O range:            {self.config.cacl2_range}",
            f"NaCl range:           {self.config.nacl_range}",
            '', sep, '汇总统计', sep,
        ]

        keys  = ['train_r2', 'train_rmse', 'train_mae',
                 'val_r2',   'val_rmse',   'val_mae',
                 'near_r2',  'near_rmse',  'near_mae',
                 'far_r2',   'far_rmse',   'far_mae']
        names = ['Train R²', 'Train RMSE', 'Train MAE',
                 'Val R²',   'Val RMSE',   'Val MAE',
                 'Near-Range R²',   'Near-Range RMSE', 'Near-Range MAE',
                 'Far-Range R²',    'Far-Range RMSE',  'Far-Range MAE']
        for k, name in zip(keys, names):
            vals = [r['metrics']['metrics'][k] for r in self.all_results]
            lines.append(f"{name:22s}: {np.mean(vals):.6f} ± {np.std(vals, ddof=1):.6f}")

        lines.append('\n物理评估:')
        for attr, name in [
            ('physics_score',      'Physics Score'),
            ('physics_boundary',   'Boundary Consistency'),
            ('physics_smoothness', 'Thermodynamic Smoothness'),
        ]:
            vals = [
                r.get(attr) for r in self.all_results
                if r.get(attr) is not None and not np.isnan(float(r.get(attr, float('nan'))))
            ]
            if vals:
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                lines.append(f"{name:22s}: {np.mean(vals):.6f} ± {std:.6f}")
            else:
                lines.append(f"{name:22s}: N/A")

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
            f"CV:   {np.std(val_r2, ddof=1) / max(abs(np.mean(val_r2)), 1e-8) * 100:.2f}%",
            '', '=' * 70,
        ]

        report = '\n'.join(lines)
        (summary_dir / 'summary_report.txt').write_text(report, encoding='utf-8')
        self.logger.info(f'\n{report}')

    def _best_model_evaluation(self) -> None:
        """所有折完成后，按 Val R² 选出最佳折，加载 CVAE 在 near/far 域独立评估。"""
        val_r2_list = [r['metrics']['metrics']['val_r2'] for r in self.all_results]
        best_idx    = int(np.argmax(val_r2_list))
        best_result = self.all_results[best_idx]
        fold_idx    = best_result['fold_idx']

        self.logger.info(
            f"\n最佳折: Fold {fold_idx}  Val R²={val_r2_list[best_idx]:.6f}"
        )

        model_path = self.output_dir / f'fold_{fold_idx}' / 'model' / 'cvae.pth'
        if not model_path.exists():
            self.logger.error(f"最佳折模型文件不存在，跳过最佳模型评估: {model_path}")
            return

        cvae     = CVAEPhysicsModel.load(str(model_path))
        best_dir = self.output_dir / 'best_model'
        best_dir.mkdir(exist_ok=True)

        X_near, y_near = self._near_data
        X_far,  y_far  = self._far_data

        summary_rows = [
            ['Best Fold', fold_idx],
            ['Val R²',    val_r2_list[best_idx]],
        ]

        for tag, X, y_true_arr in [('near', X_near, y_near), ('far', X_far, y_far)]:
            y_pred = cvae.predict(X).flatten()
            y_true = y_true_arr.flatten()
            r2   = float(r2_score(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae  = float(mean_absolute_error(y_true, y_pred))
            label = 'Near-Range' if tag == 'near' else 'Far-Range'
            self.logger.info(
                f"  best_model {label}: r²={r2:.4f}  rmse={rmse:.4f}  mae={mae:.4f}"
            )
            summary_rows += [
                [f'{label} R²',   r2],
                [f'{label} RMSE', rmse],
                [f'{label} MAE',  mae],
            ]
            pd.DataFrame({
                'y_true':   y_true,
                'y_pred':   y_pred,
                'residual': y_true - y_pred,
            }).to_excel(
                best_dir / f'best_{tag}_predictions.xlsx',
                index=False, engine='openpyxl',
            )

        pd.DataFrame(summary_rows, columns=['Metric', 'Value']).to_excel(
            best_dir / 'best_model_metrics.xlsx', index=False, engine='openpyxl',
        )


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    config = CVAEExperimentConfig()

    config.k_folds            = 5
    config.kfold_random_state = 42
    config.t_min              = 290.0
    config.t_max              = 570.0
    config.p_min              = 5e6
    config.p_max              = 5e7
    config.cacl2_range = (0.0017, 0.0127)
    config.nacl_range  = (0.0063, 0.0482)
    config.save_predictions   = True
    config.save_metrics       = True
    config.save_cvae_history  = True
    config.log_level          = logging.INFO

    # PC-CVAE 超参数
    config.cvae_config.LATENT_DIM                = 2
    config.cvae_config.N_EPOCHS                  = 500
    config.cvae_config.LEARNING_RATE             = 1e-3
    config.cvae_config.LAMBDA_KL                 = 0.001
    config.cvae_config.LAMBDA_COLLOCATION_MCH    = 0.1  # cacl2_zero（mch_zero 边界）
    config.cvae_config.LAMBDA_COLLOCATION_DEC    = 0.1   # nacl_zero （dec_zero 边界）
    config.cvae_config.LAMBDA_COLLOCATION_HMN    = 0.0   # h2o_zero  → 禁用
    config.cvae_config.N_COLLOCATION_POINTS      = 64
    config.cvae_config.COLLOCATION_T_RANGE       = (290.0, 570.0)
    config.cvae_config.COLLOCATION_P_RANGE       = (5e6,   5e7)
    config.cvae_config.Z_LOW                     = -2.0
    config.cvae_config.Z_HIGH                    = 2.0
    config.cvae_config.Z_COLLOC_WIDTH            = 0.5
    config.cvae_config.PHI_HIDDEN_DIMS           = [64, 64]
    config.cvae_config.LAMBDA_CYCLE              = 1.0
    config.cvae_config.N_CYCLE_POINTS            = 64
    config.cvae_config.CYCLE_T_RANGE             = (290.0, 570.0)
    config.cvae_config.CYCLE_P_RANGE             = (5e6,   5e7)
    config.cvae_config.USE_EARLY_STOPPING        = False
    config.cvae_config.DEVICE                    = config.device

    manager = KFoldExperimentManager(config, n_folds=config.k_folds)
    manager.run_all_folds()


if __name__ == '__main__':
    main()