"""
二元/三元数据边界对应关系完整分析脚本

核心问题：三元体系 NaCl→0 时是否真正对应 H₂O-CaCl₂ 二元系？
         三元体系 CaCl₂→0 时是否真正对应 H₂O-NaCl 二元系？

分析内容：
  1. 打印三个数据集的基本统计（T/P/组成范围对比）
  2. 三元数据的 H₂O + NaCl + CaCl₂ 是否守恒（质量分数之和）
  3. 三元数据在 NaCl→0 和 CaCl₂→0 附近的真实 TC 与二元模型预测的对比
  4. 全套可交互 3D/2D 可视化
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE_DIR = Path(r'E:\sci-4\返修\新的修改模型 - 副本')
sys.path.append(str(BASE_DIR / 'src' / 'models'))

from low_dim_model import LowDimEnsemble
from sklearn.metrics import r2_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

RESULTS_DIR = BASE_DIR / 'results' / 'boundary_analysis'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 配置：数据路径
# =============================================================================
TERNARY_PATHS = {
    'train': BASE_DIR / 'data' / 'Thermal_conductivity' / 'split_by_temperature' / 'interpolation domain.xlsx',
    'near':  BASE_DIR / 'data' / 'Thermal_conductivity' / 'split_by_temperature' / 'near-range extrapolation.xlsx',
    'far':   BASE_DIR / 'data' / 'Thermal_conductivity' / 'split_by_temperature' / 'far-range extrapolation.xlsx',
}
BINARY_DATA_PATHS = {
    'CaCl2-H2O': BASE_DIR / 'data' / 'Thermal_conductivity' / 'raw' / 'binary' / 'CaCl2-H2O.xlsx',
    'NaCl-H2O':  BASE_DIR / 'data' / 'Thermal_conductivity' / 'raw' / 'binary' / 'NaCl-H2O.xlsx',
}
BINARY_MODEL_PATHS = {
    'CaCl2-H2O': BASE_DIR / 'models' / 'Low_dim_model' / 'Thermal_conductivity' / 'CaCl2-H2O.pth',
    'NaCl-H2O':  BASE_DIR / 'models' / 'Low_dim_model' / 'Thermal_conductivity' / 'NaCl-H2O.pth',
}

sep = '=' * 70
print(f"\n{sep}")
print("二元/三元数据边界对应关系分析")
print(sep)

# =============================================================================
# 1. 加载所有数据
# =============================================================================
print("\n── 1. 加载数据 ──────────────────────────────────────────────────")

# 三元数据（合并）
ternary_dfs = {}
for split, path in TERNARY_PATHS.items():
    if path.exists():
        df = pd.read_excel(path)
        ternary_dfs[split] = df
        print(f"  三元/{split}: {df.shape}  列={list(df.columns)}")
    else:
        print(f"  ✗ 文件不存在: {path}")

# 合并三元数据
all_tern = pd.concat(ternary_dfs.values(), ignore_index=True)
# 列：T, P, H2O, NaCl, TC
T3   = all_tern.iloc[:, 0].values.astype(np.float32)
P3   = all_tern.iloc[:, 1].values.astype(np.float32)
H2O3 = all_tern.iloc[:, 2].values.astype(np.float32)
NaCl3= all_tern.iloc[:, 3].values.astype(np.float32)
TC3  = all_tern.iloc[:, 4].values.astype(np.float32)
CaCl2_3 = 1.0 - H2O3 - NaCl3   # 隐式第三组分

print(f"\n  三元合并: {len(all_tern)} 行")
print(f"    T    ∈ [{T3.min():.1f}, {T3.max():.1f}] K  唯一值: {np.sort(np.unique(T3))}")
print(f"    P    ∈ [{P3.min():.2e}, {P3.max():.2e}] Pa")
print(f"    H₂O  ∈ [{H2O3.min():.5f}, {H2O3.max():.5f}]  唯一值: {np.sort(np.unique(H2O3))}")
print(f"    NaCl ∈ [{NaCl3.min():.5f}, {NaCl3.max():.5f}]  唯一值: {np.sort(np.unique(NaCl3))}")
print(f"    CaCl₂ = 1-H₂O-NaCl ∈ [{CaCl2_3.min():.5f}, {CaCl2_3.max():.5f}]")
print(f"    TC   ∈ [{TC3.min():.5f}, {TC3.max():.5f}] W/m·K")

# 检查质量守恒
sum_comp = H2O3 + NaCl3 + CaCl2_3
print(f"\n  质量守恒检验（H₂O+NaCl+CaCl₂）:")
print(f"    min={sum_comp.min():.6f}  max={sum_comp.max():.6f}  "
      f"mean={sum_comp.mean():.6f}  (应=1.0)")
if abs(sum_comp.mean() - 1.0) > 0.01:
    print("    ⚠ 组成之和偏离1.0！数据可能有问题")
else:
    print("    ✓ 组成守恒正常")

# 二元数据
binary_data = {}
for name, path in BINARY_DATA_PATHS.items():
    if path.exists():
        df = pd.read_excel(path)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)
        binary_data[name] = {'X': X, 'y': y}
        print(f"\n  二元/{name}: {df.shape}  列={list(df.columns)}")
        print(f"    T   ∈ [{X[:,0].min():.1f}, {X[:,0].max():.1f}] K")
        print(f"    P   ∈ [{X[:,1].min():.2e}, {X[:,1].max():.2e}] Pa")
        print(f"    H₂O ∈ [{X[:,2].min():.5f}, {X[:,2].max():.5f}]")
        print(f"    TC  ∈ [{y.min():.5f}, {y.max():.5f}] W/m·K")
    else:
        print(f"  ✗ 文件不存在: {path}")

# 加载二元模型
binary_models = {}
for name, path in BINARY_MODEL_PATHS.items():
    if path.exists():
        binary_models[name] = LowDimEnsemble.load(str(path))
        binary_models[name].eval()
        print(f"\n  模型/{name} 加载成功  input_dim={binary_models[name].input_dim}")
    else:
        print(f"  ✗ 模型不存在: {path}")

# =============================================================================
# 2. T/P 范围对比：三元 vs 二元
# =============================================================================
print(f"\n── 2. T/P 范围对比 ───────────────────────────────────────────────")

print(f"\n  三元训练集 T: {np.sort(np.unique(all_tern[all_tern.index < len(ternary_dfs.get('train',pd.DataFrame()))].iloc[:,0]))}")
print(f"  CaCl₂-H₂O T: {np.sort(np.unique(binary_data.get('CaCl2-H2O',{}).get('X',np.zeros((1,3)))[:,0]))}")
print(f"  NaCl-H₂O  T: {np.sort(np.unique(binary_data.get('NaCl-H2O', {}).get('X',np.zeros((1,3)))[:,0]))}")

print(f"\n  三元 P 范围:      [{P3.min():.2e}, {P3.max():.2e}]")
for name, bd in binary_data.items():
    print(f"  {name} P 范围: [{bd['X'][:,1].min():.2e}, {bd['X'][:,1].max():.2e}]")

# T/P 重叠检验
for name, bd in binary_data.items():
    T_bin = bd['X'][:,0]
    P_bin = bd['X'][:,1]
    T_overlap = (T3.min() <= T_bin.max()) and (T3.max() >= T_bin.min())
    P_overlap = (P3.min() <= P_bin.max()) and (P3.max() >= P_bin.min())
    print(f"\n  {name} vs 三元:")
    print(f"    T 重叠: {'✓' if T_overlap else '✗'}  "
          f"三元[{T3.min():.1f},{T3.max():.1f}] vs 二元[{T_bin.min():.1f},{T_bin.max():.1f}]")
    print(f"    P 重叠: {'✓' if P_overlap else '✗'}  "
          f"三元[{P3.min():.2e},{P3.max():.2e}] vs 二元[{P_bin.min():.2e},{P_bin.max():.2e}]")

# =============================================================================
# 3. 边界对应直接验证：在三元数据中找 NaCl≈0 和 CaCl₂≈0 的点
# =============================================================================
print(f"\n── 3. 边界对应直接验证 ───────────────────────────────────────────")

for boundary_name, comp_vals, comp_label, model_key in [
    ('nacl_zero',  NaCl3,   'NaCl',  'CaCl2-H2O'),
    ('cacl2_zero', CaCl2_3, 'CaCl₂', 'NaCl-H2O'),
]:
    model = binary_models.get(model_key)
    print(f"\n  {boundary_name}（{comp_label}→0，对应 {model_key}）:")
    print(f"    {comp_label} 的唯一值: {np.sort(np.unique(comp_vals))}")
    print(f"    {comp_label} 最小值: {comp_vals.min():.6f}")

    # 找 comp 最小的点作为"边界附近"
    thresh = comp_vals.min() + comp_vals.std() * 0.1
    mask   = comp_vals <= thresh
    n_mask = mask.sum()
    print(f"    {comp_label} ≤ {thresh:.5f} 的点数: {n_mask}")

    if n_mask == 0 or model is None:
        print(f"    ✗ 无法验证（数据或模型缺失）")
        continue

    # 用二元模型预测这些点
    X_boundary = np.column_stack([T3[mask], P3[mask], H2O3[mask]]).astype(np.float32)
    TC_bin_pred = model(X_boundary).flatten()
    TC_tern_true= TC3[mask]

    r2   = float(r2_score(TC_tern_true, TC_bin_pred))
    rmse = float(np.sqrt(np.mean((TC_bin_pred - TC_tern_true)**2)))
    bias = float(np.mean(TC_bin_pred - TC_tern_true))

    print(f"    边界点 TC 对比: R²={r2:.4f}  RMSE={rmse:.5f}  bias={bias:+.5f}")
    print(f"    三元真实值 ∈ [{TC_tern_true.min():.5f}, {TC_tern_true.max():.5f}]")
    print(f"    二元预测值 ∈ [{TC_bin_pred.min():.5f}, {TC_bin_pred.max():.5f}]")

    if r2 < 0.5:
        print(f"    ⚠ R² < 0.5！三元边界数据与二元模型不对应，边界约束无法生效！")
    elif r2 < 0.8:
        print(f"    ⚠ R² < 0.8，对应关系较弱，建议检查数据")
    else:
        print(f"    ✓ 边界对应关系良好")

# =============================================================================
# 4. 可视化
# =============================================================================
print(f"\n── 4. 生成可视化 ─────────────────────────────────────────────────")

colors_split = {
    'train': '#1f77b4', 'near': '#ff7f0e', 'far': '#d62728'
}
split_labels = {
    'train': f'训练集({len(ternary_dfs.get("train",pd.DataFrame()))})',
    'near':  f'近外推({len(ternary_dfs.get("near", pd.DataFrame()))})',
    'far':   f'远外推({len(ternary_dfs.get("far",  pd.DataFrame()))})',
}

# ── 图1：三元数据组成分布（H₂O vs NaCl vs TC，按T着色）─────────────────
fig1 = go.Figure()
offset = 0
for split, df in ternary_dfs.items():
    n = len(df)
    T_s   = df.iloc[:, 0].values
    H2O_s = df.iloc[:, 2].values
    NaCl_s= df.iloc[:, 3].values
    TC_s  = df.iloc[:, 4].values
    fig1.add_trace(go.Scatter3d(
        x=H2O_s, y=NaCl_s, z=TC_s,
        mode='markers',
        marker=dict(size=4, color=T_s, colorscale='Thermal',
                    showscale=(split=='train'),
                    colorbar=dict(title='T (K)') if split=='train' else None,
                    symbol='circle' if split=='train' else
                            'square' if split=='near' else 'diamond',
                    line=dict(width=0.3, color='white')),
        name=split_labels[split],
        customdata=np.column_stack([T_s, df.iloc[:,1].values,
                                    1-H2O_s-NaCl_s]),
        hovertemplate=('H₂O=%{x:.5f}  NaCl=%{y:.5f}<br>'
                       'TC=%{z:.5f}<br>'
                       'T=%{customdata[0]:.1f}K  P=%{customdata[1]:.2e}Pa<br>'
                       'CaCl₂=%{customdata[2]:.5f}<extra>' +
                       split_labels[split] + '</extra>'),
    ))
    offset += n

fig1.update_layout(
    title=dict(text='三元数据分布（H₂O × NaCl × TC，按T着色）', x=0.5),
    scene=dict(xaxis_title='H₂O', yaxis_title='NaCl', zaxis_title='TC (W/m·K)'),
    width=950, height=700)
fig1.write_html(str(RESULTS_DIR / 'ternary_distribution.html'))
print("  ✓ ternary_distribution.html")

# ── 图2：T/P 范围对比（三元 vs 二元）──────────────────────────────────
fig2 = make_subplots(rows=1, cols=2,
    subplot_titles=['T-P 分布：三元 vs 二元（CaCl₂-H₂O）',
                    'T-P 分布：三元 vs 二元（NaCl-H₂O）'])

for col, bin_name in enumerate(['CaCl2-H2O', 'NaCl-H2O'], start=1):
    # 三元数据点
    fig2.add_trace(go.Scatter(
        x=T3, y=P3, mode='markers',
        marker=dict(size=4, color='#1f77b4', opacity=0.4, symbol='circle'),
        name='三元数据', showlegend=(col==1),
    ), row=1, col=col)
    # 二元数据点
    if bin_name in binary_data:
        Xb = binary_data[bin_name]['X']
        fig2.add_trace(go.Scatter(
            x=Xb[:,0], y=Xb[:,1], mode='markers',
            marker=dict(size=7, color='#d62728', opacity=0.8,
                        symbol='star', line=dict(width=1, color='darkred')),
            name=f'二元/{bin_name}', showlegend=True,
        ), row=1, col=col)
    fig2.update_xaxes(title_text='T (K)', row=1, col=col)
    fig2.update_yaxes(title_text='P (Pa)', row=1, col=col)

fig2.update_layout(
    title=dict(text='T-P 空间覆盖对比（三元 vs 二元）<br>二元数据必须覆盖三元数据的T/P范围，边界约束才有效', x=0.5),
    width=1200, height=500)
fig2.write_html(str(RESULTS_DIR / 'TP_coverage_comparison.html'))
print("  ✓ TP_coverage_comparison.html")

# ── 图3：边界点直接对比（固定T/P，TC vs H₂O）──────────────────────────
fig3 = make_subplots(rows=1, cols=2,
    subplot_titles=['NaCl=0 边界：三元 vs CaCl₂-H₂O 二元',
                    'CaCl₂=0 边界：三元 vs NaCl-H₂O 二元'])

for col, (comp_arr, comp_name, bin_name, x_col_tern) in enumerate([
    (NaCl3,   'NaCl',  'CaCl2-H2O', 2),   # NaCl→0，x轴用H₂O
    (CaCl2_3, 'CaCl₂', 'NaCl-H2O',  3),   # CaCl₂→0，x轴用NaCl
], start=1):
    model = binary_models.get(bin_name)

    # 三元数据按 comp 分组着色
    fig3.add_trace(go.Scatter(
        x=all_tern.iloc[:, x_col_tern-1].values,
        y=TC3,
        mode='markers',
        marker=dict(size=4, color=comp_arr,
                    colorscale='RdYlGn_r', showscale=True,
                    colorbar=dict(title=comp_name,
                                  x=0.46 if col==1 else 1.02),
                    cmin=0, cmax=float(np.percentile(comp_arr, 95)),
                    opacity=0.6),
        name=f'三元数据（颜色={comp_name}浓度，越绿越接近边界）',
        customdata=np.column_stack([T3, P3, comp_arr]),
        hovertemplate=(f'H₂O/NaCl=%{{x:.5f}}  TC=%{{y:.5f}}<br>'
                       f'T=%{{customdata[0]:.1f}}K  P=%{{customdata[1]:.2e}}Pa<br>'
                       f'{comp_name}=%{{customdata[2]:.5f}}<extra>三元</extra>'),
        showlegend=True,
    ), row=1, col=col)

    # 二元数据真实点
    if bin_name in binary_data:
        Xb = binary_data[bin_name]['X']
        yb = binary_data[bin_name]['y']
        fig3.add_trace(go.Scatter(
            x=Xb[:,2], y=yb,
            mode='markers',
            marker=dict(size=8, color='red', symbol='star',
                        line=dict(width=1, color='darkred')),
            name=f'{bin_name} 真实数据',
            customdata=np.column_stack([Xb[:,0], Xb[:,1]]),
            hovertemplate=(f'H₂O=%{{x:.5f}}  TC=%{{y:.5f}}<br>'
                           f'T=%{{customdata[0]:.1f}}K  P=%{{customdata[1]:.2e}}Pa'
                           f'<extra>{bin_name}</extra>'),
        ), row=1, col=col)

    x_label = 'H₂O' if col==1 else 'NaCl (=1-H₂O)'
    fig3.update_xaxes(title_text=x_label, row=1, col=col)
    fig3.update_yaxes(title_text='TC (W/m·K)', row=1, col=col)

fig3.update_layout(
    title=dict(
        text=('边界对应关系：三元数据中 comp→0 的点 vs 二元真实数据<br>'
              '三元绿色点（comp≈0）应与二元红色星点在同一TC范围内'),
        x=0.5),
    width=1300, height=520)
fig3.write_html(str(RESULTS_DIR / 'boundary_correspondence.html'))
print("  ✓ boundary_correspondence.html")

# ── 图4：TC vs T 对比（三元边界点 vs 二元数据，固定相近H₂O）──────────
fig4 = make_subplots(rows=1, cols=2,
    subplot_titles=['TC vs T：三元(NaCl≈0) vs 二元(CaCl₂-H₂O)',
                    'TC vs T：三元(CaCl₂≈0) vs 二元(NaCl-H₂O)'])

for col, (comp_arr, bin_name) in enumerate([
    (NaCl3,   'CaCl2-H2O'),
    (CaCl2_3, 'NaCl-H2O'),
], start=1):
    model = binary_models.get(bin_name)
    # 三元数据按 comp 升序排列，取最小的10%作为"边界附近"
    q10 = np.percentile(comp_arr, 10)
    mask_b = comp_arr <= q10

    if mask_b.sum() > 0:
        fig4.add_trace(go.Scatter(
            x=T3[mask_b], y=TC3[mask_b], mode='markers',
            marker=dict(size=6, color=P3[mask_b], colorscale='Viridis',
                        showscale=(col==1),
                        colorbar=dict(title='P(Pa)') if col==1 else None,
                        symbol='circle'),
            name=f'三元边界点({["NaCl","CaCl₂"][col-1]}≤{q10:.4f})',
            customdata=np.column_stack([P3[mask_b], comp_arr[mask_b]]),
            hovertemplate=(f'T=%{{x:.1f}}K  TC=%{{y:.5f}}<br>'
                           f'P=%{{customdata[0]:.2e}}Pa<br>'
                           f'comp=%{{customdata[1]:.5f}}<extra>三元边界</extra>'),
        ), row=1, col=col)

    # 二元真实数据
    if bin_name in binary_data:
        Xb = binary_data[bin_name]['X']
        yb = binary_data[bin_name]['y']
        fig4.add_trace(go.Scatter(
            x=Xb[:,0], y=yb, mode='markers',
            marker=dict(size=8, color='red', symbol='star',
                        line=dict(width=1, color='darkred')),
            name=f'{bin_name} 真实数据',
            customdata=np.column_stack([Xb[:,1], Xb[:,2]]),
            hovertemplate=(f'T=%{{x:.1f}}K  TC=%{{y:.5f}}<br>'
                           f'P=%{{customdata[0]:.2e}}Pa  H₂O=%{{customdata[1]:.5f}}'
                           f'<extra>{bin_name}</extra>'),
        ), row=1, col=col)

    # 二元模型预测曲线（几个固定P）
    if model is not None:
        T_scan = np.linspace(T3.min(), T3.max(), 80)
        for P_fix in [P3.min(), float(np.median(P3)), P3.max()]:
            H2O_fix = float(np.median(
                H2O3[comp_arr <= q10] if mask_b.sum() > 0 else [H2O3.mean()]
            ))
            X_s = np.column_stack([T_scan, np.full(80, P_fix),
                                    np.full(80, H2O_fix)]).astype(np.float32)
            TC_s = model(X_s).flatten()
            fig4.add_trace(go.Scatter(
                x=T_scan, y=TC_s, mode='lines',
                line=dict(width=1.5, dash='dash'),
                name=f'{bin_name} 预测(P={P_fix:.1e},H₂O={H2O_fix:.4f})',
                showlegend=True,
            ), row=1, col=col)

    fig4.update_xaxes(title_text='T (K)', row=1, col=col)
    fig4.update_yaxes(title_text='TC (W/m·K)', row=1, col=col)

fig4.update_layout(
    title=dict(
        text=('TC vs T：三元边界点（实心圆）vs 二元真实数据（红星）vs 二元模型（虚线）<br>'
              '三者TC范围应高度重叠，否则边界约束方向错误'),
        x=0.5),
    width=1300, height=540)
fig4.write_html(str(RESULTS_DIR / 'TC_vs_T_boundary_check.html'))
print("  ✓ TC_vs_T_boundary_check.html")

# ── 图5：H₂O 范围对比（三元 vs 二元，NaCl=0 侧）──────────────────────
fig5 = make_subplots(rows=1, cols=2,
    subplot_titles=['H₂O 分布：三元 vs CaCl₂-H₂O 二元',
                    'H₂O 分布：三元 vs NaCl-H₂O 二元'])

for col, bin_name in enumerate(['CaCl2-H2O', 'NaCl-H2O'], start=1):
    fig5.add_trace(go.Histogram(
        x=H2O3, nbinsx=30,
        marker_color='#1f77b4', opacity=0.6,
        name='三元 H₂O',
    ), row=1, col=col)
    if bin_name in binary_data:
        Xb = binary_data[bin_name]['X']
        fig5.add_trace(go.Histogram(
            x=Xb[:,2], nbinsx=20,
            marker_color='#d62728', opacity=0.6,
            name=f'{bin_name} H₂O',
        ), row=1, col=col)
    fig5.update_xaxes(title_text='H₂O', row=1, col=col)
    fig5.update_yaxes(title_text='频次', row=1, col=col)

fig5.update_layout(
    barmode='overlay',
    title=dict(
        text=('H₂O 分布对比：三元数据（蓝）vs 二元数据（红）<br>'
              '两者范围应重叠，若三元 H₂O 超出二元训练范围则域外外推'),
        x=0.5),
    width=1200, height=480)
fig5.write_html(str(RESULTS_DIR / 'H2O_distribution_comparison.html'))
print("  ✓ H2O_distribution_comparison.html")

# ── 图6：模型在三元边界点上的预测精度详细散点 ──────────────────────────
fig6 = make_subplots(rows=1, cols=2,
    subplot_titles=['nacl_zero 边界：TC 二元模型预测 vs 三元真实',
                    'cacl2_zero 边界：TC 二元模型预测 vs 三元真实'])

for col, (comp_arr, bin_name, label) in enumerate([
    (NaCl3,   'CaCl2-H2O', 'NaCl≈0'),
    (CaCl2_3, 'NaCl-H2O',  'CaCl₂≈0'),
], start=1):
    model = binary_models.get(bin_name)
    if model is None: continue

    # 所有三元点都用二元模型预测（不只边界点）
    X_all = np.column_stack([T3, P3, H2O3]).astype(np.float32)
    TC_bin_all = model(X_all).flatten()
    diff = TC3 - TC_bin_all

    fig6.add_trace(go.Scatter(
        x=TC3, y=TC_bin_all, mode='markers',
        marker=dict(size=4, color=comp_arr,
                    colorscale='RdYlGn_r', showscale=True,
                    colorbar=dict(title=f'{label.split("≈")[0]}浓度',
                                  x=0.46 if col==1 else 1.02),
                    cmin=0, cmax=float(np.percentile(comp_arr, 90))),
        customdata=np.column_stack([T3, P3, comp_arr, diff]),
        hovertemplate=(f'TC_ternary=%{{x:.5f}}<br>TC_binary_pred=%{{y:.5f}}<br>'
                       f'T=%{{customdata[0]:.1f}}K  P=%{{customdata[1]:.2e}}Pa<br>'
                       f'{label.split("≈")[0]}=%{{customdata[2]:.5f}}<br>'
                       f'差值=%{{customdata[3]:+.5f}}<extra></extra>'),
        name=f'{bin_name} 预测 vs 三元真实',
    ), row=1, col=col)

    v = [min(float(TC3.min()), float(TC_bin_all.min())),
         max(float(TC3.max()), float(TC_bin_all.max()))]
    fig6.add_trace(go.Scatter(
        x=v, y=v, mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='1:1线', showlegend=(col==1),
    ), row=1, col=col)

    r2_all  = float(r2_score(TC3, TC_bin_all))
    fig6.update_xaxes(title_text=f'TC 三元真实  R²={r2_all:.4f}', row=1, col=col)
    fig6.update_yaxes(title_text='TC 二元模型预测', row=1, col=col)

fig6.update_layout(
    title=dict(
        text=('二元模型在三元数据上的预测精度（颜色=盐浓度，越绿=越接近边界）<br>'
              '绿色点应落在1:1线附近，否则边界约束方向或数据有问题'),
        x=0.5),
    width=1300, height=540)
fig6.write_html(str(RESULTS_DIR / 'binary_model_on_ternary_data.html'))
print("  ✓ binary_model_on_ternary_data.html")

# =============================================================================
# 5. 汇总报告
# =============================================================================
print(f"\n── 5. 汇总报告 ────────────────────────────────────────────────────")
print(f"\n  三元数据:")
print(f"    T  范围: [{T3.min():.1f}, {T3.max():.1f}] K")
print(f"    P  范围: [{P3.min():.2e}, {P3.max():.2e}] Pa")
print(f"    H₂O范围: [{H2O3.min():.5f}, {H2O3.max():.5f}]")
print(f"    NaCl范围: [{NaCl3.min():.5f}, {NaCl3.max():.5f}]")
print(f"    CaCl₂ = 1-H₂O-NaCl 范围: [{CaCl2_3.min():.5f}, {CaCl2_3.max():.5f}]")

for bin_name in ['CaCl2-H2O', 'NaCl-H2O']:
    if bin_name not in binary_data or bin_name not in binary_models: continue
    Xb    = binary_data[bin_name]['X']
    model = binary_models[bin_name]
    X_q   = np.column_stack([T3, P3, H2O3]).astype(np.float32)
    TC_bp = model(X_q).flatten()
    r2_g  = float(r2_score(TC3, TC_bp))
    print(f"\n  {bin_name} 模型在全部三元数据上: R²={r2_g:.4f}")
    print(f"    H₂O范围对比: 三元[{H2O3.min():.5f},{H2O3.max():.5f}]  "
          f"二元[{Xb[:,2].min():.5f},{Xb[:,2].max():.5f}]")
    h_overlap = (H2O3.min() >= Xb[:,2].min()-0.01) and (H2O3.max() <= Xb[:,2].max()+0.01)
    print(f"    H₂O 范围是否覆盖: {'✓' if h_overlap else '⚠ 三元超出二元训练范围！'}")

print(f"\n{sep}")
print(f"分析完成  HTML报告保存在: {RESULTS_DIR}")
print(f"{sep}")
print("""
重点查看顺序：
  1. TC_vs_T_boundary_check.html     ← 最关键：三元边界点 vs 二元真实数据是否同量级？
  2. boundary_correspondence.html    ← 三元绿色点（comp≈0）与二元红星是否重叠？
  3. TP_coverage_comparison.html     ← 二元数据T/P范围是否覆盖三元？
  4. H2O_distribution_comparison.html ← H₂O范围是否对齐？
  5. binary_model_on_ternary_data.html ← 二元模型在全三元数据上精度如何？
  6. ternary_distribution.html       ← 三元数据整体分布
""")