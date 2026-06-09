# 机载 + IMB 点位验证与 SMOS 面反演框架

## 核心逻辑

这版主线不是一开始就直接把模型推到整个北极面，而是先在独立点位上证明：

```text
本文模型反演雪深 vs 点位真实雪深
优于
SMOS 自带雪深产品 vs 点位真实雪深
```

点位真实雪深来自 IceBridge 机载观测和 IMB 浮标观测。由于这两类数据本身都是点或轨迹样本，无法直接形成完整北极面；因此在点位验证成立后，再把训练好的模型应用到其他区域的 SMOS 亮温格网，生成面状雪深反演图，用作空间可视化展示。

## 数据划分

全部点样本先划分为：

- 90%：建模数据
- 10%：独立验证数据，作为最终点位真值对比，不参与训练和调参

90% 建模数据内部再划分，例如：

- 70%：内部训练
- 30%：内部预测/验证

这一步用于调参、筛选特征、降低误差。调优完成后，用完整 90% 建模数据训练模型，再拿 10% 独立验证数据做最终点位对比。

## 10% 独立点位上的两个对比

对每一个 10% 独立验证点，保留其真实雪深 `Snow_Depth_m`。

第一组对比：

```text
对应日期、对应位置的 SMOS 自带雪深产品值
vs
点位真实雪深
```

第二组对比：

```text
对应日期、对应位置的 SMOS 亮温
输入本文训练好的模型得到的雪深
vs
点位真实雪深
```

最终比较两组指标：

- RMSE
- MAE
- Bias
- R2
- 散点图：预测/产品雪深 vs 点位真实雪深

如果本文模型在 10% 独立点位上的 RMSE、MAE 更低，Bias 更合理，就可以说明：在这些独立点位样本上，本文模型相较 SMOS 自带雪深产品具有更好的反演精度。

## 区域面反演

完成点位验证后，再进入应用阶段：

1. 读取其他区域或其他日期的 SMOS 亮温格网。
2. 构建与训练阶段完全一致的亮温特征。
3. 将每个格点输入最终模型。
4. 输出整个北极或目标区域的面状雪深反演图。
5. 与 SMOS 自带雪深产品做空间分布对比和差值图。

这一部分主要用于展示模型推广到空间连续格网后的整体分布效果；论文或汇报中的精度证明重点应放在前面的 10% 独立点位对比。

## 当前脚本

脚本：

```text
random_forest/run_combined_regional_framework.py
```

本地或服务器先跑点位验证：

```powershell
python random_forest\run_combined_regional_framework.py --skip-regional
```

完整运行，包括区域面反演：

```powershell
python random_forest\run_combined_regional_framework.py
```

输出目录：

```text
reports/combined_regional_framework/
```

主要输出：

- `inner_70_30_tuning_metrics.csv`：90% 建模数据内部调参结果
- `holdout_10pct_point_comparison.csv`：10% 独立点位逐点对比
- `holdout_10pct_point_metrics.csv`：SMOS 产品与本文模型的点位验证指标
- `holdout_10pct_point_scatter.png`：点位散点图
- `holdout_10pct_metric_comparison.png`：模型与 SMOS 产品误差柱状图
- `combined_regional_model.joblib`：最终区域应用模型
- `day_*/regional_reference_retrieval_error.png`：区域反演与 SMOS 产品对比图
- `six_day_mean_reference_retrieval_error.png`：多日平均区域对比图

## 关于 SMOS 产品点位列

脚本会自动识别点样本表中的 SMOS 产品雪深列，支持的列名包括：

- `SMOS_Product_Snow_Depth_m`
- `SMOS_Snow_Depth_m`
- `monthsnow`
- `Product_Snow_Depth_m`
- `Reference_Snow_Depth_m`

如果当前点样本表还没有这个列，脚本仍会完成“本文模型 vs 点位真实雪深”的验证，但无法输出“SMOS 产品 vs 点位真实雪深”的那一组指标。服务器上如果已经有对应天、对应点匹配好的 SMOS 产品值，只需要把列名整理成上述任意一种即可。

## 服务器运行建议

服务器上拉取代码后，先安装依赖：

```bash
pip install -r requirements.txt
```

先跑轻量版本检查流程：

```bash
python random_forest/run_combined_regional_framework.py --skip-regional
```

再跑完整版本，并按服务器上的实际路径传入 SMOS 区域文件：

```bash
python random_forest/run_combined_regional_framework.py \
  --regional-tb-mat /path/to/2017_02_fivedays_corrected.mat \
  --mask-mat /path/to/common_snow_mask.mat \
  --reference-dir /path/to/actual_plots_with_mask
```

## 汇报表述

可以这样说：

> 本研究首先将 IceBridge 机载观测和 IMB 浮标观测合并为点位监督样本，采用 90% 数据进行模型训练与调优，并保留 10% 独立点位样本进行最终验证。在独立验证点上，分别提取 SMOS 自带雪深产品值和本文模型基于 SMOS 亮温反演得到的雪深，并与点位真实雪深进行对比。结果用于判断本文模型是否优于 SMOS 产品。随后，将优化后的模型应用于 SMOS 区域亮温格网，生成北极海冰雪深的空间连续反演结果，用于展示整体空间分布。
