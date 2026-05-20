# 被动微波厚度反演思想迁移说明

## 老师让看的点

文档里的被动反演不是直接用一个亮温值量出冰厚，而是利用不同频率微波对不同深度的敏感性不同：

```text
低频穿透相对更深
高频更偏表层
不同频率/极化的亮温差异能反映雪冰层的结构和厚度相关信息
```

所以被动反演厚度的基本思想可以粗略写成：

```text
多频亮温观测 -> 构造频率响应差异 -> 约束雪/冰厚度或内部参数
```

## 怎么迁移到当前数据

当前数据没有 P-L 波段，也没有探冰雷达回波，所以不能完整复现文档里的冰盖主被动联合反演。

但当前数据有 18/23/36/89 GHz 多频亮温，可以先做一个简化迁移：

```text
用低频和高频亮温差异，构造被动微波深度敏感性特征
再用这些特征预测 Snow_Depth_m
```

这不是完整物理模型，但比单纯把 8 个通道丢进随机森林更接近文档里“多频亮温深度响应”的思想。

## 新增特征

脚本新增了 `--include-passive-physics` 开关，会自动生成以下 `PM_` 特征：

```text
PM_LOW_FREQ_MEAN
PM_HIGH_FREQ_MEAN
PM_LOW_HIGH_DIFF
PM_LOW_HIGH_RATIO
PM_DEPTH_SENSITIVITY
PM_SPECTRAL_SLOPE_V
PM_SPECTRAL_SLOPE_H
PM_POL_DIFF_LOW
PM_POL_DIFF_HIGH
PM_POL_DIFF_CHANGE
PM_MEAN_PR
PM_MEAN_GR
```

这些特征的含义大概是：

```text
低频平均亮温：偏深层响应
高频平均亮温：偏浅层/雪层散射响应
低高频差异：表征频率穿透差异
谱斜率：亮温随频率变化的趋势
极化差异：表征表面和雪冰界面对 V/H 极化的影响
```

## 推荐跑法

基础版：

```powershell
python random_forest\run_rf_experiments.py --include-passive-physics --out-dir random_forest\outputs_passive_physics
```

更适合组会讲的版本：

```powershell
python random_forest\run_rf_experiments.py --include-passive-physics --include-context --exclude-columns Air_Pressure_mbar --leave-one-buoy-out --out-dir random_forest\outputs_passive_physics_context_no_pressure
```

## 组会可以怎么讲

可以说：

> 老师给的文档里，被动微波反演厚度的关键不是单个亮温值，而是多频亮温对不同深度的敏感性差异。我目前的数据没有 P-L 波段和主动雷达，所以不能完整复现文档里的主被动联合反演。但我先把这个思想迁移到现有 18-89 GHz 数据上，构造了低频/高频差异、谱斜率和极化差异等被动微波启发特征，用来做雪深反演。

再接一句：

> 这一步相当于是从纯统计特征往物理启发特征过渡，后面如果能拿到低频亮温或雷达回波，再继续往完整主被动联合反演靠。

