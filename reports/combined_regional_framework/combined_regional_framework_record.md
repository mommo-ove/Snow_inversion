# Combined IceBridge + IMB SMOS Inversion Record

## Protocol

Samples are split into 90% modeling data and 10% independent point validation data. The 90% modeling data is internally split again for parameter tuning, then the tuned model is evaluated on the untouched 10% points. The key point validation compares the existing SMOS product and the trained model against point snow-depth truth.

## Best model parameters

{
  "min_samples_leaf": 2,
  "max_features": "sqrt"
}

## Inner tuning results

| min_samples_leaf | max_features | n_train | n_validation | rmse_m | mae_m  | bias_m  | r2      |
| ---------------- | ------------ | ------- | ------------ | ------ | ------ | ------- | ------- |
| 2                | sqrt         | 7102    | 3045         | 34.12  | 0.735  | -0.5201 | 0.1246  |
| 1                | sqrt         | 7102    | 3045         | 34.17  | 0.7185 | -0.536  | 0.1222  |
| 4                | sqrt         | 7102    | 3045         | 34.38  | 0.7755 | -0.4921 | 0.1115  |
| 8                | 0.5          | 7102    | 3045         | 34.44  | 0.8055 | -0.4722 | 0.1084  |
| 1                | 1            | 7102    | 3045         | 34.44  | 0.702  | -0.5656 | 0.1081  |
| 4                | 0.5          | 7102    | 3045         | 34.73  | 0.7845 | -0.4921 | 0.09339 |
| 8                | sqrt         | 7102    | 3045         | 34.79  | 0.8316 | -0.4635 | 0.09    |
| 2                | 0.5          | 7102    | 3045         | 34.89  | 0.7759 | -0.5016 | 0.08482 |
| 2                | 1            | 7102    | 3045         | 34.9   | 0.7588 | -0.5288 | 0.08418 |
| 1                | 0.5          | 7102    | 3045         | 34.99  | 0.7667 | -0.5133 | 0.07934 |
| 4                | 1            | 7102    | 3045         | 35.24  | 0.8104 | -0.4971 | 0.06618 |
| 8                | 1            | 7102    | 3045         | 35.59  | 0.8649 | -0.4637 | 0.04764 |

## 10% independent point validation

| method_id | method | subset    | n    | rmse_m | mae_m  | bias_m | r2      |
| --------- | ------ | --------- | ---- | ------ | ------ | ------ | ------- |
| model     | 本文模型   | all       | 1128 | 7.586  | 0.5837 | 0.564  | 0.03149 |
| model     | 本文模型   | icebridge | 2    | 62.67  | 61.74  | 61.74  | 0.4869  |
| model     | 本文模型   | imb       | 1126 | 7.119  | 0.475  | 0.4553 | -1639   |

Note: no matched SMOS product snow-depth column was found in the point samples, so this run only reports model-vs-point-truth metrics. Add a column such as `SMOS_Product_Snow_Depth_m` to enable the direct SMOS-product-vs-truth comparison.


## Regional visualization/product comparison

Regional inversion was not run.

## Model features

- `TB_18V`
- `TB_18H`
- `TB_23V`
- `TB_23H`
- `TB_36V`
- `TB_36H`
- `TB_89V`
- `TB_89H`
- `GR_36V_18V`
- `GR_89V_18V`
- `PR_18`
- `PR_23`
- `PR_36`
- `PR_89`
- `PM_LOW_FREQ_MEAN`
- `PM_HIGH_FREQ_MEAN`
- `PM_LOW_HIGH_DIFF`
- `PM_LOW_HIGH_RATIO`
- `PM_DEPTH_SENSITIVITY`
- `PM_SPECTRAL_SLOPE_V`
- `PM_SPECTRAL_SLOPE_H`
- `PM_POL_DIFF_LOW`
- `PM_POL_DIFF_HIGH`
- `PM_POL_DIFF_CHANGE`
- `PM_MEAN_PR`
- `PM_MEAN_GR`
