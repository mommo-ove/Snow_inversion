# Combined IceBridge + IMB SMOS Inversion Record

## Protocol

Samples are split into 90% modeling data and 10% independent point validation data. The 90% modeling data is internally split again for parameter tuning, then the tuned model is evaluated on the untouched 10% points. The key point validation compares the existing SMOS product and the trained model against point snow-depth truth.

## Best model parameters

{
  "min_samples_leaf": 1,
  "max_features": 0.5
}

## Inner tuning results

| min_samples_leaf | max_features | n_train | n_validation | rmse_m  | mae_m   | bias_m    | r2     |
| ---------------- | ------------ | ------- | ------------ | ------- | ------- | --------- | ------ |
| 1                | 0.5          | 12555   | 5381         | 0.06069 | 0.03155 | 0.001784  | 0.819  |
| 1                | sqrt         | 12555   | 5381         | 0.06106 | 0.03179 | 0.001494  | 0.8167 |
| 1                | 1            | 12555   | 5381         | 0.06358 | 0.03284 | 0.001069  | 0.8013 |
| 2                | 0.5          | 12555   | 5381         | 0.06455 | 0.03362 | 0.001231  | 0.7952 |
| 2                | sqrt         | 12555   | 5381         | 0.06551 | 0.03477 | 0.001294  | 0.7891 |
| 4                | 0.5          | 12555   | 5381         | 0.07115 | 0.03793 | 0.001051  | 0.7511 |
| 2                | 1            | 12555   | 5381         | 0.07157 | 0.0385  | 0.0007236 | 0.7482 |
| 4                | sqrt         | 12555   | 5381         | 0.07394 | 0.04001 | 0.0009778 | 0.7312 |
| 8                | 0.5          | 12555   | 5381         | 0.08069 | 0.04446 | 0.0008173 | 0.68   |
| 4                | 1            | 12555   | 5381         | 0.08289 | 0.04685 | 0.0007503 | 0.6622 |
| 8                | sqrt         | 12555   | 5381         | 0.08429 | 0.0472  | 0.0008085 | 0.6507 |
| 8                | 1            | 12555   | 5381         | 0.09452 | 0.05505 | 0.0003857 | 0.5608 |

## 10% independent point validation

| method_id | method | subset    | n    | rmse_m  | mae_m   | bias_m   | r2     |
| --------- | ------ | --------- | ---- | ------- | ------- | -------- | ------ |
| model     | 本文模型   | all       | 1993 | 0.05207 | 0.02764 | 0.001587 | 0.859  |
| model     | 本文模型   | icebridge | 867  | 0.06822 | 0.04861 | 0.00219  | 0.6204 |
| model     | 本文模型   | imb       | 1126 | 0.03486 | 0.01149 | 0.001122 | 0.9497 |

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
