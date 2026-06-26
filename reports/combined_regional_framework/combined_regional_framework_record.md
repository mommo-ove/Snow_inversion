# Combined IceBridge + IMB SMOS Inversion Record

## Protocol

Samples are split into 90% modeling data and 10% independent point validation data. The 90% modeling data is internally split again for parameter tuning, then the tuned model is evaluated on the untouched 10% points. The key point validation compares the trained model and any matched product columns against point snow-depth truth.

## Best model parameters

{
  "min_samples_leaf": 1,
  "max_features": 0.5
}

## Inner tuning results

| min_samples_leaf | max_features | n_train | n_validation | rmse_m  | mae_m   | bias_m    | r2     |
| ---------------- | ------------ | ------- | ------------ | ------- | ------- | --------- | ------ |
| 1                | 0.5          | 12555   | 5381         | 0.06044 | 0.0314  | 0.001706  | 0.8204 |
| 1                | sqrt         | 12555   | 5381         | 0.06086 | 0.03178 | 0.001678  | 0.8179 |
| 1                | 1            | 12555   | 5381         | 0.06358 | 0.03286 | 0.0009984 | 0.8013 |
| 2                | 0.5          | 12555   | 5381         | 0.06462 | 0.03376 | 0.001478  | 0.7947 |
| 2                | sqrt         | 12555   | 5381         | 0.06576 | 0.03477 | 0.001314  | 0.7874 |
| 4                | 0.5          | 12555   | 5381         | 0.07112 | 0.03794 | 0.001061  | 0.7513 |
| 2                | 1            | 12555   | 5381         | 0.07151 | 0.03854 | 0.0007524 | 0.7486 |
| 4                | sqrt         | 12555   | 5381         | 0.07408 | 0.04006 | 0.0009768 | 0.7302 |
| 8                | 0.5          | 12555   | 5381         | 0.08067 | 0.04445 | 0.0008153 | 0.6801 |
| 4                | 1            | 12555   | 5381         | 0.0829  | 0.04684 | 0.0007526 | 0.6622 |
| 8                | sqrt         | 12555   | 5381         | 0.08428 | 0.04719 | 0.0008102 | 0.6508 |
| 8                | 1            | 12555   | 5381         | 0.09452 | 0.05504 | 0.0003758 | 0.5608 |

## 10% independent point validation

| method_id       | method            | subset                            | n    | rmse_m  | mae_m   | bias_m    | r2      |
| --------------- | ----------------- | --------------------------------- | ---- | ------- | ------- | --------- | ------- |
| model           | RF model          | all                               | 1993 | 0.05223 | 0.02778 | 0.001647  | 0.8581  |
| model           | RF model          | icebridge                         | 867  | 0.06831 | 0.0488  | 0.002209  | 0.6193  |
| model           | RF model          | imb                               | 1126 | 0.03514 | 0.01159 | 0.001214  | 0.9489  |
| smos_product    | AMSR/SMOS product | all                               | 336  | 0.1721  | 0.1085  | 0.03696   | -0.158  |
| smos_product    | AMSR/SMOS product | icebridge                         | 160  | 0.09132 | 0.07689 | 0.05887   | -0.9761 |
| smos_product    | AMSR/SMOS product | imb                               | 176  | 0.2213  | 0.1373  | 0.01703   | -0.1766 |
| ecco_icecovered | ECCO ice-covered  | all                               | 1944 | 0.1749  | 0.1265  | 0.004679  | -0.5714 |
| ecco_icecovered | ECCO ice-covered  | icebridge                         | 853  | 0.1358  | 0.1098  | 0.084     | -0.5028 |
| ecco_icecovered | ECCO ice-covered  | imb                               | 1091 | 0.2002  | 0.1396  | -0.05734  | -0.624  |
| ecco_areaavg    | ECCO area-average | all                               | 1979 | 0.1712  | 0.1229  | -0.005615 | -0.5199 |
| ecco_areaavg    | ECCO area-average | icebridge                         | 853  | 0.1281  | 0.1027  | 0.0736    | -0.3391 |
| ecco_areaavg    | ECCO area-average | imb                               | 1126 | 0.1977  | 0.1383  | -0.06563  | -0.6177 |
| model           | RF model          | smos_product_matched              | 336  | 0.04749 | 0.02237 | 0.00251   | 0.9118  |
| model           | RF model          | smos_product_matched_icebridge    | 160  | 0.05803 | 0.03535 | 0.009214  | 0.2022  |
| model           | RF model          | smos_product_matched_imb          | 176  | 0.03528 | 0.01056 | -0.003585 | 0.9701  |
| smos_product    | AMSR/SMOS product | smos_product_matched              | 336  | 0.1721  | 0.1085  | 0.03696   | -0.158  |
| smos_product    | AMSR/SMOS product | smos_product_matched_icebridge    | 160  | 0.09132 | 0.07689 | 0.05887   | -0.9761 |
| smos_product    | AMSR/SMOS product | smos_product_matched_imb          | 176  | 0.2213  | 0.1373  | 0.01703   | -0.1766 |
| model           | RF model          | ecco_icecovered_matched           | 1944 | 0.05228 | 0.02778 | 0.001517  | 0.8596  |
| model           | RF model          | ecco_icecovered_matched_icebridge | 853  | 0.0684  | 0.04873 | 0.002348  | 0.6184  |
| model           | RF model          | ecco_icecovered_matched_imb       | 1091 | 0.03481 | 0.0114  | 0.0008673 | 0.9509  |
| ecco_icecovered | ECCO ice-covered  | ecco_icecovered_matched           | 1944 | 0.1749  | 0.1265  | 0.004679  | -0.5714 |
| ecco_icecovered | ECCO ice-covered  | ecco_icecovered_matched_icebridge | 853  | 0.1358  | 0.1098  | 0.084     | -0.5028 |
| ecco_icecovered | ECCO ice-covered  | ecco_icecovered_matched_imb       | 1091 | 0.2002  | 0.1396  | -0.05734  | -0.624  |
| model           | RF model          | ecco_areaavg_matched              | 1979 | 0.05215 | 0.0276  | 0.001703  | 0.859   |
| model           | RF model          | ecco_areaavg_matched_icebridge    | 853  | 0.0684  | 0.04873 | 0.002348  | 0.6184  |
| model           | RF model          | ecco_areaavg_matched_imb          | 1126 | 0.03514 | 0.01159 | 0.001214  | 0.9489  |
| ecco_areaavg    | ECCO area-average | ecco_areaavg_matched              | 1979 | 0.1712  | 0.1229  | -0.005615 | -0.5199 |
| ecco_areaavg    | ECCO area-average | ecco_areaavg_matched_icebridge    | 853  | 0.1281  | 0.1027  | 0.0736    | -0.3391 |
| ecco_areaavg    | ECCO area-average | ecco_areaavg_matched_imb          | 1126 | 0.1977  | 0.1383  | -0.06563  | -0.6177 |

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
