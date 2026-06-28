# Product Comparison Report

This report separates product comparison from model training. The point truth remains IceBridge/IMB snow depth. External products are compared only on common valid points and, where possible, on screened or grid-aggregated subsets.

## Why This Is Needed

A direct point-vs-grid comparison can exaggerate product errors because point observations and satellite/reanalysis products have different spatial supports. The table below therefore includes common-valid, distance-screened, sea-ice-screened, and approximate grid-aggregated comparisons.

## Metrics

| comparison              | product_id      | method_id       | method            |    n |   rmse_m |   mae_m |     bias_m |      r2 |
|:------------------------|:----------------|:----------------|:------------------|-----:|---------:|--------:|-----------:|--------:|
| common_valid_points     | amsr_smos       | model           | RF model          |  336 |  0.04749 | 0.02237 |  0.00251   |  0.9118 |
| common_valid_points     | amsr_smos       | amsr_smos       | AMSR/SMOS product |  336 |  0.1721  | 0.1085  |  0.03696   | -0.158  |
| grid_aggregated_0.25deg | amsr_smos       | model           | RF model          |   38 |  0.01929 | 0.00863 |  0.0004929 |  0.9543 |
| grid_aggregated_0.25deg | amsr_smos       | amsr_smos       | AMSR/SMOS product |   38 |  0.0977  | 0.08357 |  0.02154   | -0.1713 |
| common_valid_points     | ecco_icecovered | model           | RF model          | 1944 |  0.05228 | 0.02778 |  0.001517  |  0.8596 |
| common_valid_points     | ecco_icecovered | ecco_icecovered | ECCO ice-covered  | 1944 |  0.1749  | 0.1265  |  0.004679  | -0.5714 |
| grid_aggregated_0.5deg  | ecco_icecovered | model           | RF model          |  237 |  0.03286 | 0.01426 |  0.003677  |  0.9334 |
| grid_aggregated_0.5deg  | ecco_icecovered | ecco_icecovered | ECCO ice-covered  |  237 |  0.1753  | 0.1259  | -0.03538   | -0.8949 |
| common_valid_points     | ecco_areaavg    | model           | RF model          | 1979 |  0.05215 | 0.0276  |  0.001703  |  0.859  |
| common_valid_points     | ecco_areaavg    | ecco_areaavg    | ECCO area-average | 1979 |  0.1712  | 0.1229  | -0.005615  | -0.5199 |
| grid_aggregated_0.5deg  | ecco_areaavg    | model           | RF model          |  240 |  0.0327  | 0.0142  |  0.003751  |  0.9336 |
| grid_aggregated_0.5deg  | ecco_areaavg    | ecco_areaavg    | ECCO area-average |  240 |  0.1729  | 0.1238  | -0.04245   | -0.8563 |

## Interpretation Rule

Use `common_valid_points` as the main point-level comparison. Use distance and SIC subsets as diagnostic checks. Use grid-aggregated rows as a fairer product-scale comparison, but report the number of aggregated cells because it may be much smaller than the point count.

Aggregated product-scale rows: 515
