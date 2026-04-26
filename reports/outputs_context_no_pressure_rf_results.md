# Random Forest Results: outputs_context_no_pressure

## Metrics

| experiment    |   n_test |   rmse_m |      mae_m |        r2 | held_out_buoys    |
|:--------------|---------:|---------:|-----------:|----------:|:------------------|
| random_split  |     2252 | 0.018172 | 0.00660418 |  0.987606 | nan               |
| buoy_held_out |      931 | 0.178403 | 0.167744   | -2.78817  | 2014B,2015E,2015I |

## Figures

- random_split_observed_vs_predicted.png

![random_split_observed_vs_predicted](reports/figures/outputs_context_no_pressure/random_split_observed_vs_predicted.png)

- random_split_feature_importance.png

![random_split_feature_importance](reports/figures/outputs_context_no_pressure/random_split_feature_importance.png)

- buoy_held_out_observed_vs_predicted.png

![buoy_held_out_observed_vs_predicted](reports/figures/outputs_context_no_pressure/buoy_held_out_observed_vs_predicted.png)

- buoy_held_out_feature_importance.png

![buoy_held_out_feature_importance](reports/figures/outputs_context_no_pressure/buoy_held_out_feature_importance.png)

## Notes

`random_split` is an 80/20 random split. `buoy_held_out` holds out complete buoys from training.
