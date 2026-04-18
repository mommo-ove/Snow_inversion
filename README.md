# Snow Inversion

Snow depth retrieval experiments for IMB buoy and IceBridge-style tabular data.

## Data layout

Put local data files under `data/`. The folder is ignored by Git so large or private datasets are not uploaded.

Expected files:

```text
data/Master_Dataset_AllFeatures.mat
data/Validation_Dataset_IMB_AllBuoys_Combined_AllData.csv
```

The current baseline predicts `Snow_Depth_m` from microwave brightness temperatures plus optional time, location, and meteorological variables.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Inspect data

```powershell
python random_forest\inspect_data.py
```

This prints the CSV columns, summary statistics, buoy counts, and the internal structure of the MATLAB v7.3 file using `h5py`.

## Run random forest experiments

```powershell
python random_forest\run_rf_experiments.py
```

Outputs are written to:

```text
random_forest/outputs/
```

The script also copies the key metrics and figures to `reports/`, which is meant for sharing through GitHub without uploading the raw data.

The script reports two validation settings:

```text
random_split
```

Random 80/20 train-test split. This is useful as a quick baseline, but can be optimistic for buoy time series.

```text
buoy_held_out
```

Group split by `Buoy`, so complete buoys are held out from training. This is a stricter test of generalization.

The old scripts are kept as compatibility entry points, but the recommended workflow is:

```powershell
python random_forest\inspect_data.py
python random_forest\run_rf_experiments.py
```

## Share results

After running experiments on the server, commit the generated report files instead of the ignored raw output folder:

```powershell
git add reports/
git commit -m "Add latest RF results"
git push
```

Then collaborators can inspect the metrics and selected figures from GitHub.
