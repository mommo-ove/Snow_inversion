from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils_data import DEFAULT_CSV, DEFAULT_MAT, TARGET_COL, TB_FEATURES, add_passive_microwave_features


ICEBRIDGE_COLUMNS = [
    "Year",
    "Month",
    "Day",
    "Latitude",
    "Longitude",
    "TB_18V",
    "TB_18H",
    "TB_23V",
    "TB_23H",
    "TB_36V",
    "TB_36H",
    "TB_89V",
    "TB_89H",
    "Ice_Thickness_m",
    "Mean_Freeboard_m",
    TARGET_COL,
    "Surface_Roughness",
    "KT19_Surface_Temp",
    "MY_Ice_Fraction",
]

PRODUCT_COL = "SMOS_Product_Snow_Depth_m"
PRODUCT_ALIASES = [
    PRODUCT_COL,
    "SMOS_Snow_Depth_m",
    "SMOS_snow_depth_m",
    "Smos_Snow_Depth_m",
    "monthsnow",
    "month_snow",
    "Reference_Snow_Depth_m",
    "Product_Snow_Depth_m",
    "Existing_SMOS_Snow_Depth_m",
]
EXTRA_PRODUCT_SPECS = [
    ("ecco_icecovered", "ECCO ice-covered", "ECCO_IceCovered_Snow_Depth_m"),
    ("ecco_areaavg", "ECCO area-average", "ECCO_AreaAvg_Snow_Depth_m"),
]


def import_mat_tools():
    from scipy.io import loadmat, savemat

    return loadmat, savemat


def import_ml_tools():
    from joblib import dump
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    return {
        "dump": dump,
        "RandomForestRegressor": RandomForestRegressor,
        "mean_absolute_error": mean_absolute_error,
        "r2_score": r2_score,
        "train_test_split": train_test_split,
    }


def import_plotting():
    import matplotlib.pyplot as plt

    return plt


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def parse_max_features(value: str):
    value = value.strip()
    if value in {"sqrt", "log2"}:
        return value
    if value.lower() in {"none", "null"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return value
    return int(parsed) if parsed >= 1 and parsed.is_integer() else parsed


def normalize_product_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PRODUCT_ALIASES:
        if col in out.columns:
            out[PRODUCT_COL] = out[col]
            return out
    out[PRODUCT_COL] = np.nan
    return out


def available_product_specs(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    specs = [("smos_product", "AMSR/SMOS product", PRODUCT_COL)]
    specs.extend(spec for spec in EXTRA_PRODUCT_SPECS if spec[2] in df.columns)
    return specs


def load_numeric_mat_variable(path: Path, variable_name: str, expected_min_cols: int | None = None) -> np.ndarray:
    loadmat, _ = import_mat_tools()
    try:
        mat = loadmat(path, squeeze_me=True)
        if variable_name not in mat:
            raise KeyError(f"{path} does not contain variable {variable_name}")
        data = np.asarray(mat[variable_name], dtype=float)
    except NotImplementedError as exc:
        try:
            import h5py
        except ImportError as import_exc:
            raise ImportError(
                f"{path} is a MATLAB v7.3/HDF5 file. Install h5py to read it: pip install h5py"
            ) from import_exc

        with h5py.File(path, "r") as handle:
            if variable_name not in handle:
                raise KeyError(f"{path} does not contain variable {variable_name}") from exc
            data = np.asarray(handle[variable_name], dtype=float)

    if data.ndim == 2 and expected_min_cols is not None:
        # MATLAB v7.3 numeric arrays are often exposed by h5py as transposed
        # relative to the MATLAB workspace shape.
        if data.shape[0] == expected_min_cols and data.shape[1] > expected_min_cols:
            data = data.T
        elif data.shape[1] < expected_min_cols and data.shape[0] >= expected_min_cols:
            data = data.T
    return data


def load_icebridge_mat(path: Path) -> pd.DataFrame:
    data = load_numeric_mat_variable(path, "Final_Data", expected_min_cols=len(ICEBRIDGE_COLUMNS))
    if data.ndim != 2 or data.shape[1] < len(ICEBRIDGE_COLUMNS):
        raise ValueError(
            f"Expected Final_Data with at least {len(ICEBRIDGE_COLUMNS)} columns, got {data.shape}"
        )

    df = pd.DataFrame(data[:, : len(ICEBRIDGE_COLUMNS)], columns=ICEBRIDGE_COLUMNS)
    df["source"] = "icebridge"
    df["Hour"] = np.nan
    df["Distance_Km"] = np.nan
    return normalize_product_column(df)


def load_imb_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = "imb"
    for col in ["Mean_Freeboard_m", "Surface_Roughness", "KT19_Surface_Temp", "MY_Ice_Fraction"]:
        if col not in df.columns:
            df[col] = np.nan
    return normalize_product_column(df)


def load_combined_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "source" not in df.columns:
        df["source"] = "unknown"
    return normalize_product_column(df)


def build_combined_training_table(icebridge_path: Path, imb_path: Path) -> pd.DataFrame:
    icebridge = load_icebridge_mat(icebridge_path)
    imb = load_imb_csv(imb_path)
    common_cols = [
        "source",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Latitude",
        "Longitude",
        "Distance_Km",
        *TB_FEATURES,
        TARGET_COL,
        PRODUCT_COL,
        "Ice_Thickness_m",
        "Mean_Freeboard_m",
        "Surface_Roughness",
        "KT19_Surface_Temp",
        "MY_Ice_Fraction",
    ]
    combined = pd.concat(
        [icebridge.reindex(columns=common_cols), imb.reindex(columns=common_cols)],
        ignore_index=True,
    )
    combined = combined.replace([np.inf, -np.inf], np.nan)
    return combined[combined[TARGET_COL] > 0].copy()


def feature_columns(df: pd.DataFrame) -> list[str]:
    enriched = add_passive_microwave_features(df)
    candidates = [*TB_FEATURES]
    candidates.extend(col for col in enriched.columns if col.startswith(("GR_", "PR_", "PM_")))
    # Main model uses only features that can also be built from gridded SMOS TB.
    return [col for col in dict.fromkeys(candidates) if col in enriched.columns]


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    enriched = add_passive_microwave_features(df)
    features = feature_columns(enriched)
    model_df = enriched.dropna(subset=[TARGET_COL, *features]).copy()
    return model_df[features], model_df[TARGET_COL], model_df, features


def make_model(random_state: int, n_estimators: int, min_samples_leaf: int, max_features):
    tools = import_ml_tools()
    return tools["RandomForestRegressor"](
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    tools = import_ml_tools()
    return {
        "rmse_m": rmse(y_true, y_pred),
        "mae_m": float(tools["mean_absolute_error"](y_true, y_pred)),
        "bias_m": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
        "r2": float(tools["r2_score"](y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def frame_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    rendered = df.copy()
    for col in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[col]):
            rendered[col] = rendered[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
        else:
            rendered[col] = rendered[col].map(lambda value: "" if pd.isna(value) else str(value))
    columns = [str(col) for col in rendered.columns]
    rows = rendered.values.tolist()
    widths = [max(len(col), *[len(str(row[idx])) for row in rows]) for idx, col in enumerate(columns)]
    header = "| " + " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def split_train_holdout(args: argparse.Namespace, x, y, meta):
    tools = import_ml_tools()
    stratify = meta["source"] if meta["source"].value_counts().min() >= 2 else None
    return tools["train_test_split"](
        x,
        y,
        meta,
        test_size=args.holdout_size,
        random_state=args.random_state,
        stratify=stratify,
    )


def tune_on_modeling_set(args: argparse.Namespace, x_model, y_model, meta_model) -> tuple[dict, pd.DataFrame]:
    tools = import_ml_tools()
    stratify = meta_model["source"] if meta_model["source"].value_counts().min() >= 2 else None
    x_train, x_val, y_train, y_val = tools["train_test_split"](
        x_model,
        y_model,
        test_size=args.inner_validation_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    leaf_values = [int(v) for v in args.tune_min_samples_leaf.split(",") if v.strip()]
    max_feature_values = [parse_max_features(v) for v in args.tune_max_features.split(",") if v.strip()]
    rows = []
    best = None
    for min_leaf in leaf_values:
        for max_features in max_feature_values:
            model = make_model(args.random_state, args.tuning_n_estimators, min_leaf, max_features)
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            metrics = evaluate_predictions(y_val, pred)
            row = {
                "min_samples_leaf": min_leaf,
                "max_features": str(max_features),
                "n_train": len(y_train),
                "n_validation": len(y_val),
                **metrics,
            }
            rows.append(row)
            if best is None or row["rmse_m"] < best["rmse_m"]:
                best = row

    tuning = pd.DataFrame(rows).sort_values("rmse_m")
    assert best is not None
    best_params = {
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": parse_max_features(str(best["max_features"])),
    }
    return best_params, tuning


def make_holdout_predictions(args: argparse.Namespace, combined: pd.DataFrame, out_dir: Path):
    tools = import_ml_tools()
    x, y, meta, features = make_xy(combined)
    x_model, x_holdout, y_model, y_holdout, meta_model, meta_holdout = split_train_holdout(args, x, y, meta)

    best_params, tuning = tune_on_modeling_set(args, x_model, y_model, meta_model)
    tuning.to_csv(out_dir / "inner_70_30_tuning_metrics.csv", index=False)

    holdout_model = make_model(
        args.random_state,
        args.n_estimators,
        best_params["min_samples_leaf"],
        best_params["max_features"],
    )
    holdout_model.fit(x_model, y_model)
    model_pred = holdout_model.predict(x_holdout)

    product_cols = [col for _, _, col in available_product_specs(meta_holdout) if col in meta_holdout.columns]
    meta_cols = ["source", "Year", "Month", "Day", "Hour", "Latitude", "Longitude", TARGET_COL, *product_cols]
    pred_df = meta_holdout[meta_cols].copy()
    pred_df["Model_Retrieved_Snow_Depth_m"] = model_pred
    pred_df["Model_Error_m"] = pred_df["Model_Retrieved_Snow_Depth_m"] - pred_df[TARGET_COL]
    for method_id, _, pred_col in available_product_specs(pred_df):
        if pred_col in pred_df.columns:
            pred_df[f"{method_id}_Error_m"] = pred_df[pred_col] - pred_df[TARGET_COL]
    pred_df.to_csv(out_dir / "holdout_10pct_point_comparison.csv", index=False)

    metrics = build_holdout_metric_table(pred_df)
    metrics.to_csv(out_dir / "holdout_10pct_point_metrics.csv", index=False)

    deployment_model = make_model(
        args.random_state,
        args.n_estimators,
        best_params["min_samples_leaf"],
        best_params["max_features"],
    )
    deployment_model.fit(x, y)
    tools["dump"](
        {
            "model": deployment_model,
            "features": features,
            "best_params": best_params,
            "args": {k: str(v) for k, v in vars(args).items()},
        },
        out_dir / "combined_regional_model.joblib",
    )

    save_holdout_scatter(pred_df, out_dir / "holdout_10pct_point_scatter.png")
    save_metric_comparison(metrics, out_dir / "holdout_10pct_metric_comparison.png")
    save_feature_importance(deployment_model, features, out_dir / "combined_feature_importance.csv")
    combined.to_csv(out_dir / "combined_training_table_preview.csv", index=False)
    return deployment_model, features, metrics, tuning, best_params


def build_holdout_metric_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    product_specs = available_product_specs(pred_df)
    method_specs = [("model", "RF model", "Model_Retrieved_Snow_Depth_m"), *product_specs]
    for method_id, method_name, pred_col in method_specs:
        if pred_col not in pred_df.columns:
            continue
        available = pred_df.dropna(subset=[pred_col, TARGET_COL])
        for subset, part in [("all", available), *available.groupby("source")]:
            if len(part) == 0:
                continue
            rows.append(
                {
                    "method_id": method_id,
                    "method": method_name,
                    "subset": subset,
                    "n": len(part),
                    **evaluate_predictions(part[TARGET_COL], part[pred_col]),
                }
            )

    for product_id, _, product_col in product_specs:
        if product_col not in pred_df.columns:
            continue
        common = pred_df.dropna(subset=[product_col, "Model_Retrieved_Snow_Depth_m", TARGET_COL])
        if len(common) > 0:
            common_specs = [
                ("model", "RF model", "Model_Retrieved_Snow_Depth_m"),
                next(spec for spec in product_specs if spec[0] == product_id),
            ]
            for method_id, method_name, pred_col in common_specs:
                for subset, part in [
                    (f"{product_id}_matched", common),
                    *[(f"{product_id}_matched_{k}", v) for k, v in common.groupby("source")],
                ]:
                    if len(part) == 0:
                        continue
                    rows.append(
                        {
                            "method_id": method_id,
                            "method": method_name,
                            "subset": subset,
                            "n": len(part),
                            **evaluate_predictions(part[TARGET_COL], part[pred_col]),
                        }
                    )
    return pd.DataFrame(rows)


def save_holdout_scatter(pred_df: pd.DataFrame, fig_path: Path) -> None:
    plt = import_plotting()
    panels = [("RF model", "Model_Retrieved_Snow_Depth_m")]
    panels.extend(
        (name, col)
        for _, name, col in available_product_specs(pred_df)
        if col in pred_df.columns and pred_df[col].notna().any()
    )
    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(5.8 * ncols, 5.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for ax, (title, pred_col) in zip(axes, panels):
        part = pred_df.dropna(subset=[pred_col, TARGET_COL])
        if part.empty:
            ax.text(0.5, 0.5, "No matched data", ha="center", va="center")
            ax.axis("off")
            continue
        colors = part["source"].astype("category").cat.codes
        ax.scatter(part[TARGET_COL], part[pred_col], c=colors, s=13, alpha=0.65)
        lo = float(min(part[TARGET_COL].min(), part[pred_col].min()))
        hi = float(max(part[TARGET_COL].max(), part[pred_col].max()))
        ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Point observed snow depth (m)")
        ax.set_ylabel("Retrieved / product snow depth (m)")
        ax.set_title(f"{title} vs point truth")
        ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def save_metric_comparison(metrics: pd.DataFrame, fig_path: Path) -> None:
    if metrics.empty:
        return
    plt = import_plotting()
    plot_df = metrics[metrics["subset"] == "all"].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    x = np.arange(len(plot_df))
    width = 0.35
    ax.bar(x - width / 2, plot_df["rmse_m"], width, label="RMSE")
    ax.bar(x + width / 2, plot_df["mae_m"], width, label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["method"], rotation=15, ha="right")
    ax.set_ylabel("Error (m)")
    ax.set_title("10% independent point validation")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def save_feature_importance(model, features: list[str], csv_path: Path) -> None:
    if hasattr(model, "feature_importances_"):
        pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        ).to_csv(csv_path, index=False)


def matlab_cell_to_list(value) -> list:
    return [item for item in np.asarray(value, dtype=object).ravel()]


def mean_three(layers: list[np.ndarray], start: int) -> np.ndarray:
    return (np.asarray(layers[start]) + np.asarray(layers[start + 1]) + np.asarray(layers[start + 2])) / 3.0


def build_feature_maps_from_regional_tb(day_cell) -> dict[str, np.ndarray]:
    layers = [np.asarray(layer, dtype=float) for layer in matlab_cell_to_list(day_cell)]
    if len(layers) < 24:
        raise ValueError(f"Expected 24 TB layers for a day, got {len(layers)}")

    maps = {
        "TB_18H": mean_three(layers, 0),
        "TB_18V": mean_three(layers, 3),
        "TB_23H": mean_three(layers, 6),
        "TB_23V": mean_three(layers, 9),
        "TB_36H": mean_three(layers, 12),
        "TB_36V": mean_three(layers, 15),
        "TB_89H": mean_three(layers, 18),
        "TB_89V": mean_three(layers, 21),
    }
    tb_df = pd.DataFrame({name: grid.ravel() for name, grid in maps.items()})
    enriched = add_passive_microwave_features(tb_df)
    for col in enriched.columns:
        if col not in maps:
            maps[col] = enriched[col].to_numpy().reshape(maps["TB_18H"].shape)
    return maps


def maps_to_feature_matrix(feature_maps: dict[str, np.ndarray], features: list[str], base_mask: np.ndarray):
    shape = np.asarray(base_mask).shape
    valid = np.asarray(base_mask, dtype=bool).copy()
    flat_cols = []
    for feature in features:
        if feature not in feature_maps:
            raise KeyError(f"Regional TB maps cannot build required feature: {feature}")
        grid = np.asarray(feature_maps[feature], dtype=float)
        valid &= np.isfinite(grid)
        flat_cols.append(grid.ravel())
    for feature in TB_FEATURES:
        valid &= np.asarray(feature_maps[feature], dtype=float) > 50
    x_map = pd.DataFrame(np.vstack(flat_cols).T, columns=features)
    return x_map.loc[valid.ravel()], valid.reshape(shape)


def apply_regional_model(args: argparse.Namespace, model, features: list[str], out_dir: Path) -> pd.DataFrame:
    loadmat, savemat = import_mat_tools()
    regional = loadmat(args.regional_tb_mat, squeeze_me=True)
    mask_mat = loadmat(args.mask_mat, squeeze_me=True)
    if "data" not in regional:
        raise KeyError(f"{args.regional_tb_mat} does not contain variable data")
    if "land_mask_base_60x720" not in mask_mat:
        raise KeyError(f"{args.mask_mat} does not contain variable land_mask_base_60x720")

    days = matlab_cell_to_list(regional["data"])
    base_mask = np.asarray(mask_mat["land_mask_base_60x720"], dtype=bool)
    output_mask = np.flipud(base_mask) if args.flipud_output else base_mask

    rows = []
    pred_stack = []
    ref_stack = []
    for day_idx, day_cell in enumerate(days, start=1):
        feature_maps = build_feature_maps_from_regional_tb(day_cell)
        x_map, valid_mask_raw = maps_to_feature_matrix(feature_maps, features, base_mask)
        pred_raw = np.full(base_mask.shape, np.nan)
        pred_raw[valid_mask_raw] = model.predict(x_map)
        pred_raw = np.clip(pred_raw, 0.0, args.max_snow_depth)

        pred = np.flipud(pred_raw) if args.flipud_output else pred_raw
        pred[~output_mask] = np.nan

        ref = load_reference_product(args.reference_dir, day_idx)
        if ref is not None:
            ref = np.asarray(ref, dtype=float)
            ref[~output_mask] = np.nan

        day_dir = out_dir / f"day_{day_idx}"
        day_dir.mkdir(parents=True, exist_ok=True)
        savemat(day_dir / "combined_regional_retrieval.mat", {"pred": pred, "ref": ref})
        save_regional_panels(ref, pred, day_dir / "regional_reference_retrieval_error.png", day_idx)

        rows.append(make_regional_metric_row(day_idx, ref, pred))
        pred_stack.append(pred)
        if ref is not None:
            ref_stack.append(ref)

    if pred_stack:
        pred_mean = np.nanmean(np.stack(pred_stack), axis=0)
        ref_mean = np.nanmean(np.stack(ref_stack), axis=0) if ref_stack else None
        save_regional_panels(ref_mean, pred_mean, out_dir / "six_day_mean_reference_retrieval_error.png", 0)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "regional_product_comparison_metrics.csv", index=False)
    return metrics


def load_reference_product(reference_dir: str, day_idx: int):
    if not reference_dir:
        return None
    loadmat, _ = import_mat_tools()
    path = Path(reference_dir) / f"day_{day_idx}_actual_masked.mat"
    if not path.exists():
        return None
    return loadmat(path, squeeze_me=True).get("actual_snow_depth")


def make_regional_metric_row(day_idx: int, ref, pred) -> dict[str, float | int]:
    row = {"day": day_idx, "valid_retrieval_pixels": int(np.isfinite(pred).sum())}
    if ref is None:
        return row
    valid = np.isfinite(ref) & np.isfinite(pred)
    row["comparison_pixels"] = int(valid.sum())
    if valid.any():
        err = pred[valid] - ref[valid]
        row.update(
            {
                "reference_mean_m": float(np.mean(ref[valid])),
                "retrieval_mean_m": float(np.mean(pred[valid])),
                "rmse_m": float(np.sqrt(np.mean(err**2))),
                "mae_m": float(np.mean(np.abs(err))),
                "bias_m": float(np.mean(err)),
                "correlation": float(np.corrcoef(ref[valid], pred[valid])[0, 1]),
            }
        )
    return row


def save_regional_panels(ref, pred, fig_path: Path, day_idx: int) -> None:
    plt = import_plotting()
    err = pred - ref if ref is not None else np.full_like(pred, np.nan)
    valid_snow = pred[np.isfinite(pred)]
    if ref is not None:
        valid_snow = np.concatenate([valid_snow, ref[np.isfinite(ref)]])
    snow_max = max(float(np.nanpercentile(valid_snow, 98)) if valid_snow.size else 0.5, 0.15)
    err_valid = err[np.isfinite(err)]
    err_lim = max(float(np.nanpercentile(np.abs(err_valid), 98)) if err_valid.size else 0.2, 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = [
        ("SMOS snow product (m)", ref, "viridis", 0, snow_max),
        ("Model regional retrieval (m)", pred, "viridis", 0, snow_max),
        ("Model - product (m)", err, "coolwarm", -err_lim, err_lim),
    ]
    for ax, (title, grid, cmap, vmin, vmax) in zip(axes, panels):
        if grid is None:
            ax.text(0.5, 0.5, "No product map", ha="center", va="center")
            ax.axis("off")
            continue
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Six-day mean regional inversion" if day_idx == 0 else f"Day {day_idx} regional inversion")
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def write_run_summary(
    out_dir: Path,
    metrics: pd.DataFrame,
    tuning: pd.DataFrame,
    regional_metrics: pd.DataFrame | None,
    features: list[str],
    best_params: dict,
) -> None:
    has_product_metrics = any(method_id != "model" for method_id in set(metrics.get("method_id", [])))
    report_path = out_dir / "combined_regional_framework_record.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Combined IceBridge + IMB SMOS Inversion Record\n\n")
        handle.write("## Protocol\n\n")
        handle.write(
            "Samples are split into 90% modeling data and 10% independent point validation data. "
            "The 90% modeling data is internally split again for parameter tuning, then the tuned model is evaluated on the untouched 10% points. "
            "The key point validation compares the trained model and any matched product columns against point snow-depth truth.\n\n"
        )
        handle.write("## Best model parameters\n\n")
        handle.write(json.dumps(best_params, ensure_ascii=False, indent=2))
        handle.write("\n\n## Inner tuning results\n\n")
        handle.write(frame_to_markdown(tuning.head(20)))
        handle.write("\n\n## 10% independent point validation\n\n")
        handle.write(frame_to_markdown(metrics))
        if not has_product_metrics:
            handle.write(
                "\n\nNote: no matched product snow-depth columns were found in the point samples, "
                "so this run only reports model-vs-point-truth metrics. Add columns such as "
                f"`{PRODUCT_COL}` or `ECCO_IceCovered_Snow_Depth_m` to enable product-vs-truth comparisons.\n"
            )
        handle.write("\n\n## Regional visualization/product comparison\n\n")
        if regional_metrics is None or regional_metrics.empty:
            handle.write("Regional inversion was not run.\n\n")
        else:
            handle.write(frame_to_markdown(regional_metrics))
            handle.write("\n\n")
        handle.write("## Model features\n\n")
        handle.write("\n".join(f"- `{feature}`" for feature in features))
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined IceBridge+IMB training, 10% point validation against SMOS product/model, and regional SMOS TB inversion."
    )
    parser.add_argument("--icebridge-mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--imb-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=None,
        help="Optional prebuilt combined point table, for example one augmented with SMOS product values.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/combined_regional_framework"))
    parser.add_argument("--holdout-size", type=float, default=0.10)
    parser.add_argument("--inner-validation-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--tuning-n-estimators", type=int, default=250)
    parser.add_argument("--tune-min-samples-leaf", default="1,2,4,8")
    parser.add_argument("--tune-max-features", default="sqrt,0.5,1.0")
    parser.add_argument("--regional-tb-mat", default=r"C:\Users\lsl\Desktop\2017_02_fivedays_corrected.mat")
    parser.add_argument("--mask-mat", default=r"C:\Users\lsl\Desktop\common_snow_mask.mat")
    parser.add_argument("--reference-dir", default=r"C:\Users\lsl\Desktop\actual_plots_with_mask")
    parser.add_argument("--skip-regional", action="store_true")
    parser.add_argument("--flipud-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-snow-depth", type=float, default=1.2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({k: str(v) for k, v in vars(args).items()}, handle, indent=2, ensure_ascii=False)

    combined = load_combined_csv(args.combined_csv) if args.combined_csv else build_combined_training_table(args.icebridge_mat, args.imb_csv)
    model, features, metrics, tuning, best_params = make_holdout_predictions(args, combined, args.out_dir)

    regional_metrics = None
    if not args.skip_regional:
        regional_metrics = apply_regional_model(args, model, features, args.out_dir)

    write_run_summary(args.out_dir, metrics, tuning, regional_metrics, features, best_params)
    print("Saved combined SMOS inversion outputs to:", args.out_dir)


if __name__ == "__main__":
    main()
