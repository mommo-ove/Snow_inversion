from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COL = "Snow_Depth_m"
MODEL_COL = "Model_Retrieved_Snow_Depth_m"

PRODUCT_SPECS = [
    ("amsr_smos", "AMSR/SMOS product", "SMOS_Product_Snow_Depth_m", "SMOS_Product_Match_Distance_Km"),
    ("ecco_icecovered", "ECCO ice-covered", "ECCO_IceCovered_Snow_Depth_m", "ECCO_Product_Match_Distance_Km"),
    ("ecco_areaavg", "ECCO area-average", "ECCO_AreaAvg_Snow_Depth_m", "ECCO_Product_Match_Distance_Km"),
]


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_pred.to_numpy() - y_true.to_numpy()) ** 2)))


def evaluate(part: pd.DataFrame, pred_col: str) -> dict[str, float | int]:
    valid = part.dropna(subset=[TARGET_COL, pred_col])
    if valid.empty:
        return {"n": 0, "rmse_m": np.nan, "mae_m": np.nan, "bias_m": np.nan, "r2": np.nan}
    err = valid[pred_col] - valid[TARGET_COL]
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((valid[TARGET_COL] - valid[TARGET_COL].mean()) ** 2))
    return {
        "n": int(len(valid)),
        "rmse_m": rmse(valid[TARGET_COL], valid[pred_col]),
        "mae_m": float(np.mean(np.abs(err))),
        "bias_m": float(np.mean(err)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def add_metric_rows(rows: list[dict], subset: str, part: pd.DataFrame, product_id: str, product_name: str, product_col: str) -> None:
    for method_id, method_name, pred_col in [
        ("model", "RF model", MODEL_COL),
        (product_id, product_name, product_col),
    ]:
        metrics = evaluate(part, pred_col)
        if metrics["n"] == 0:
            continue
        rows.append(
            {
                "comparison": subset,
                "product_id": product_id,
                "method_id": method_id,
                "method": method_name,
                **metrics,
            }
        )


def rounded_grid_id(lat: pd.Series, lon: pd.Series, resolution_deg: float) -> pd.Series:
    lat_id = np.round(lat.astype(float) / resolution_deg) * resolution_deg
    lon_id = np.round(lon.astype(float) / resolution_deg) * resolution_deg
    return lat_id.map(lambda v: f"{v:.3f}") + "_" + lon_id.map(lambda v: f"{v:.3f}")


def aggregate_to_grid(
    part: pd.DataFrame,
    product_col: str,
    resolution_deg: float,
    min_points_per_cell: int,
) -> pd.DataFrame:
    work = part.dropna(subset=[TARGET_COL, MODEL_COL, product_col, "Latitude", "Longitude"]).copy()
    if work.empty:
        return work
    work["date"] = work[["Year", "Month", "Day"]].astype(int).astype(str).agg("-".join, axis=1)
    work["grid_id"] = rounded_grid_id(work["Latitude"], work["Longitude"], resolution_deg)
    grouped = (
        work.groupby(["date", "grid_id"], as_index=False)
        .agg(
            n_points=(TARGET_COL, "size"),
            Snow_Depth_m=(TARGET_COL, "mean"),
            Model_Retrieved_Snow_Depth_m=(MODEL_COL, "mean"),
            product_value=(product_col, "mean"),
        )
        .query("n_points >= @min_points_per_cell")
        .rename(columns={"product_value": product_col})
    )
    return grouped


def build_comparison_tables(df: pd.DataFrame, min_points_per_cell: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    aggregate_rows = []

    for product_id, product_name, product_col, distance_col in PRODUCT_SPECS:
        if product_col not in df.columns:
            continue

        common = df.dropna(subset=[TARGET_COL, MODEL_COL, product_col])
        add_metric_rows(rows, "common_valid_points", common, product_id, product_name, product_col)

        if distance_col in common.columns:
            near = common[common[distance_col] <= 20]
            add_metric_rows(rows, "distance_le_20km", near, product_id, product_name, product_col)

        if product_id.startswith("ecco") and "ECCO_Sea_Ice_Concentration" in common.columns:
            high_ice = common[common["ECCO_Sea_Ice_Concentration"] >= 0.8]
            add_metric_rows(rows, "ecco_sic_ge_0p8", high_ice, product_id, product_name, product_col)

        resolution = 0.5 if product_id.startswith("ecco") else 0.25
        agg = aggregate_to_grid(common, product_col, resolution, min_points_per_cell)
        add_metric_rows(rows, f"grid_aggregated_{resolution:g}deg", agg, product_id, product_name, product_col)
        if not agg.empty:
            aggregate_rows.append(
                agg.assign(product_id=product_id, product_name=product_name, grid_resolution_deg=resolution)
            )

    metrics = pd.DataFrame(rows)
    aggregates = pd.concat(aggregate_rows, ignore_index=True) if aggregate_rows else pd.DataFrame()
    return metrics, aggregates


def write_report(metrics: pd.DataFrame, aggregates: pd.DataFrame, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# Product Comparison Report\n\n")
        handle.write("This report separates product comparison from model training. The point truth remains IceBridge/IMB snow depth. ")
        handle.write("External products are compared only on common valid points and, where possible, on screened or grid-aggregated subsets.\n\n")
        handle.write("## Why This Is Needed\n\n")
        handle.write(
            "A direct point-vs-grid comparison can exaggerate product errors because point observations and satellite/reanalysis products have different spatial supports. "
            "The table below therefore includes common-valid, distance-screened, sea-ice-screened, and approximate grid-aggregated comparisons.\n\n"
        )
        handle.write("## Metrics\n\n")
        if metrics.empty:
            handle.write("_No comparable product rows found._\n")
        else:
            handle.write(metrics.to_markdown(index=False, floatfmt=".4g"))
            handle.write("\n\n")
        handle.write("## Interpretation Rule\n\n")
        handle.write(
            "Use `common_valid_points` as the main point-level comparison. Use distance and SIC subsets as diagnostic checks. "
            "Use grid-aggregated rows as a fairer product-scale comparison, but report the number of aggregated cells because it may be much smaller than the point count.\n"
        )
        if not aggregates.empty:
            handle.write(f"\nAggregated product-scale rows: {len(aggregates)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fair product comparison diagnostics from holdout point predictions.")
    parser.add_argument(
        "--point-csv",
        type=Path,
        default=Path("reports/combined_regional_framework/holdout_10pct_point_comparison.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/product_comparison_diagnostics"))
    parser.add_argument("--min-points-per-cell", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.point_csv)
    metrics, aggregates = build_comparison_tables(df, args.min_points_per_cell)
    metrics.to_csv(args.out_dir / "product_comparison_metrics.csv", index=False)
    if not aggregates.empty:
        aggregates.to_csv(args.out_dir / "product_scale_aggregated_points.csv", index=False)
    write_report(metrics, aggregates, args.out_dir / "product_comparison_report.md")
    print("Saved product comparison diagnostics to:", args.out_dir)


if __name__ == "__main__":
    main()
