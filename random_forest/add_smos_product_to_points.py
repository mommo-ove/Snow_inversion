from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from run_combined_regional_framework import (
    PRODUCT_COL,
    build_combined_training_table,
    load_combined_csv,
)
from utils_data import DEFAULT_CSV, DEFAULT_MAT


LAT_PATH = "/HDFEOS/GRIDS/NpPolarGrid12km/lat"
LON_PATH = "/HDFEOS/GRIDS/NpPolarGrid12km/lon"
DEFAULT_PRODUCT_PATH = "/HDFEOS/GRIDS/NpPolarGrid12km/Data Fields/SI_12km_NH_SNOWDEPTH_5DAY"


def import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("This script needs h5py. Install it with: pip install h5py") from exc
    return h5py


def resolve_lon_360(lon: np.ndarray) -> np.ndarray:
    return np.mod(lon, 360.0)


def latlon_to_unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lon_360 = resolve_lon_360(np.asarray(lon, dtype=float))
    lat_rad = np.radians(np.asarray(lat, dtype=float))
    lon_rad = np.radians(lon_360)
    return np.column_stack(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )


def find_first_he5(amsr_dir: Path) -> Path:
    files = sorted(amsr_dir.rglob("*.he5"))
    if not files:
        raise FileNotFoundError(f"No .he5 files found under {amsr_dir}")
    return files[0]


def build_file_index(amsr_dir: Path) -> dict[str, Path]:
    out = {}
    for path in sorted(amsr_dir.rglob("*.he5")):
        match = re.search(r"(20\d{6})", path.name)
        if match and match.group(1) not in out:
            out[match.group(1)] = path
    if not out:
        raise FileNotFoundError(f"No dated .he5 files found under {amsr_dir}")
    return out


def read_grid(sample_file: Path) -> tuple[np.ndarray, np.ndarray]:
    h5py = import_h5py()
    with h5py.File(sample_file, "r") as handle:
        lat = np.asarray(handle[LAT_PATH], dtype=float)
        lon = np.asarray(handle[LON_PATH], dtype=float)
    return lat, lon


def read_product_grid(path: Path, product_path: str, scale: float, invalid_min_cm: float) -> np.ndarray:
    h5py = import_h5py()
    with h5py.File(path, "r") as handle:
        raw = np.asarray(handle[product_path], dtype=float)
    product = raw * scale
    product[(raw >= invalid_min_cm) | (raw < 0)] = np.nan
    return product


def parse_datetime(df: pd.DataFrame) -> pd.Series:
    if "DateTime" in df.columns:
        parsed = pd.to_datetime(df["DateTime"], errors="coerce")
        if parsed.notna().any():
            return parsed
    required = {"Year", "Month", "Day"}
    if not required.issubset(df.columns):
        raise ValueError("Point table needs DateTime or Year/Month/Day columns.")
    return pd.to_datetime(
        pd.DataFrame(
            {
                "year": df["Year"],
                "month": df["Month"],
                "day": df["Day"],
            }
        ),
        errors="coerce",
    )


def load_point_table(args: argparse.Namespace) -> pd.DataFrame:
    if args.combined_csv:
        return load_combined_csv(args.combined_csv)
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        if "source" not in df.columns:
            df["source"] = "csv"
        return df
    return build_combined_training_table(args.icebridge_mat, args.imb_csv)


def add_product_values(args: argparse.Namespace) -> pd.DataFrame:
    df = load_point_table(args).copy()
    if not {"Latitude", "Longitude"}.issubset(df.columns):
        raise ValueError("Point table needs Latitude and Longitude columns.")

    df["_match_datetime"] = parse_datetime(df)
    df["_match_datestr"] = df["_match_datetime"].dt.strftime("%Y%m%d")

    file_index = build_file_index(args.amsr_dir)
    sample_file = find_first_he5(args.amsr_dir)
    grid_lat, grid_lon = read_grid(sample_file)
    tree = cKDTree(latlon_to_unit_xyz(grid_lat.ravel(), grid_lon.ravel()))

    product_values = np.full(len(df), np.nan)
    distance_km = np.full(len(df), np.nan)
    matched_file = np.full(len(df), "", dtype=object)

    valid_point = df["_match_datestr"].notna() & df["Latitude"].notna() & df["Longitude"].notna()
    for date_str, group in df[valid_point].groupby("_match_datestr"):
        product_file = file_index.get(date_str)
        if product_file is None:
            continue

        product = read_product_grid(
            product_file,
            args.product_path,
            scale=args.product_scale,
            invalid_min_cm=args.invalid_min_raw,
        )
        query = latlon_to_unit_xyz(group["Latitude"].to_numpy(), group["Longitude"].to_numpy())
        chord_dist, flat_idx = tree.query(query, k=1)
        km = 6371.0 * 2.0 * np.arcsin(np.clip(chord_dist / 2.0, 0.0, 1.0))
        values = product.ravel()[flat_idx]

        rows = group.index.to_numpy()
        product_values[rows] = values
        distance_km[rows] = km
        matched_file[rows] = product_file.name

    df[PRODUCT_COL] = product_values
    df["SMOS_Product_Match_Distance_Km"] = distance_km
    df["SMOS_Product_File"] = matched_file
    df = df.drop(columns=["_match_datetime", "_match_datestr"])
    return df


def write_summary(df: pd.DataFrame, out_csv: Path) -> None:
    matched = df[PRODUCT_COL].notna()
    print("Saved:", out_csv)
    print("Rows:", len(df))
    print("Rows with matched SMOS product:", int(matched.sum()))
    if matched.any():
        print(df.loc[matched, PRODUCT_COL].describe())
        if "source" in df.columns:
            print("Matched by source:")
            print(df.loc[matched, "source"].value_counts())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add point-matched AMSR/SMOS snow-depth product values to a training/validation point table."
    )
    parser.add_argument("--amsr-dir", type=Path, required=True, help="Local folder containing dated AMSR/AMSR2 .he5 files.")
    parser.add_argument("--out-csv", type=Path, default=Path("data/combined_points_with_smos_product.csv"))
    parser.add_argument("--combined-csv", type=Path, default=None, help="Existing combined point table to augment.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Generic point CSV to augment.")
    parser.add_argument("--icebridge-mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--imb-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--product-path", default=DEFAULT_PRODUCT_PATH)
    parser.add_argument(
        "--product-scale",
        type=float,
        default=0.01,
        help="Scale raw product units to meters. AMSR snow-depth product units are cm, so default is 0.01.",
    )
    parser.add_argument(
        "--invalid-min-raw",
        type=float,
        default=110.0,
        help="Raw product values >= this threshold are product flags, not snow depth.",
    )
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = add_product_values(args)
    df.to_csv(args.out_csv, index=False)
    write_summary(df, args.out_csv)


if __name__ == "__main__":
    main()
