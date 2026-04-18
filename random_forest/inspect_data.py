from __future__ import annotations

import argparse

from utils_data import (
    CONTEXT_FEATURES,
    DEFAULT_CSV,
    DEFAULT_MAT,
    GROUP_COL,
    TARGET_COL,
    TB_FEATURES,
    load_buoy_csv,
    print_hdf5_tree,
    summarize_columns,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local IMB/IceBridge data files.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to buoy CSV file.")
    parser.add_argument("--mat", default=str(DEFAULT_MAT), help="Path to MATLAB v7.3 file.")
    parser.add_argument("--skip-mat", action="store_true", help="Do not inspect the MAT file.")
    args = parser.parse_args()

    df = load_buoy_csv(args.csv)
    print("CSV shape:", df.shape)
    print("CSV columns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nFirst rows:")
    print(df.head())

    if GROUP_COL in df.columns:
        print("\nBuoy counts:")
        print(df[GROUP_COL].value_counts().sort_index())

    print("\nTarget summary:")
    print(df[TARGET_COL].describe())

    print("\nFeature summary:")
    print(summarize_columns(df, TB_FEATURES + CONTEXT_FEATURES + [TARGET_COL]))

    if not args.skip_mat:
        print("\nMAT/HDF5 structure:")
        print_hdf5_tree(args.mat)


if __name__ == "__main__":
    main()
