from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

DEFAULT_CSV = DATA_DIR / "Validation_Dataset_IMB_AllBuoys_Combined_AllData.csv"
DEFAULT_MAT = DATA_DIR / "Master_Dataset_AllFeatures.mat"
TARGET_COL = "Snow_Depth_m"
GROUP_COL = "Buoy"

TB_FEATURES = [
    "TB_18V",
    "TB_18H",
    "TB_23V",
    "TB_23H",
    "TB_36V",
    "TB_36H",
    "TB_89V",
    "TB_89H",
]

CONTEXT_FEATURES = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "Latitude",
    "Longitude",
    "Distance_Km",
    "Air_Temp_C",
    "Air_Pressure_mbar",
]

LEAKAGE_COLUMNS = {
    TARGET_COL,
    "Ice_Thickness_m",
    "Top_Ice_Pos_m",
    "Bottom_Ice_Pos_m",
    "DateTime",
    GROUP_COL,
}


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def require_file(path: str | Path) -> Path:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Missing file: {resolved}\n"
            "Put the data under the repo data/ folder or pass an explicit path."
        )
    return resolved


def load_buoy_csv(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


def add_brightness_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = [
        ("GR_36V_18V", "TB_36V", "TB_18V"),
        ("GR_89V_18V", "TB_89V", "TB_18V"),
        ("PR_18", "TB_18V", "TB_18H"),
        ("PR_23", "TB_23V", "TB_23H"),
        ("PR_36", "TB_36V", "TB_36H"),
        ("PR_89", "TB_89V", "TB_89H"),
    ]
    for name, high, low in pairs:
        if high in out.columns and low in out.columns:
            denom = out[high] + out[low]
            out[name] = np.where(denom != 0, (out[high] - out[low]) / denom, np.nan)
    return out


def available_features(df: pd.DataFrame, include_context: bool, include_derived: bool) -> list[str]:
    candidates = list(TB_FEATURES)
    if include_context:
        candidates.extend(CONTEXT_FEATURES)
    if include_derived:
        candidates.extend([c for c in df.columns if c.startswith(("GR_", "PR_"))])
    return [c for c in candidates if c in df.columns and c not in LEAKAGE_COLUMNS]


def make_model_table(
    df: pd.DataFrame,
    include_context: bool = False,
    include_derived: bool = True,
    min_snow_depth: float | None = 0.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, list[str]]:
    data = add_brightness_temperature_features(df)
    features = available_features(data, include_context, include_derived)

    required = features + [TARGET_COL]
    if GROUP_COL in data.columns:
        required.append(GROUP_COL)

    model_df = data[required].replace([np.inf, -np.inf], np.nan)
    if min_snow_depth is not None:
        model_df = model_df[model_df[TARGET_COL] > min_snow_depth]
    model_df = model_df.dropna(subset=features + [TARGET_COL])

    x = model_df[features]
    y = model_df[TARGET_COL]
    groups = model_df[GROUP_COL] if GROUP_COL in model_df.columns else None
    return x, y, groups, features


def print_hdf5_tree(path: str | Path, max_depth: int = 4) -> None:
    import h5py

    mat_path = require_file(path)
    print(f"MAT file: {mat_path}")
    with h5py.File(mat_path, "r") as handle:
        def visit(name: str, obj) -> None:
            depth = name.count("/")
            if depth > max_depth:
                return
            indent = "  " * depth
            if hasattr(obj, "shape"):
                print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{indent}{name}/")

        handle.visititems(visit)


def summarize_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    present = [c for c in columns if c in df.columns]
    return df[present].describe().T
