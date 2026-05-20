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


def add_passive_microwave_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_brightness_temperature_features(df)
    required = set(TB_FEATURES)
    if not required.issubset(out.columns):
        return out

    low_v = (out["TB_18V"] + out["TB_23V"]) / 2.0
    low_h = (out["TB_18H"] + out["TB_23H"]) / 2.0
    high_v = (out["TB_36V"] + out["TB_89V"]) / 2.0
    high_h = (out["TB_36H"] + out["TB_89H"]) / 2.0
    low_mean = (low_v + low_h) / 2.0
    high_mean = (high_v + high_h) / 2.0

    out["PM_LOW_FREQ_MEAN"] = low_mean
    out["PM_HIGH_FREQ_MEAN"] = high_mean
    out["PM_LOW_HIGH_DIFF"] = low_mean - high_mean
    out["PM_LOW_HIGH_RATIO"] = np.where(high_mean != 0, low_mean / high_mean, np.nan)
    out["PM_DEPTH_SENSITIVITY"] = np.where(
        (low_mean + high_mean) != 0,
        (low_mean - high_mean) / (low_mean + high_mean),
        np.nan,
    )
    out["PM_SPECTRAL_SLOPE_V"] = (out["TB_89V"] - out["TB_18V"]) / (89.0 - 18.0)
    out["PM_SPECTRAL_SLOPE_H"] = (out["TB_89H"] - out["TB_18H"]) / (89.0 - 18.0)
    out["PM_POL_DIFF_LOW"] = low_v - low_h
    out["PM_POL_DIFF_HIGH"] = high_v - high_h
    out["PM_POL_DIFF_CHANGE"] = out["PM_POL_DIFF_HIGH"] - out["PM_POL_DIFF_LOW"]

    pr_cols = [c for c in ["PR_18", "PR_23", "PR_36", "PR_89"] if c in out.columns]
    gr_cols = [c for c in ["GR_36V_18V", "GR_89V_18V"] if c in out.columns]
    if pr_cols:
        out["PM_MEAN_PR"] = out[pr_cols].mean(axis=1)
    if gr_cols:
        out["PM_MEAN_GR"] = out[gr_cols].mean(axis=1)
    return out


def available_features(
    df: pd.DataFrame,
    include_context: bool,
    include_derived: bool,
    include_passive_physics: bool = False,
) -> list[str]:
    candidates = list(TB_FEATURES)
    if include_context:
        candidates.extend(CONTEXT_FEATURES)
    if include_derived:
        candidates.extend([c for c in df.columns if c.startswith(("GR_", "PR_"))])
    if include_passive_physics:
        candidates.extend([c for c in df.columns if c.startswith("PM_")])
    return [c for c in candidates if c in df.columns and c not in LEAKAGE_COLUMNS]


def make_model_table(
    df: pd.DataFrame,
    include_context: bool = False,
    include_derived: bool = True,
    include_passive_physics: bool = False,
    min_snow_depth: float | None = 0.0,
    exclude_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, list[str]]:
    data = add_passive_microwave_features(df) if include_passive_physics else add_brightness_temperature_features(df)
    features = available_features(data, include_context, include_derived, include_passive_physics)
    excluded = {c for c in (exclude_columns or []) if c}
    features = [c for c in features if c not in excluded]

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
