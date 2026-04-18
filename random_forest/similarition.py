from utils_data import (
    DEFAULT_CSV,
    TARGET_COL,
    TB_FEATURES,
    add_brightness_temperature_features,
    load_buoy_csv,
)


def main() -> None:
    df = load_buoy_csv(DEFAULT_CSV)
    df = df[df[TARGET_COL] > 0].copy()
    df = add_brightness_temperature_features(df)
    cols = TB_FEATURES + [c for c in df.columns if c.startswith(("GR_", "PR_"))]

    print("Correlation with measured snow depth:")
    for col in cols:
        if col in df.columns:
            corr = df[col].corr(df[TARGET_COL])
            print(f"{col}: {corr:.3f}")


if __name__ == "__main__":
    main()
