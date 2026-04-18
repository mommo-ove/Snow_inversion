from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from utils_data import DEFAULT_CSV, TARGET_COL, load_buoy_csv


def main() -> None:
    df = load_buoy_csv(DEFAULT_CSV)
    target_tb = "TB_36H"
    corr = df[target_tb].corr(df[TARGET_COL])

    out_dir = Path("random_forest/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "tb36h_vs_snow_depth.png"

    plt.figure(figsize=(10, 6))
    sns.regplot(
        x=df[target_tb],
        y=df[TARGET_COL],
        scatter_kws={"alpha": 0.3, "s": 10, "color": "blue"},
        line_kws={"color": "red"},
    )
    plt.title(f"IMB data: {target_tb} vs snow depth\nCorrelation R = {corr:.2f}")
    plt.xlabel(f"Brightness temperature {target_tb} (K)")
    plt.ylabel("Measured snow depth (m)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    print(f"{target_tb} correlation with snow depth: {corr:.3f}")
    print("Saved:", fig_path)


if __name__ == "__main__":
    main()
