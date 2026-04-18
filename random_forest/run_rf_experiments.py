from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from utils_data import DEFAULT_CSV, GROUP_COL, TARGET_COL, load_buoy_csv, make_model_table


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def make_model(args: argparse.Namespace) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
    )


def evaluate(name: str, model: RandomForestRegressor, x_test, y_test) -> dict[str, float | str]:
    pred = model.predict(x_test)
    return {
        "experiment": name,
        "n_test": len(y_test),
        "rmse_m": rmse(y_test, pred),
        "mae_m": mean_absolute_error(y_test, pred),
        "r2": r2_score(y_test, pred),
    }


def save_observed_vs_predicted(name: str, y_true, y_pred, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{name}_observed_vs_predicted.png"

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.55)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    plt.xlabel("Observed snow depth (m)")
    plt.ylabel("Predicted snow depth (m)")
    plt.title(name.replace("_", " "))
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print("Saved:", fig_path)


def save_importance(name: str, model: RandomForestRegressor, feature_names: list[str], out_dir: Path) -> None:
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    csv_path = out_dir / f"{name}_feature_importance.csv"
    importances.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    top = importances.head(20).iloc[::-1]
    fig_path = out_dir / f"{name}_feature_importance.png"
    plt.figure(figsize=(8, max(4, len(top) * 0.3)))
    plt.barh(top["feature"], top["importance"], color="teal")
    plt.xlabel("Relative importance")
    plt.title(f"{name.replace('_', ' ')} feature importance")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print("Saved:", fig_path)


def save_permutation_importance(
    name: str,
    model: RandomForestRegressor,
    x_test,
    y_test,
    out_dir: Path,
    random_state: int,
) -> None:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    table = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    csv_path = out_dir / f"{name}_permutation_importance.csv"
    table.to_csv(csv_path, index=False)
    print("Saved:", csv_path)


def run_random_split(args, x, y, features, out_dir: Path) -> dict[str, float | str]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.random_state
    )
    model = make_model(args)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    save_observed_vs_predicted("random_split", y_test, pred, out_dir)
    save_importance("random_split", model, features, out_dir)
    return evaluate("random_split", model, x_test, y_test)


def run_group_split(args, x, y, groups, features, out_dir: Path) -> dict[str, float | str]:
    if groups is None:
        raise ValueError(f"Group split requires a {GROUP_COL} column.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    test_groups = sorted(groups.iloc[test_idx].astype(str).unique())

    model = make_model(args)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    save_observed_vs_predicted("buoy_held_out", y_test, pred, out_dir)
    save_importance("buoy_held_out", model, features, out_dir)
    save_permutation_importance(
        "buoy_held_out", model, x_test, y_test, out_dir, args.random_state
    )

    metrics = evaluate("buoy_held_out", model, x_test, y_test)
    metrics["held_out_buoys"] = ",".join(test_groups)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random forest snow depth experiments.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to buoy CSV file.")
    parser.add_argument("--out-dir", default="random_forest/outputs", help="Output folder.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--no-derived", action="store_true")
    args = parser.parse_args()

    df = load_buoy_csv(args.csv)
    x, y, groups, features = make_model_table(
        df,
        include_context=args.include_context,
        include_derived=not args.no_derived,
        min_snow_depth=0.0,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Rows used:", len(x))
    print("Target:", TARGET_COL)
    print("Features:")
    for feature in features:
        print(f"  - {feature}")

    metrics = [
        run_random_split(args, x, y, features, out_dir),
        run_group_split(args, x, y, groups, features, out_dir),
    ]

    metrics_df = pd.DataFrame(metrics)
    metrics_path = out_dir / "rf_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("\nMetrics:")
    print(metrics_df)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
