"""Generate exploratory data analysis plots for the PHC no-show dataset.

Reuses the same feature prep as train_model.py, so the charts reflect
exactly the features the model actually sees.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from train_model import DEFAULT_INPUT_CSV, _prepare_data

DEFAULT_PLOTS_DIR = Path("artifacts/plots")


def make_plots(input_csv: Path, plots_dir: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_csv.resolve()}")

    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    x, y = _prepare_data(df)
    data = x.copy()
    data["no_show"] = y

    # 1. No-show rate by age group
    data["age_group"] = pd.cut(
        data["Age"], bins=[-1, 12, 18, 35, 50, 65, 120],
        labels=["child", "teen", "young_adult", "adult", "older_adult", "senior"],
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    data.groupby("age_group", observed=True)["no_show"].mean().plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("No-show rate")
    ax.set_title("No-show rate by age group")
    plt.tight_layout()
    fig.savefig(plots_dir / "no_show_by_age_group.png", dpi=150)
    plt.close(fig)

    # 2. Wait days vs no-show
    fig, ax = plt.subplots(figsize=(6, 4))
    data.boxplot(column="wait_days", by="no_show", ax=ax)
    ax.set_xticklabels(["Showed up", "No-show"])
    ax.set_ylabel("Days between booking and appointment")
    ax.set_title("Waiting time vs no-show")
    plt.suptitle("")
    plt.tight_layout()
    fig.savefig(plots_dir / "wait_days_vs_no_show.png", dpi=150)
    plt.close(fig)

    # 3. No-show rate by SMS received
    fig, ax = plt.subplots(figsize=(5, 4))
    data.groupby("SMS_received")["no_show"].mean().plot(kind="bar", ax=ax, color="#55A868")
    ax.set_xticklabels(["No SMS", "Got SMS"], rotation=0)
    ax.set_ylabel("No-show rate")
    ax.set_title("No-show rate by SMS reminder")
    plt.tight_layout()
    fig.savefig(plots_dir / "no_show_by_sms.png", dpi=150)
    plt.close(fig)

    # 4. No-show rate by chronic conditions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data.groupby("Hipertension")["no_show"].mean().plot(kind="bar", ax=axes[0], color="#C44E52")
    axes[0].set_title("No-show rate: Hypertension")
    axes[0].set_xticklabels(["No", "Yes"], rotation=0)
    data.groupby("Diabetes")["no_show"].mean().plot(kind="bar", ax=axes[1], color="#8172B2")
    axes[1].set_title("No-show rate: Diabetes")
    axes[1].set_xticklabels(["No", "Yes"], rotation=0)
    plt.tight_layout()
    fig.savefig(plots_dir / "no_show_by_chronic_conditions.png", dpi=150)
    plt.close(fig)

    print(f"Saved 4 plots to {plots_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA plots for PHC no-show dataset")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_plots(input_csv=args.input_csv, plots_dir=args.plots_dir)
