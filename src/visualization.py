import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List
import os

import config


def plot_series_forecast(df: pd.DataFrame, series_id: str,
                         pred_cols: List[str], save_path: str | None = None):
    s = df[df["unique_id"] == series_id].sort_values("ds")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s["ds"], s["y"], label="Actual", color="black", linewidth=1.5)
    colors = plt.cm.tab10.colors
    for i, col in enumerate(pred_cols):
        if col in s.columns:
            ax.plot(s["ds"], s[col], label=col, color=colors[i % len(colors)], linewidth=1, alpha=0.8)
    ax.set_title(f"Series: {series_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unit Sales")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_metrics_comparison(metrics_table: pd.DataFrame, save_path: str | None = None):
    n_metrics = len(metrics_table.columns)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_table.columns):
        values = metrics_table[metric].sort_values()
        bars = ax.barh(values.index, values.values)
        bars[0].set_color("green")
        ax.set_title(metric)
        ax.set_xlabel(metric)

    plt.suptitle("Model Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_sample_forecasts(df: pd.DataFrame, pred_cols: List[str],
                          n_samples: int = 6, save_dir: str | None = None):
    uids = df["unique_id"].unique()
    rng = np.random.RandomState(config.RANDOM_SEED)
    sample_ids = rng.choice(uids, min(n_samples, len(uids)), replace=False)

    for uid in sample_ids:
        path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"forecast_{uid}.png")
        plot_series_forecast(df, uid, pred_cols, save_path=path)
