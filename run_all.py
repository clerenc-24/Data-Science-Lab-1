"""
run_all.py
==========
Reproduce the full experiment: train the baseline and both challengers on the
same data split, collect test metrics, and emit:

    results/metrics.json          — all metrics
    results/comparison.png        — 4-panel metric comparison + prediction trace

Run:
    python run_all.py
    python run_all.py --epochs 40

Author: Clerence Mashile — ELTE Budapest
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG
from data.data_loader import get_data
from models import DISPLAY_NAMES
from train import train_one
from evaluate import save_results

ORDER = ["baseline", "cnn_lstm", "dalg"]
COLORS = {"baseline": "#8A8F98", "cnn_lstm": "#C9A227", "dalg": "#1F3864"}


def make_comparison_figure(all_metrics, cfg=CFG, path=None):
    path = path or os.path.join(cfg.eval.results_dir, "comparison.png")
    names = [DISPLAY_NAMES[k] for k in ORDER]
    colors = [COLORS[k] for k in ORDER]

    rmse = [all_metrics[k]["rmse"] for k in ORDER]
    da = [all_metrics[k]["directional_accuracy"] * 100 for k in ORDER]
    sharpe = [all_metrics[k]["sharpe"] for k in ORDER]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Baseline vs Challengers — EUR/USD next-step return forecasting",
                 fontsize=14, fontweight="bold")

    ax[0, 0].bar(names, rmse, color=colors)
    ax[0, 0].set_title("RMSE (lower is better)")
    ax[0, 0].tick_params(axis="x", rotation=15)

    ax[0, 1].bar(names, da, color=colors)
    ax[0, 1].axhline(50, ls="--", c="red", lw=1, label="coin flip (50%)")
    ax[0, 1].set_title("Directional accuracy % (higher is better)")
    ax[0, 1].legend()
    ax[0, 1].tick_params(axis="x", rotation=15)

    ax[1, 0].bar(names, sharpe, color=colors)
    ax[1, 0].axhline(0, c="black", lw=0.8)
    ax[1, 0].set_title("Annualised Sharpe (higher is better)")
    ax[1, 0].tick_params(axis="x", rotation=15)

    # Prediction trace for the best model (by Sharpe) on a window of the test set
    best = max(ORDER, key=lambda k: all_metrics[k]["sharpe"])
    yt = all_metrics[best]["_y_true"][:150]
    yp = all_metrics[best]["_y_pred"][:150]
    ax[1, 1].plot(yt, label="realised", c="black", lw=1)
    ax[1, 1].plot(yp, label="predicted", c=COLORS[best], lw=1)
    ax[1, 1].set_title(f"Predicted vs realised return — {DISPLAY_NAMES[best]}")
    ax[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(cfg.eval.results_dir, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[run_all] Wrote figure -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    if args.epochs is not None:
        CFG.train.epochs = args.epochs

    # Build the data ONCE so every model sees the identical split.
    loaders = get_data(CFG)

    all_metrics = {}
    for name in ORDER:
        _, metrics, _ = train_one(name, cfg=CFG, loaders=loaders, verbose=True)
        all_metrics[name] = metrics

    save_results(all_metrics, cfg=CFG)
    make_comparison_figure(all_metrics, cfg=CFG)

    print("\n================ SUMMARY ================")
    print(f"{'Model':<28}{'RMSE':>10}{'DirAcc%':>10}{'Sharpe':>10}")
    for k in ORDER:
        m = all_metrics[k]
        print(f"{DISPLAY_NAMES[k]:<28}{m['rmse']:>10.6f}"
              f"{m['directional_accuracy']*100:>10.2f}{m['sharpe']:>10.3f}")


if __name__ == "__main__":
    main()
