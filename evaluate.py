"""
evaluate.py
===========
Evaluation metrics for next-step return forecasts.

Regression quality
-------------------
* RMSE, MAE — error on the predicted return.

Trading relevance
-----------------
* Directional accuracy — fraction of days the predicted sign matches the
  realised sign. 50 % is the coin-flip baseline; what matters for profit.
* Sharpe ratio — annualised risk-adjusted return of a simple long/short
  strategy that takes position sign(prediction) each day and earns the
  realised return. Annualised with sqrt(252).

Author: Clerence Mashile — ELTE Budapest
"""

import json
import os

import numpy as np
import torch

from config import CFG


# --------------------------------------------------------------------------- #
# Core metrics                                                                 #
# --------------------------------------------------------------------------- #
def compute_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_directional_accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def compute_sharpe(y_true, y_pred, trading_days: int = 252) -> float:
    """Annualised Sharpe of a sign(prediction) long/short strategy."""
    strat = np.sign(y_pred) * y_true            # daily strategy return
    mu, sd = strat.mean(), strat.std()
    if sd < 1e-12:
        return 0.0
    return float(mu / sd * np.sqrt(trading_days))


# --------------------------------------------------------------------------- #
# Prediction + evaluation                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, loader, device="cpu"):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)
        if isinstance(out, tuple):
            out = out[0]
        preds.append(out.cpu().numpy().ravel())
        trues.append(yb.numpy().ravel())
    return np.concatenate(trues), np.concatenate(preds)


def evaluate_model(model, loader, cfg=CFG, device="cpu") -> dict:
    y_true, y_pred = predict(model, loader, device=device)
    td = cfg.eval.trading_days
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "directional_accuracy": compute_directional_accuracy(y_true, y_pred),
        "sharpe": compute_sharpe(y_true, y_pred, td),
        "_y_true": y_true, "_y_pred": y_pred,   # kept for plotting; stripped on save
    }


def print_results(name: str, metrics: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  RMSE                 : {metrics['rmse']:.6f}")
    print(f"  MAE                  : {metrics['mae']:.6f}")
    print(f"  Directional accuracy : {metrics['directional_accuracy'] * 100:.2f}%")
    print(f"  Sharpe ratio         : {metrics['sharpe']:.3f}")


def save_results(all_metrics: dict, cfg=CFG, filename="metrics.json") -> str:
    os.makedirs(cfg.eval.results_dir, exist_ok=True)
    clean = {
        name: {k: v for k, v in m.items() if not k.startswith("_")}
        for name, m in all_metrics.items()
    }
    path = os.path.join(cfg.eval.results_dir, filename)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n[eval] Wrote metrics to {path}")
    return path
