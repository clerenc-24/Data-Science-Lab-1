"""
explain.py
==========
Explainability for the DALG model.

DALG exposes two attention maps via ``model(x, return_attn=True)``:

    feature attention  — averaged over the test set, shows which engineered
                         inputs (RSI, volatility, momentum, ...) the model
                         relies on most.
    temporal attention — averaged over the test set, shows which positions in
                         the 30-day look-back window carry the most weight.

These maps are the basis of the trustworthiness analysis in the report and a
lightweight, model-native alternative to post-hoc SHAP.

Run:
    python explain.py          # requires results/dalg.pt from train.py/run_all.py

Author: Clerence Mashile — ELTE Budapest
"""

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG
from data.data_loader import get_data, FEATURE_COLS
from models import build_dalg

NAVY, GOLD = "#1F3864", "#C9A227"


@torch.no_grad()
def collect_attention(model, loader, device="cpu"):
    model.eval()
    feat_acc, temp_acc, n = None, None, 0
    for xb, _ in loader:
        xb = xb.to(device)
        _, attn = model(xb, return_attn=True)
        f = attn["feature"].mean(dim=1).cpu().numpy()   # (B, F) avg over time
        t = attn["temporal"].cpu().numpy()              # (B, T)
        feat_acc = f.sum(0) if feat_acc is None else feat_acc + f.sum(0)
        temp_acc = t.sum(0) if temp_acc is None else temp_acc + t.sum(0)
        n += len(xb)
    return feat_acc / n, temp_acc / n


def main(path=None):
    path = path or os.path.join(CFG.eval.results_dir, "dalg_explainability.png")
    _, _, test_loader, meta = get_data(CFG)

    model = build_dalg(meta["n_features"], cfg=CFG)
    ckpt = os.path.join(CFG.eval.results_dir, "dalg.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"{ckpt} not found — run run_all.py or train.py first.")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    feat_w, temp_w = collect_attention(model, test_loader)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("DALG explainability — average attention on the EUR/USD test set",
                 fontsize=13, fontweight="bold")

    order = np.argsort(feat_w)
    ax[0].barh(np.array(FEATURE_COLS)[order], feat_w[order], color=NAVY)
    ax[0].set_title("Feature attention (which inputs matter)")
    ax[0].set_xlabel("mean attention weight")

    ax[1].plot(range(1, len(temp_w) + 1), temp_w, color=GOLD, lw=2, marker="o", ms=3)
    ax[1].set_title("Temporal attention (which look-back days matter)")
    ax[1].set_xlabel("position in 30-day window (1 = oldest, 30 = most recent)")
    ax[1].set_ylabel("mean attention weight")
    ax[1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(CFG.eval.results_dir, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[explain] Wrote {path}")
    # Also print a small ranked table for the report
    print("\nTop features by attention:")
    for i in order[::-1]:
        print(f"  {FEATURE_COLS[i]:<14} {feat_w[i]:.4f}")


if __name__ == "__main__":
    main()
