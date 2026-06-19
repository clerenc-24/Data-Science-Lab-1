"""
train.py
========
Unified training loop for all three models.

Usage
-----
    python train.py --model baseline --cell lstm
    python train.py --model cnn_lstm
    python train.py --model dalg

Features: Adam + weight decay, gradient clipping, ReduceLROnPlateau, early
stopping on validation loss, and best-checkpoint saving to results/.

"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from config import CFG
from data.data_loader import get_data
from models import build_model, DISPLAY_NAMES
from evaluate import evaluate_model, print_results


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one(model_name: str, cfg=CFG, cell=None, loaders=None, verbose=True):
    """Train a single model and return (model, test_metrics, history)."""
    tc = cfg.train
    set_seed(tc.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if loaders is None:
        train_loader, val_loader, test_loader, meta = get_data(cfg)
    else:
        train_loader, val_loader, test_loader, meta = loaders

    model = build_model(model_name, meta["n_features"], cfg=cfg, cell=cell).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=tc.scheduler_factor, patience=tc.scheduler_patience,
    )
    loss_fn = nn.MSELoss()

    best_val, best_state, patience = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    n_params = sum(p.numel() for p in model.parameters())
    disp = DISPLAY_NAMES.get(model_name, model_name)
    if verbose:
        print(f"\n>>> Training {disp} on {device} | {n_params:,} params "
              f"| train={meta['n_train']} val={meta['n_val']} test={meta['n_test']}")

    for epoch in range(1, tc.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            if isinstance(out, tuple):
                out = out[0]
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= meta["n_train"]

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                if isinstance(out, tuple):
                    out = out[0]
                va_loss += loss_fn(out, yb).item() * len(xb)
        va_loss /= meta["n_val"]

        sched.step(va_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_loss < best_val - 1e-9:
            best_val, best_state, patience = va_loss, \
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | train {tr_loss:.6e} | val {va_loss:.6e}")

        if patience >= tc.early_stop_patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best val {best_val:.6e})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(cfg.eval.results_dir, exist_ok=True)
    ckpt = os.path.join(cfg.eval.results_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), ckpt)

    metrics = evaluate_model(model, test_loader, cfg=cfg, device=device)
    if verbose:
        print_results(disp, metrics)
        print(f"  checkpoint -> {ckpt}")
    return model, metrics, history


def main():
    ap = argparse.ArgumentParser(description="Train a Forex forecasting model.")
    ap.add_argument("--model", required=True,
                    choices=["baseline", "cnn_lstm", "dalg"])
    ap.add_argument("--cell", default=None, choices=["lstm", "gru", "rnn"],
                    help="RNN cell for the baseline only.")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    if args.epochs is not None:
        CFG.train.epochs = args.epochs
    train_one(args.model, cfg=CFG, cell=args.cell)


if __name__ == "__main__":
    main()
