"""
data/data_loader.py
===================
End-to-end data pipeline for the Forex ML project.

Pipeline
--------
1. load_data()              -> raw OHLCV DataFrame (yfinance, else synthetic).
2. add_technical_indicators -> engineered feature columns.
3. prepare_sequences()      -> sliding-window tensors (X, y) + the realised
                               next-step returns used for trading metrics.
4. ForexDataset / make_loaders / get_data -> PyTorch DataLoaders + metadata.

Target convention
-----------------
We predict the *next-step log return* r_{t+1}. Returns (rather than raw price)
are stationary, make directional accuracy and the Sharpe ratio meaningful, and
avoid the trivial "predict yesterday's price" baseline.

Author: Clerence Mashile — ELTE Budapest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from config import CFG
from data.synthetic import generate_synthetic_eurusd


# --------------------------------------------------------------------------- #
# 1. Loading                                                                   #
# --------------------------------------------------------------------------- #
def load_data(cfg=CFG) -> pd.DataFrame:
    """Return a daily OHLCV DataFrame for the configured instrument.

    Tries yfinance first (real market data). If that fails — no network, the
    symbol is blocked, or an empty frame is returned — it falls back to the
    synthetic GBM generator so the rest of the pipeline always runs.

    To use REAL EUR/USD data, run outside a sandbox with internet access and
    set ``CFG.data.use_synthetic = False`` (and ``pip install yfinance``).
    """
    dc = cfg.data
    if not dc.use_synthetic:
        try:
            import yfinance as yf  # imported lazily so it isn't a hard dep
            df = yf.download(
                dc.ticker, start=dc.start, end=dc.end,
                interval=dc.interval, progress=False, auto_adjust=True,
            )
            if df is not None and len(df) > 0:
                # Flatten possible MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                print(f"[data] Loaded {len(df)} live rows for {dc.ticker} via yfinance.")
                return df
            print("[data] yfinance returned no rows — using synthetic data.")
        except Exception as exc:  # noqa: BLE001
            print(f"[data] yfinance unavailable ({exc}) — using synthetic data.")

    df = generate_synthetic_eurusd(n=dc.synthetic_n, seed=cfg.train.seed)
    print(f"[data] Using synthetic EUR/USD: {len(df)} rows.")
    return df


# --------------------------------------------------------------------------- #
# 2. Feature engineering                                                       #
# --------------------------------------------------------------------------- #
def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def add_technical_indicators(df: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Append engineered features. All are causal (use only past/current data)."""
    dc = cfg.data
    out = df.copy()
    close = out["Close"]

    out["log_ret"] = np.log(close / close.shift(1))
    for w in dc.sma_windows:
        out[f"sma_ratio_{w}"] = close / close.rolling(w).mean() - 1.0
    out["rsi"] = _rsi(close, dc.rsi_window) / 100.0
    out["volatility"] = out["log_ret"].rolling(dc.vol_window).std()
    out["momentum"] = close / close.shift(dc.mom_window) - 1.0
    out["hl_range"] = (out["High"] - out["Low"]) / close
    out["vol_chg"] = out["Volume"].pct_change().clip(-3, 3)

    out = out.dropna().reset_index(drop=False)
    return out


FEATURE_COLS = [
    "log_ret",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20",
    "rsi", "volatility", "momentum", "hl_range", "vol_chg",
]


# --------------------------------------------------------------------------- #
# 3. Sequencing                                                                #
# --------------------------------------------------------------------------- #
def prepare_sequences(df: pd.DataFrame, cfg=CFG):
    """Build sliding-window samples.

    Returns
    -------
    X : (N, seq_len, n_features) float32
    y : (N,)  next-step log return (regression target)
    feat_cols : list[str]
    """
    dc = cfg.data
    feats = df[FEATURE_COLS].values.astype(np.float32)
    target = df["log_ret"].shift(-dc.horizon).values.astype(np.float32)  # r_{t+1}

    X, y = [], []
    last = len(df) - dc.horizon
    for t in range(dc.seq_len, last):
        X.append(feats[t - dc.seq_len: t])
        y.append(target[t - 1])  # return realised one step after the window
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y, FEATURE_COLS


class ForexDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def _chrono_split(X, y, cfg=CFG):
    n = len(X)
    n_tr = int(n * cfg.data.train_frac)
    n_va = int(n * cfg.data.val_frac)
    sl = slice
    return (
        (X[sl(0, n_tr)], y[sl(0, n_tr)]),
        (X[sl(n_tr, n_tr + n_va)], y[sl(n_tr, n_tr + n_va)]),
        (X[sl(n_tr + n_va, n)], y[sl(n_tr + n_va, n)]),
    )


def make_loaders(cfg=CFG):
    """Return train/val/test DataLoaders plus metadata dict.

    Feature scaling is fit on the TRAIN split only and applied to all splits to
    avoid look-ahead bias.
    """
    df = add_technical_indicators(load_data(cfg), cfg)
    X, y, feat_cols = prepare_sequences(df, cfg)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _chrono_split(X, y, cfg)

    # Scale features using train statistics only
    scaler = StandardScaler()
    nf = Xtr.shape[-1]
    scaler.fit(Xtr.reshape(-1, nf))

    def _scale(a):
        s = a.shape
        return scaler.transform(a.reshape(-1, nf)).reshape(s).astype(np.float32)

    Xtr, Xva, Xte = _scale(Xtr), _scale(Xva), _scale(Xte)

    bs = cfg.train.batch_size
    train_loader = DataLoader(ForexDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    val_loader = DataLoader(ForexDataset(Xva, yva), batch_size=bs, shuffle=False)
    test_loader = DataLoader(ForexDataset(Xte, yte), batch_size=bs, shuffle=False)

    meta = {
        "n_features": nf,
        "feat_cols": feat_cols,
        "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "seq_len": cfg.data.seq_len,
    }
    return train_loader, val_loader, test_loader, meta


def get_data(cfg=CFG):
    """Convenience wrapper used by train.py / run_all.py."""
    return make_loaders(cfg)


if __name__ == "__main__":
    tr, va, te, meta = make_loaders()
    print("Metadata:", meta)
    xb, yb = next(iter(tr))
    print("Batch X:", tuple(xb.shape), "| Batch y:", tuple(yb.shape))
