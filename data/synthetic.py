"""
data/synthetic.py
=================
Offline EUR/USD generator used when live market data (yfinance) is unavailable
— e.g. inside sandboxed containers where outbound network calls are blocked.

We model the spot price with geometric Brownian motion (GBM) whose volatility
switches between calm and turbulent regimes via a 2-state Markov chain. This
reproduces volatility clustering — the stylised fact that large moves cluster
together — so the synthetic series behaves qualitatively like real FX data and
gives the models something non-trivial to learn.

Author: Clerence Mashile — ELTE Budapest
"""

import numpy as np
import pandas as pd


def generate_synthetic_eurusd(
    n: int = 2600,
    s0: float = 1.10,
    mu: float = 0.00,
    vol_calm: float = 0.0045,
    vol_turbulent: float = 0.0130,
    p_stay_calm: float = 0.985,
    p_stay_turbulent: float = 0.960,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic daily EUR/USD OHLC-style frame.

    Parameters
    ----------
    n : number of trading days to simulate.
    s0 : initial spot price.
    mu : daily drift (kept ~0; FX is close to a random walk).
    vol_calm / vol_turbulent : daily volatility in each regime.
    p_stay_calm / p_stay_turbulent : Markov self-transition probabilities.
    seed : RNG seed for reproducibility.

    Returns
    -------
    pandas.DataFrame indexed by business day with columns
    [Open, High, Low, Close, Volume].
    """
    rng = np.random.default_rng(seed)

    # --- 2-state Markov volatility regime ---------------------------------
    regimes = np.zeros(n, dtype=int)  # 0 = calm, 1 = turbulent
    for t in range(1, n):
        if regimes[t - 1] == 0:
            regimes[t] = 0 if rng.random() < p_stay_calm else 1
        else:
            regimes[t] = 1 if rng.random() < p_stay_turbulent else 0
    vol = np.where(regimes == 0, vol_calm, vol_turbulent)

    # --- GBM log returns --------------------------------------------------
    shocks = rng.standard_normal(n)
    log_ret = (mu - 0.5 * vol**2) + vol * shocks
    close = s0 * np.exp(np.cumsum(log_ret))

    # --- Build OHLC + volume around the close path ------------------------
    open_ = np.empty(n)
    open_[0] = s0
    open_[1:] = close[:-1]
    intraday = np.abs(rng.standard_normal(n)) * vol * close
    high = np.maximum(open_, close) + intraday
    low = np.minimum(open_, close) - intraday
    volume = (1e6 * (1.0 + 2.5 * (regimes == 1))).astype(int)  # heavier in turbulence

    idx = pd.bdate_range(start="2014-01-01", periods=n)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
    df.index.name = "Date"
    return df


if __name__ == "__main__":
    d = generate_synthetic_eurusd()
    print(d.head())
    print(f"\nGenerated {len(d)} rows | "
          f"Close range [{d.Close.min():.4f}, {d.Close.max():.4f}]")
