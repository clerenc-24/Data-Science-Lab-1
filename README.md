# Machine Learning in Forex Prediction - Baseline vs Challengers (EUR/USD)

Machine-learning models for next-step EUR/USD return forecasting, built for the
**Data Science Lab 1 ** project at **ELTE Budapest**.

> *How can hybrid mathematical–ML models best predict EUR/USD movements, and how
> do explainability tools affect model trustworthiness for real-world
> deployment?*

This repository implements and compares three sequence models on a common,
leak-free pipeline, and exposes attention-based explanations for the headline
model.

| Tag | Model | Role |
|-----|-------|------|
| —   | **Baseline LSTM** (`models/baseline_lstm.py`) | Reference recurrent model (LSTM/GRU/RNN selectable) |
| C1  | **CNN-LSTM-Attention** (`models/cnn_lstm_attention.py`) | Conv front-end + LSTM + temporal attention |
| C2  | **DALG** (`models/dalg.py`) | **Dual-Attention LSTM-GRU** — the refined contribution |

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Reproduce the whole experiment (trains all 3, writes results/)
python run_all.py

# 3. Generate DALG attention explanations
python explain.py
```

Train a single model instead:

```bash
python train.py --model baseline --cell lstm
python train.py --model cnn_lstm
python train.py --model dalg
```

Outputs land in `results/`: `metrics.json`, `comparison.png`,
`dalg_explainability.png`, and one `*.pt` checkpoint per model.

---

## Data

By default the pipeline uses an **offline synthetic EUR/USD generator**
(`data/synthetic.py`): geometric Brownian motion with a 2-state Markov
volatility regime, which reproduces volatility clustering. This keeps the repo
fully runnable in sandboxes where outbound network calls are blocked.

**To use real EUR/USD data:**

```python
# config.py
DataConfig.use_synthetic = False    # and: pip install yfinance
```

`data/data_loader.py` will then pull daily bars via `yfinance` and fall back to
synthetic data automatically if the download fails.

### Pipeline

```
load_data  ->  add_technical_indicators  ->  prepare_sequences  ->  DataLoaders
```

* **Features (9):** log return, 5/10/20-day SMA ratios, RSI, rolling
  volatility, momentum, high–low range, volume change.
* **Target:** next-step log return `r_{t+1}` (stationary; makes directional
  accuracy and Sharpe meaningful).
* **Look-back:** 30 days. **Split:** 70 / 15 / 15 chronological. Feature
  scaling is fit on the **train split only** to avoid look-ahead bias.

---

## Evaluation

| Metric | Meaning |
|--------|---------|
| RMSE / MAE | Point-forecast error on the return |
| Directional accuracy | Share of days the predicted sign is correct (50 % = coin flip) |
| Sharpe ratio | Annualised risk-adjusted return of a `sign(prediction)` long/short strategy |

See `evaluate.py`.

---

## Repository layout

```
forex_ml_repo/
├── config.py                 # all hyperparameters & settings (dataclasses)
├── requirements.txt
├── train.py                  # unified training loop (argparse)
├── run_all.py                # train all 3 + comparison figure
├── evaluate.py               # metrics + JSON export
├── explain.py                # DALG attention explainability
├── data/
│   ├── data_loader.py        # load → features → sequences → loaders
│   └── synthetic.py          # offline GBM EUR/USD generator
├── models/
│   ├── baseline_lstm.py      # Baseline RNN
│   ├── cnn_lstm_attention.py # Challenger C1
│   └── dalg.py               # Challenger C2 (DALG)
└── results/                  # generated artifacts
```

---

## Notes & limitations

The default experiment runs on **synthetic, near-random-walk data**, so absolute
performance is intentionally modest — no model produces a profitable strategy on
a random walk, which is the correct outcome. The repository is designed so that
swapping in real data and richer features (GARCH volatility, sentiment) is a
configuration change, not a rewrite. Planned extensions: GARCH-augmented DALG,
multi-regime evaluation (pre/COVID/post), and SHAP cross-validation of the
attention maps.
