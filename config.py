"""
config.py
=========
Central configuration for the Forex ML project.

All hyperparameters, data settings, and evaluation options live here so that
experiments are reproducible and easy to sweep. Import `CFG` everywhere.

Author: Clerence Mashile — MSc Computer Science (Data Science), ELTE Budapest
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    # Instrument / data source -------------------------------------------------
    ticker: str = "EURUSD=X"          # yfinance symbol for EUR/USD spot
    start: str = "2014-01-01"
    end: str = "2024-12-31"
    interval: str = "1d"
    use_synthetic: bool = True         # yfinance is often blocked offline; fall back to GBM
    synthetic_n: int = 2600            # ~10 trading years of daily bars

    # Feature engineering ------------------------------------------------------
    seq_len: int = 30                  # look-back window (timesteps fed to the model)
    horizon: int = 1                   # predict the return `horizon` steps ahead
    # Technical indicator windows
    sma_windows: tuple = (5, 10, 20)
    rsi_window: int = 14
    vol_window: int = 10
    mom_window: int = 5

    # Splits (chronological, no shuffling — this is a time series) -------------
    train_frac: float = 0.70
    val_frac: float = 0.15             # test_frac = 1 - train - val


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0             # max global grad norm
    scheduler_patience: int = 4        # ReduceLROnPlateau patience (epochs)
    scheduler_factor: float = 0.5
    early_stop_patience: int = 10
    seed: int = 42
    device: str = "cpu"               # auto-upgraded to "cuda" in train.py if available


@dataclass
class ModelConfig:
    # Shared
    hidden_size: int = 64
    dropout: float = 0.2
    # Baseline RNN
    baseline_layers: int = 2
    baseline_cell: str = "lstm"       # one of {lstm, gru, rnn}
    # CNN-LSTM-Attention (C1)
    cnn_channels: int = 32
    cnn_kernel: int = 3
    # DALG (C2)
    dalg_lstm_hidden: int = 64
    dalg_gru_hidden: int = 64


@dataclass
class EvalConfig:
    # Annualisation factor for the Sharpe ratio (252 trading days)
    trading_days: int = 252
    # Directional accuracy uses sign(prediction) vs sign(realised return)
    results_dir: str = "results"
    metrics: List[str] = field(
        default_factory=lambda: ["rmse", "mae", "directional_accuracy", "sharpe"]
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# Single importable instance
CFG = Config()
