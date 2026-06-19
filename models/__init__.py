"""Model package: baseline RNN and the two challengers (C1, C2)."""
from config import CFG
from models.baseline_lstm import BaselineRNN, build_baseline
from models.cnn_lstm_attention import (
    CNNLSTMAttention, CombinedAttention, build_cnn_lstm,
)
from models.dalg import DALG, FeatureAttention, TemporalAttention, build_dalg


def build_model(name: str, n_features: int, cfg=CFG, cell: str | None = None):
    """Factory: map a model name to its builder.

    name in {"baseline", "cnn_lstm", "dalg"}.
    `cell` only applies to the baseline (lstm/gru/rnn).
    """
    name = name.lower()
    if name == "baseline":
        return build_baseline(n_features, cfg=cfg, cell=cell)
    if name in ("cnn_lstm", "cnnlstm", "c1"):
        return build_cnn_lstm(n_features, cfg=cfg)
    if name in ("dalg", "c2"):
        return build_dalg(n_features, cfg=cfg)
    raise ValueError(f"Unknown model '{name}'. Choose baseline | cnn_lstm | dalg.")


DISPLAY_NAMES = {
    "baseline": "Baseline LSTM",
    "cnn_lstm": "CNN-LSTM-Attention (C1)",
    "dalg": "DALG (C2)",
}

__all__ = [
    "BaselineRNN", "build_baseline",
    "CNNLSTMAttention", "CombinedAttention", "build_cnn_lstm",
    "DALG", "FeatureAttention", "TemporalAttention", "build_dalg",
    "build_model", "DISPLAY_NAMES",
]
