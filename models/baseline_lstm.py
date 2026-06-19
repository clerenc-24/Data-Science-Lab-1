"""
models/baseline_lstm.py
=======================
Baseline recurrent model — the reference point every challenger must beat.

A stacked RNN (LSTM by default; GRU or vanilla RNN selectable) reads the
look-back window and a linear head maps the final hidden state to a single
next-step return. This is deliberately the simplest sensible sequence model:
in 21 of the 24 papers surveyed for this project, a vanilla LSTM/ARIMA is the
baseline, so we reproduce that convention.

Author: Clerence Mashile — ELTE Budapest
"""

import torch
import torch.nn as nn

from config import CFG


class BaselineRNN(nn.Module):
    def __init__(self, n_features, cfg=CFG, cell: str | None = None):
        super().__init__()
        mc = cfg.model
        cell = (cell or mc.baseline_cell).lower()
        self.cell = cell

        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[cell]
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=mc.hidden_size,
            num_layers=mc.baseline_layers,
            batch_first=True,
            dropout=mc.dropout if mc.baseline_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(mc.hidden_size),
            nn.Dropout(mc.dropout),
            nn.Linear(mc.hidden_size, 1),
        )

    def forward(self, x):                      # x: (B, T, F)
        out, _ = self.rnn(x)                   # out: (B, T, H)
        last = out[:, -1, :]                   # final timestep
        return self.head(last)                 # (B, 1)


def build_baseline(n_features, cfg=CFG, cell: str | None = None) -> BaselineRNN:
    return BaselineRNN(n_features, cfg=cfg, cell=cell)
