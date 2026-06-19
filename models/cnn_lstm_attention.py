"""
models/cnn_lstm_attention.py
============================
Challenger C1 — CNN-LSTM with temporal attention.

Architecture (mirrors the hybrid family in the literature review):

    Input (B, T, F)
      -> Conv1d over time  : local pattern extraction across the window
      -> LSTM              : long-range temporal dependencies
      -> Additive attention: weight informative timesteps
      -> Linear head       : next-step return

The 1-D convolution acts as a learnable smoothing / pattern detector before the
recurrent layer, and the attention pooling replaces "take the last hidden state"
with a soft, content-based weighting over all timesteps.

Author: Clerence Mashile — ELTE Budapest
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


class CombinedAttention(nn.Module):
    """Additive (Bahdanau-style) attention pooling over the time axis."""

    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, h):                       # h: (B, T, H)
        w = self.score(h)                       # (B, T, 1)
        a = F.softmax(w, dim=1)                 # attention weights over time
        context = (a * h).sum(dim=1)            # (B, H)
        return context, a.squeeze(-1)


class CNNLSTMAttention(nn.Module):
    def __init__(self, n_features, cfg=CFG):
        super().__init__()
        mc = cfg.model
        pad = mc.cnn_kernel // 2
        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=mc.cnn_channels,
            kernel_size=mc.cnn_kernel,
            padding=pad,
        )
        self.act = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=mc.cnn_channels,
            hidden_size=mc.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.attn = CombinedAttention(mc.hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(mc.hidden_size),
            nn.Dropout(mc.dropout),
            nn.Linear(mc.hidden_size, 1),
        )

    def forward(self, x, return_attn=False):    # x: (B, T, F)
        z = x.transpose(1, 2)                   # (B, F, T) for Conv1d
        z = self.act(self.conv(z))              # (B, C, T)
        z = z.transpose(1, 2)                   # (B, T, C)
        h, _ = self.lstm(z)                     # (B, T, H)
        context, attn = self.attn(h)            # (B, H)
        out = self.head(context)                # (B, 1)
        if return_attn:
            return out, attn
        return out


def build_cnn_lstm(n_features, cfg=CFG) -> CNNLSTMAttention:
    return CNNLSTMAttention(n_features, cfg=cfg)
