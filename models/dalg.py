"""
models/dalg.py
==============
Challenger C2 — DALG: Dual-Attention LSTM-GRU.

This is the refined contribution of the project. Two complementary attention
mechanisms wrap a stacked LSTM->GRU recurrent core:

    1. FeatureAttention  — learns which *input features* matter at each step
                           (e.g. RSI vs volatility vs momentum), applied before
                           the recurrent layers.
    2. LSTM -> GRU core  — the LSTM captures long memory; the GRU refines it
                           with a lighter gating that trains faster.
    3. TemporalAttention — pools the GRU outputs over time by learned weights,
                           highlighting the most informative timesteps.

Both attention maps are exposed for explainability (SHAP-style feature
importance and attention-over-time visualisations), which is central to the
thesis question on model trustworthiness.

Author: Clerence Mashile — ELTE Budapest
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


class FeatureAttention(nn.Module):
    """Per-timestep gating over input features (channel attention)."""

    def __init__(self, n_features):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Tanh(),
            nn.Linear(n_features, n_features),
        )

    def forward(self, x):                       # x: (B, T, F)
        scores = self.gate(x)                   # (B, T, F)
        weights = torch.softmax(scores, dim=-1) # weights across features
        return x * weights, weights


class TemporalAttention(nn.Module):
    """Additive attention pooling over the time axis."""

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h):                       # h: (B, T, H)
        e = self.v(torch.tanh(self.proj(h)))    # (B, T, 1)
        a = F.softmax(e, dim=1)                 # (B, T, 1)
        context = (a * h).sum(dim=1)            # (B, H)
        return context, a.squeeze(-1)


class DALG(nn.Module):
    def __init__(self, n_features, cfg=CFG):
        super().__init__()
        mc = cfg.model
        self.feat_attn = FeatureAttention(n_features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=mc.dalg_lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_size=mc.dalg_lstm_hidden,
            hidden_size=mc.dalg_gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.temp_attn = TemporalAttention(mc.dalg_gru_hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(mc.dalg_gru_hidden),
            nn.Dropout(mc.dropout),
            nn.Linear(mc.dalg_gru_hidden, 1),
        )

    def forward(self, x, return_attn=False):    # x: (B, T, F)
        xf, feat_w = self.feat_attn(x)          # (B, T, F)
        h_lstm, _ = self.lstm(xf)               # (B, T, H_l)
        h_gru, _ = self.gru(h_lstm)             # (B, T, H_g)
        context, time_w = self.temp_attn(h_gru) # (B, H_g)
        out = self.head(context)                # (B, 1)
        if return_attn:
            return out, {"feature": feat_w, "temporal": time_w}
        return out


def build_dalg(n_features, cfg=CFG) -> DALG:
    return DALG(n_features, cfg=cfg)
