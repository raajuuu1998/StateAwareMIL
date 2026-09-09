"""Attention-based MIL building blocks used throughout the paper."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedAttentionEncoder(nn.Module):
    """Gated-attention MIL encoder following Ilse et al.

    Parameters match the final ASCI experiments: a linear projection to a
    256-dimensional tile representation followed by 128-dimensional gated
    attention.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, attention_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.v = nn.Linear(hidden_dim, attention_dim)
        self.u = nn.Linear(hidden_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1, bias=False)

    def attention_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        scores = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1)
        return h, scores

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        _, scores = self.attention_logits(x)
        return torch.softmax(scores, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, scores = self.attention_logits(x)
        weights = torch.softmax(scores, dim=0)
        return torch.sum(weights.unsqueeze(-1) * h, dim=0)
