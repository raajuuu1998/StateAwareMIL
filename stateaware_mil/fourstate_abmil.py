"""Four-state ABMIL comparison baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import GatedAttentionEncoder


class FourStateABMIL(nn.Module):
    """Single gated-attention encoder with an unconstrained four-class head."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, attention_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.encoder = GatedAttentionEncoder(in_dim, hidden_dim, attention_dim, dropout)
        self.state_head = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.state_head(self.encoder(x))
