"""State-Aware Interaction MIL."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import GatedAttentionEncoder


class StateAwareInteractionMIL(nn.Module):
    """Biomarker-specific MIL branches with explicit representation interaction.

    The four structured state logits are
        q = [0, b, a, a + b + i]
    where a and b are biomarker-specific logits and i is the learned
    interaction logit.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, attention_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.a_encoder = GatedAttentionEncoder(in_dim, hidden_dim, attention_dim, dropout)
        self.b_encoder = GatedAttentionEncoder(in_dim, hidden_dim, attention_dim, dropout)
        self.a_head = nn.Linear(hidden_dim, 1)
        self.b_head = nn.Linear(hidden_dim, 1)
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        za = self.a_encoder(x)
        zb = self.b_encoder(x)
        a = self.a_head(za).squeeze()
        b = self.b_head(zb).squeeze()
        interaction_features = torch.cat([za, zb, za * zb, torch.abs(za - zb)], dim=-1)
        i = self.interaction(interaction_features).squeeze()
        state_logits = torch.stack([torch.zeros_like(a), b, a, a + b + i])
        return {
            "state_logits": state_logits,
            "a_logit": a,
            "b_logit": b,
            "interaction_logit": i,
            "za": za,
            "zb": zb,
        }
