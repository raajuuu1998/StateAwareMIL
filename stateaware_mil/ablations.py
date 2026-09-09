"""State-Aware Interaction MIL ablation variants reported in the paper."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import GatedAttentionEncoder

VALID_ABLATIONS = {"full", "no_interaction", "no_auxiliary", "binary_joint"}


class StateAwareAblation(nn.Module):
    def __init__(self, in_dim: int, variant: str, hidden_dim: int = 256, attention_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        variant = variant.lower()
        if variant not in VALID_ABLATIONS:
            raise ValueError(f"Unknown ablation '{variant}'. Expected one of {sorted(VALID_ABLATIONS)}")
        self.variant = variant
        self.a_encoder = GatedAttentionEncoder(in_dim, hidden_dim, attention_dim, dropout)
        self.b_encoder = GatedAttentionEncoder(in_dim, hidden_dim, attention_dim, dropout)
        self.a_head = nn.Linear(hidden_dim, 1)
        self.b_head = nn.Linear(hidden_dim, 1)
        if variant != "no_interaction":
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
        if self.variant == "no_interaction":
            i = torch.zeros_like(a)
        else:
            u = torch.cat([za, zb, za * zb, torch.abs(za - zb)], dim=-1)
            i = self.interaction(u).squeeze()
        joint_logit = a + b + i
        state_logits = torch.stack([torch.zeros_like(a), b, a, joint_logit])
        return {
            "state_logits": state_logits,
            "joint_logit": joint_logit,
            "a_logit": a,
            "b_logit": b,
            "interaction_logit": i,
            "za": za,
            "zb": zb,
        }
