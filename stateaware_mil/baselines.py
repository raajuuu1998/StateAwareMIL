"""Comparison methods reported in the paper."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .attention import GatedAttentionEncoder


class DirectJoint(nn.Module):
    """Binary ABMIL model trained directly on the joint-positive endpoint."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, attention_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.encoder = GatedAttentionEncoder(in_dim, hidden_dim, attention_dim, dropout)
        self.joint_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {"joint_logit": self.joint_head(z).squeeze(), "z": z}


class IndependentPair(nn.Module):
    """Two independently parameterized biomarker MIL branches.

    The joint-positive probability is computed downstream as p(A=1) * p(B=1),
    exactly as in the final experiments.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, attention_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.a_encoder = GatedAttentionEncoder(in_dim, hidden_dim, attention_dim, dropout)
        self.b_encoder = GatedAttentionEncoder(in_dim, hidden_dim, attention_dim, dropout)
        self.a_head = nn.Linear(hidden_dim, 1)
        self.b_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        za = self.a_encoder(x)
        zb = self.b_encoder(x)
        return {
            "a_logit": self.a_head(za).squeeze(),
            "b_logit": self.b_head(zb).squeeze(),
            "za": za,
            "zb": zb,
        }


class NaiveMTL(nn.Module):
    """Shared ABMIL representation with biomarker and joint prediction heads."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, attention_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.encoder = GatedAttentionEncoder(in_dim, hidden_dim, attention_dim, dropout)
        self.a_head = nn.Linear(hidden_dim, 1)
        self.b_head = nn.Linear(hidden_dim, 1)
        self.joint_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {
            "a_logit": self.a_head(z).squeeze(),
            "b_logit": self.b_head(z).squeeze(),
            "joint_logit": self.joint_head(z).squeeze(),
            "z": z,
        }


def probability_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def posthoc_features(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Return biomarker logits used by PostHoc-LR."""
    return np.column_stack([probability_logit(p_a), probability_logit(p_b)])


def build_posthoc_lr(c: float = 1.0, max_iter: int = 2000, random_state: int = 42):
    """Class-balanced logistic regression used for PostHoc-LR."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=max_iter,
            random_state=random_state,
        ),
    )
