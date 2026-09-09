"""Evaluation utilities for the reported joint-positive endpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def joint_metrics(y, p) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {
        "N": int(len(y)),
        "Pos": int(y.sum()),
        "Prevalence": float(y.mean()),
        "AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "AP": float(average_precision_score(y, p)) if y.sum() > 0 else np.nan,
    }


def summarize_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "N": int(fold_metrics["N"].sum()),
            "Pos": int(fold_metrics["Pos"].sum()),
            "Prevalence": float(fold_metrics["Pos"].sum() / fold_metrics["N"].sum()),
            "AUROC_mean": float(fold_metrics["AUROC"].mean()),
            "AUROC_sd": float(fold_metrics["AUROC"].std(ddof=1)),
            "AP_mean": float(fold_metrics["AP"].mean()),
            "AP_sd": float(fold_metrics["AP"].std(ddof=1)),
        }
    ])
