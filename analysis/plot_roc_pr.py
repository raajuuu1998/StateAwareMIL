"""Generate the UNI2 ROC and precision-recall curves shown in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

METHODS = [
    ("DirectJoint", "DirectJoint"),
    ("IndependentPair", "IndependentPair"),
    ("PostHocLR", "PostHoc-LR"),
    ("NaiveMTL", "NaiveMTL"),
    ("StateAware", "StateAware"),
]


def load_fold_metrics(root: Path, method: str, fm_folder: str):
    return pd.read_csv(root / method / fm_folder / "fold_metrics.csv")


def plot_roc(root: Path, fm_folder: str, output: Path):
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for method, label in METHODS:
        tprs = []
        for fold in range(5):
            x = pd.read_csv(root / method / fm_folder / f"fold{fold}_predictions.csv")
            fpr, tpr, _ = roc_curve(x["joint"], x["p_joint"])
            interp = np.interp(grid, fpr, tpr)
            interp[0], interp[-1] = 0.0, 1.0
            tprs.append(interp)
        metrics = load_fold_metrics(root, method, fm_folder)
        ax.plot(grid, np.vstack(tprs).mean(axis=0), linewidth=2.8 if method == "StateAware" else 1.8,
                label=f"{label} ({metrics['AUROC'].mean():.3f} ± {metrics['AUROC'].std(ddof=1):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="No-skill")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pr(root: Path, fm_folder: str, output: Path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    prevalence = None
    for method, label in METHODS:
        x = pd.read_csv(root / method / fm_folder / "oof_predictions.csv")
        precision, recall, _ = precision_recall_curve(x["joint"], x["p_joint"])
        ax.step(recall, precision, where="post", linewidth=2.8 if method == "StateAware" else 1.8, label=label)
        if method == "StateAware":
            prevalence = float(x["joint"].mean())
    ax.axhline(prevalence, linestyle="--", linewidth=1.5, label=f"No-skill ({prevalence:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Cohort output root, e.g. outputs/CRC_BRAF_MSI.")
    parser.add_argument("--fm", default="UNI2", choices=["UNI2", "CONCH"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True, help="Filename prefix, e.g. crc or luad.")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_roc(root, args.fm, out / f"{args.prefix}_roc_curves")
    plot_pr(root, args.fm, out / f"{args.prefix}_pr_curves")
    print("Saved ROC and precision-recall figures to", out)


if __name__ == "__main__":
    main()
