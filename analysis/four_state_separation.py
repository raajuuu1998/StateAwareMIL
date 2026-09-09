"""Four-state molecular-separation analysis used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def analyze(predictions: pd.DataFrame):
    x = predictions.copy()
    if not {"state_code", "p_joint"}.issubset(x.columns):
        raise ValueError("Predictions must contain 'state_code' and 'p_joint'.")
    x["State"] = x["state_code"].map({0: "00", 1: "01", 2: "10", 3: "11"})
    order = ["00", "01", "10", "11"]

    summary_rows = []
    groups = []
    for state in order:
        values = x.loc[x["State"] == state, "p_joint"].to_numpy(float)
        if len(values) == 0:
            raise ValueError(f"State {state} is absent from the predictions.")
        groups.append(values)
        summary_rows.append({
            "State": state,
            "N": len(values),
            "Mean": values.mean(),
            "SD": values.std(ddof=1) if len(values) > 1 else np.nan,
            "Median": np.median(values),
            "Q1": np.quantile(values, 0.25),
            "Q3": np.quantile(values, 0.75),
        })

    kw = kruskal(*groups)
    joint = x.loc[x["State"] == "11", "p_joint"].to_numpy(float)
    pairwise = []
    raw_p = []
    for state in ["00", "01", "10"]:
        other = x.loc[x["State"] == state, "p_joint"].to_numpy(float)
        u, p = mannwhitneyu(joint, other, alternative="greater")
        raw_p.append(p)
        pairwise.append({
            "Joint_state": "11",
            "Compared_state": state,
            "Joint_median": float(np.median(joint)),
            "Compared_median": float(np.median(other)),
            "U": float(u),
            "p_value": float(p),
            "P_sup": float(u / (len(joint) * len(other))),
        })
    adjusted = benjamini_hochberg(raw_p)
    for row, p_adj in zip(pairwise, adjusted):
        row["BH_adjusted_p"] = float(p_adj)

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(pairwise),
        pd.DataFrame([{"Kruskal_Wallis_H": float(kw.statistic), "p_value": float(kw.pvalue)}]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="State-Aware UNI2 OOF predictions CSV.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    summary, pairwise, global_test = analyze(predictions)
    summary.to_csv(out / "four_state_summary.csv", index=False)
    pairwise.to_csv(out / "four_state_pairwise_tests.csv", index=False)
    global_test.to_csv(out / "four_state_global_test.csv", index=False)

    print(global_test.to_string(index=False))
    print("\nState summary")
    print(summary.to_string(index=False))
    print("\n11 vs alternative states")
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
