# Reproducing the manuscript experiments

The final study evaluates two cohorts and two frozen pathology foundation-model representations:

- TCGA-COAD+READ: BRAF/MSI
- TCGA-LUAD: EGFR/TP53
- UNI2-h
- CONCH

For each cohort × representation combination, run the main comparison methods, State-Aware, FourState-ABMIL, and component ablations.

## 1. Main comparison table

For each configuration, run:

```bash
python -m experiments.run_baselines --config <CONFIG> --fm <FM> \
  --method all --manifest <MANIFEST> --embedding-dir <EMBEDDINGS> --output-dir outputs

python -m experiments.run_state_aware --config <CONFIG> --fm <FM> \
  --manifest <MANIFEST> --embedding-dir <EMBEDDINGS> --output-dir outputs
```

For LUAD, add `--fold-file <FOLD_CSV>` when the manifest does not already contain the fold assignments.

Each method writes:

```text
fold0_best.pt ... fold4_best.pt
fold0_history.csv ... fold4_history.csv
fold0_val_predictions.csv ... fold4_val_predictions.csv
fold0_predictions.csv ... fold4_predictions.csv
fold_metrics.csv
oof_predictions.csv
summary_metrics.csv
pooled_oof_metrics.csv
```

The manuscript reports `AUROC_mean ± AUROC_sd` and `AP_mean ± AP_sd` from `summary_metrics.csv`.

## 2. FourState-ABMIL comparison

```bash
python -m experiments.run_fourstate_abmil --config <CONFIG> --fm <FM> \
  --manifest <MANIFEST> --embedding-dir <EMBEDDINGS> --output-dir outputs
```

FourState-ABMIL uses one gated-attention slide encoder followed by an unconstrained four-class linear head, optimized with the same class-weighted state cross-entropy used by State-Aware.

## 3. Component ablation

```bash
python -m experiments.run_ablations --config <CONFIG> --fm <FM> \
  --variant all --manifest <MANIFEST> --embedding-dir <EMBEDDINGS> --output-dir outputs
```

Variants:

- `no_interaction`: sets the learned interaction term to zero;
- `no_auxiliary`: removes biomarker-specific auxiliary BCE losses;
- `binary_joint`: replaces the four-state objective with weighted binary joint-positive supervision while retaining auxiliary biomarker losses.

## 4. Four-state molecular separation

The manuscript uses out-of-fold State-Aware UNI2-h scores:

```bash
python analysis/four_state_separation.py \
  --predictions outputs/<COHORT>/StateAware/UNI2/oof_predictions.csv \
  --output-dir analysis_outputs/<COHORT>/four_state
```

The script computes:

- Kruskal-Wallis across states 00/01/10/11;
- one-sided Mann-Whitney U for `11 > 00`, `11 > 01`, and `11 > 10`;
- Benjamini-Hochberg correction across the three pairwise comparisons within a cohort;
- probability of superiority `P_sup = U / (n_11 * n_comparison)`.

## 5. ROC and precision-recall curves

```bash
python analysis/plot_roc_pr.py \
  --root outputs/<COHORT> --fm UNI2 --prefix <crc_or_luad> \
  --output-dir analysis_outputs/curves
```

The precision-recall figure includes a horizontal prevalence reference line.

## 6. Spatial attention

The paper visualizes one held-out case from each cohort using UNI2-h. The State-Aware joint attention is

```text
sqrt(alpha_A * alpha_B)
```

followed by normalization over tissue tiles. Percentile scaling and spatial smoothing are visualization-only transformations.
