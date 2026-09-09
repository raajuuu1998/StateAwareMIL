# State-Aware Interaction MIL

Code accompanying **“State-Aware Interaction MIL for Rare Joint Molecular Phenotype Prediction in Colorectal Cancer and Lung Adenocarcinoma.”**

State-Aware Interaction MIL preserves biomarker-specific whole-slide representations, models their interaction explicitly, and supervises the complete four-state molecular configuration rather than collapsing all non-joint states into a single negative class.

<p align="center">
  <img src="assets/architecture.png" width="820" alt="State-Aware Interaction MIL architecture">
</p>

## Paper setup

The experiments evaluate two rare joint molecular phenotypes from frozen pathology foundation-model representations:

| Cohort | Joint phenotype | Patients | Joint-positive prevalence |
|---|---|---:|---:|
| TCGA-COAD+READ | BRAF+/MSI+ | 443 | 6.8% |
| TCGA-LUAD | EGFR+/TP53+ | 428 | 8.6% |

Two frozen representations are supported: **UNI2-h (1536-D)** and **CONCH (512-D)**.

### Main reported results

| Cohort | Representation | State-Aware AUROC | State-Aware AP |
|---|---|---:|---:|
| TCGA-COAD+READ | UNI2-h | 0.9080 ± 0.0965 | 0.5566 ± 0.2015 |
| TCGA-COAD+READ | CONCH | 0.8876 ± 0.0473 | 0.4410 ± 0.0446 |
| TCGA-LUAD | UNI2-h | 0.6725 ± 0.1746 | 0.2784 ± 0.1714 |
| TCGA-LUAD | CONCH | 0.5725 ± 0.1285 | 0.1659 ± 0.1078 |

Complete paper-level results are provided in [`results/`](results/).

## Method

For biomarkers \(A\) and \(B\), two separately parameterized gated-attention MIL branches produce slide representations \(z_A\) and \(z_B\). The interaction module uses

```text
[z_A, z_B, z_A * z_B, |z_A - z_B|]
```

and the structured four-state logits are

```text
q = [0, b, a, a + b + i]
```

corresponding to molecular states `00`, `01`, `10`, and `11`. The joint-positive prediction is `softmax(q)[3]`.

## Repository layout

```text
state-aware-interaction-mil/
├── configs/                    # CRC and LUAD experiment definitions
├── stateaware_mil/             # Models, data handling, training, evaluation
├── experiments/                # Main, baseline, FourState-ABMIL, and ablation runners
├── analysis/                   # Paper analyses and figure generation
├── results/                    # Paper-level summary tables
├── docs/                       # Data preparation and reproduction instructions
├── tests/                      # Lightweight implementation checks
└── assets/                     # Architecture figure
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For real-WSI attention visualization, install the optional OpenSlide dependency:

```bash
pip install -e ".[attention]"
```

The system OpenSlide library must also be installed for your operating system.

## Data preparation

The repository does **not** redistribute TCGA whole-slide images, molecular labels, pretrained foundation-model weights, or extracted patient embeddings. Prepare these resources according to their original access terms and licenses.

Each experiment requires:

1. a patient-level manifest containing `patient_id`, the two biomarker labels, and an outer `fold` assignment (or a separate fold CSV);
2. one `.pt` embedding bag per patient for UNI2-h or CONCH;
3. the exact five outer folds used for evaluation.

See [`docs/data_setup.md`](docs/data_setup.md) for the expected formats.

## Reproducing the experiments

Examples below assume the repository has been installed with `pip install -e .`.

### State-Aware Interaction MIL

```bash
python -m experiments.run_state_aware \
  --config configs/crc_braf_msi.yaml \
  --fm uni2 \
  --manifest /path/to/crc_manifest.csv \
  --embedding-dir /path/to/crc/uni2h \
  --output-dir outputs
```

### Main baselines

The paper compares against **DirectJoint**, **IndependentPair**, **NaiveMTL**, and **PostHoc-LR**.

```bash
python -m experiments.run_baselines \
  --config configs/crc_braf_msi.yaml \
  --fm uni2 \
  --method all \
  --manifest /path/to/crc_manifest.csv \
  --embedding-dir /path/to/crc/uni2h \
  --output-dir outputs
```

`PostHoc-LR` uses the fitted IndependentPair model from the same fold. Running `--method all` handles this dependency in the correct order.

### FourState-ABMIL

```bash
python -m experiments.run_fourstate_abmil \
  --config configs/luad_egfr_tp53.yaml \
  --fm conch \
  --manifest /path/to/luad_manifest.csv \
  --fold-file /path/to/luad_folds.csv \
  --embedding-dir /path/to/luad/conch \
  --output-dir outputs
```

### Component ablations

```bash
python -m experiments.run_ablations \
  --config configs/crc_braf_msi.yaml \
  --fm uni2 \
  --variant all \
  --manifest /path/to/crc_manifest.csv \
  --embedding-dir /path/to/crc/uni2h \
  --output-dir outputs
```

The reported variants are `no_interaction`, `no_auxiliary`, and `binary_joint`, with `full` included as a reference.

## Experimental protocol preserved in this implementation

- five fixed patient-level outer folds;
- 15% inner validation split stratified by the four-state molecular label;
- deterministic sampling of up to 3,000 training tiles per patient;
- all available embeddings used for validation and testing;
- 256-D tile projection and 128-D gated attention;
- dropout 0.10;
- AdamW, learning rate `1e-4`, weight decay `1e-5`;
- maximum 30 epochs;
- gradient accumulation over 4 patient bags;
- checkpoint selection by validation AP;
- early stopping after 6 epochs without AP improvement;
- State-Aware auxiliary biomarker-loss weight 0.25;
- AUROC and average precision reported as mean ± SD across outer folds.

## Paper analyses

Four-state molecular separation:

```bash
python analysis/four_state_separation.py \
  --predictions outputs/CRC_BRAF_MSI/StateAware/UNI2/oof_predictions.csv \
  --output-dir analysis_outputs/crc_four_state
```

ROC and precision-recall curves:

```bash
python analysis/plot_roc_pr.py \
  --root outputs/CRC_BRAF_MSI \
  --fm UNI2 \
  --prefix crc \
  --output-dir analysis_outputs/curves
```

Real-WSI attention visualization:

```bash
python analysis/attention_visualization.py \
  --slide /path/to/case.svs \
  --features /path/to/case_uni2h.pt \
  --direct-checkpoint /path/to/DirectJoint/UNI2/foldX_best.pt \
  --state-checkpoint /path/to/StateAware/UNI2/foldX_best.pt \
  --output analysis_outputs/case_attention
```

For State-Aware, joint attention is the normalized geometric mean of the two biomarker-specific attention distributions. Percentile scaling and spatial smoothing are applied only for visualization.
