# State-Aware Interaction MIL

### Rare joint molecular phenotype prediction from whole-slide histology

Code accompanying:

**“State-Aware Interaction MIL for Rare Joint Molecular Phenotype Prediction in Colorectal Cancer and Lung Adenocarcinoma.”**

State-Aware Interaction MIL is a weakly supervised multiple-instance learning framework for predicting **rare co-occurring molecular phenotypes** from whole-slide histopathology.

Most histology-based molecular prediction pipelines either predict biomarkers independently or define the joint-positive phenotype as a binary endpoint. Independent prediction does not explicitly model interactions between biomarker-specific histological representations, while binary joint prediction merges the double-negative and both single-positive configurations into one negative class.

State-Aware Interaction MIL addresses this by:

- learning **separate biomarker-specific slide representations**;
- explicitly modeling interactions between those representations;
- retaining the complete molecular configuration during training;
- distinguishing the four states `00`, `01`, `10`, and `11`;
- using state `11` as the joint-positive prediction target.

<p align="center">
  <img src="assets/architecture.png" width="860" alt="State-Aware Interaction MIL architecture">
</p>

---

# Motivation

Consider two binary biomarkers \(A\) and \(B\).

A patient can belong to one of four molecular configurations:

| State | Biomarker A | Biomarker B | Interpretation |
|---|---:|---:|---|
| `00` | 0 | 0 | Double negative |
| `01` | 0 | 1 | Biomarker B only |
| `10` | 1 | 0 | Biomarker A only |
| `11` | 1 | 1 | Joint positive |

A conventional binary formulation for joint prediction transforms these states into

```text
00 -> negative
01 -> negative
10 -> negative
11 -> positive
```

This discards the distinction between three biologically different non-joint configurations.

State-Aware Interaction MIL instead supervises all four states while retaining a dedicated joint-positive probability.

The molecular-state index is defined as

\[
y_s = 2A + B,
\]

giving

\[
00 \rightarrow 0,\qquad
01 \rightarrow 1,\qquad
10 \rightarrow 2,\qquad
11 \rightarrow 3.
\]

---

# Method

## Biomarker-specific MIL representations

Given a patient bag of frozen pathology foundation-model embeddings

\[
X=\{x_1,\ldots,x_N\},
\]

two separately parameterized gated-attention MIL branches are used:

\[
z_A = \mathrm{MIL}_A(X),
\qquad
z_B = \mathrm{MIL}_B(X).
\]

Each branch contains:

```text
Foundation-model embeddings
        ↓
256-D tile projection
        ↓
Gated attention
        ↓
256-D slide representation
```

The two branches therefore allow the model to learn different histological representations for biomarkers \(A\) and \(B\).

Biomarker-specific linear heads produce logits

\[
a=f_A(z_A),
\qquad
b=f_B(z_B).
\]

---

## Explicit biomarker interaction

The interaction module receives

\[
u =
[z_A,\;
 z_B,\;
 z_A \odot z_B,\;
 |z_A-z_B|].
\]

Because each slide representation is 256-dimensional,

\[
u\in\mathbb{R}^{1024}.
\]

The interaction network

\[
g:\mathbb{R}^{1024}
\rightarrow
\mathbb{R}^{256}
\rightarrow
\mathbb{R}
\]

produces a scalar interaction logit

\[
i=g(u).
\]

The interaction therefore combines:

- the original biomarker-A representation;
- the original biomarker-B representation;
- their elementwise product;
- their absolute difference.

---

## Structured molecular-state prediction

The four molecular-state logits are

\[
q =
[q_{00},q_{01},q_{10},q_{11}]
=
[0,\;b,\;a,\;a+b+i].
\]

The double-negative state is used as the reference category:

\[
q_{00}=0.
\]

The single-positive states use their corresponding biomarker logits:

\[
q_{01}=b,
\qquad
q_{10}=a.
\]

The joint-positive state combines both biomarker logits and their interaction:

\[
q_{11}=a+b+i.
\]

When the interaction is removed,

\[
i=0
\]

and therefore

\[
q_{11}=a+b.
\]

State probabilities are obtained using

\[
p=\mathrm{softmax}(q),
\]

and the final joint prediction is

\[
p_{\text{joint}}=p_{11}.
\]

---

## Training objective

The primary loss is class-weighted cross-entropy over the four molecular states:

\[
\mathcal{L}_{state}
=
-w_{y_s}\log p_{y_s}.
\]

For state \(c\),

\[
w_c =
\frac{N_{\text{train}}}{4N_c}.
\]

Each biomarker-specific branch also receives direct supervision through weighted binary cross-entropy losses

\[
\mathcal{L}_A
\quad\text{and}\quad
\mathcal{L}_B.
\]

The final objective is

\[
\mathcal{L}
=
\mathcal{L}_{state}
+
0.25
\left(
\frac{\mathcal{L}_A+\mathcal{L}_B}{2}
\right).
\]

---

# Study cohorts

Two joint molecular phenotype prediction tasks are evaluated.

| Cohort | Biomarker A | Biomarker B | Joint phenotype | Patients | Joint-positive cases | Prevalence |
|---|---|---|---|---:|---:|---:|
| TCGA-COAD+READ | BRAF | MSI | BRAF+/MSI+ | 443 | 30 | 6.8% |
| TCGA-LUAD | EGFR | TP53 | EGFR+/TP53+ | 428 | 37 | 8.6% |

The full four-state distributions are:

### TCGA-COAD+READ

| State | Count |
|---|---:|
| `00` | 375 |
| `01` | 27 |
| `10` | 11 |
| `11` | 30 |

### TCGA-LUAD

| State | Count |
|---|---:|
| `00` | 186 |
| `01` | 184 |
| `10` | 21 |
| `11` | 37 |

The low prevalence of state `11` motivates reporting **average precision (AP)** alongside AUROC.

---

# Pathology foundation-model representations

Experiments use frozen tile embeddings from two pathology foundation models.

| Representation | Embedding dimension |
|---|---:|
| UNI2-h | 1536 |
| CONCH | 512 |

Whole-slide images are processed using:

- 20× magnification;
- approximately 0.5 µm/pixel;
- tissue detection with Otsu thresholding;
- 224 × 224-pixel tiles.

During training, at most **3,000 tissue tiles per patient** are sampled.

During validation and testing, **all available embeddings are used**.

---

# Comparison methods

The repository implements all principal comparison methods used in the study.

## DirectJoint

A single gated-attention MIL model directly distinguishes the joint-positive phenotype from all other molecular states.

```text
00 + 01 + 10 -> negative
11           -> positive
```

## IndependentPair

Two separate biomarker-specific MIL models are trained independently.

Their marginal probabilities are combined as

\[
p_{\text{joint}} = p_A p_B.
\]

This baseline evaluates whether independently estimating the two biomarkers is sufficient for joint prediction.

## NaiveMTL

NaiveMTL uses one shared gated-attention slide representation with separate heads for:

- biomarker A;
- biomarker B;
- joint-positive status.

It provides multi-task supervision but does not explicitly maintain separate biomarker-specific slide representations or model their interaction.

## PostHoc-LR

PostHoc-LR uses predictions from IndependentPair.

Biomarker logits are standardized using training-set statistics and combined using class-balanced logistic regression.

The implementation uses:

```text
C = 1.0
solver = liblinear
max_iter = 2000
```

## FourState-ABMIL

FourState-ABMIL tests whether **four-state supervision alone** is sufficient.

It uses:

```text
Single gated-attention MIL encoder
                ↓
        4-class linear head
                ↓
          00 / 01 / 10 / 11
```

Unlike State-Aware Interaction MIL, FourState-ABMIL does not contain:

- separate biomarker-specific MIL branches;
- an explicit biomarker interaction module;
- biomarker-specific auxiliary heads.

---

# Main benchmark results

Performance is reported as **mean ± standard deviation across five outer test folds**.

## TCGA-COAD+READ — BRAF+/MSI+

### UNI2-h

| Method | AUROC | Average Precision |
|---|---:|---:|
| DirectJoint | 0.8998 ± 0.0580 | 0.4489 ± 0.1479 |
| IndependentPair | 0.9090 ± 0.0726 | 0.4864 ± 0.1146 |
| NaiveMTL | **0.9096 ± 0.0627** | 0.5161 ± 0.1531 |
| PostHoc-LR | 0.9093 ± 0.0776 | 0.4907 ± 0.1122 |
| **State-Aware** | 0.9080 ± 0.0965 | **0.5566 ± 0.2015** |

Joint-positive prevalence: **6.8%**

State-Aware achieved the largest AP among the comparison methods, while AUROC was similar to the leading baselines.

### CONCH

| Method | AUROC | Average Precision |
|---|---:|---:|
| DirectJoint | 0.8780 ± 0.0565 | 0.3932 ± 0.0658 |
| IndependentPair | 0.8372 ± 0.0605 | 0.3448 ± 0.0916 |
| NaiveMTL | 0.8269 ± 0.0317 | 0.3225 ± 0.0502 |
| PostHoc-LR | 0.8441 ± 0.0622 | 0.3430 ± 0.1020 |
| **State-Aware** | **0.8876 ± 0.0473** | **0.4410 ± 0.0446** |

With CONCH, State-Aware achieved the largest AUROC and AP among the evaluated comparison methods.

---

# TCGA-LUAD — EGFR+/TP53+

### UNI2-h

| Method | AUROC | Average Precision |
|---|---:|---:|
| DirectJoint | 0.6264 ± 0.1122 | 0.1928 ± 0.1262 |
| IndependentPair | 0.6842 ± 0.1476 | 0.2525 ± 0.1347 |
| NaiveMTL | 0.6485 ± 0.1724 | 0.2202 ± 0.1820 |
| PostHoc-LR | **0.6894 ± 0.1459** | 0.2177 ± 0.1224 |
| **State-Aware** | 0.6725 ± 0.1746 | **0.2784 ± 0.1714** |

Joint-positive prevalence: **8.6%**

State-Aware again achieved the largest AP, although PostHoc-LR produced the largest mean AUROC.

### CONCH

| Method | AUROC | Average Precision |
|---|---:|---:|
| DirectJoint | 0.5292 ± 0.1481 | 0.1226 ± 0.0728 |
| IndependentPair | 0.4926 ± 0.0869 | 0.1012 ± 0.0178 |
| NaiveMTL | 0.4996 ± 0.1063 | 0.1101 ± 0.0345 |
| PostHoc-LR | 0.4990 ± 0.1033 | 0.1049 ± 0.0219 |
| **State-Aware** | **0.5725 ± 0.1285** | **0.1659 ± 0.1078** |

State-Aware achieved the largest AUROC and AP among the evaluated comparison methods with CONCH.

---

# Summary of the main benchmark

Across the four cohort × representation combinations:

- State-Aware achieved the largest **average precision in all four settings** among DirectJoint, IndependentPair, NaiveMTL, and PostHoc-LR.
- With UNI2-h, AUROC remained close to the leading comparison methods but was not the largest.
- With CONCH, State-Aware achieved the largest mean AUROC and AP in both cohorts.
- The AP results should be interpreted in the context of the low joint-positive prevalence:
  - 6.8% in TCGA-COAD+READ;
  - 8.6% in TCGA-LUAD.

Complete numerical summaries are provided in:

[`results/main_results.csv`](results/main_results.csv)

---

# FourState-ABMIL comparison

FourState-ABMIL was added to determine whether simply replacing binary supervision with a conventional four-class MIL formulation could reproduce the State-Aware results.

| Cohort | Representation | Model | AUROC | AP |
|---|---|---|---:|---:|
| TCGA-COAD+READ | UNI2-h | FourState-ABMIL | 0.8978 ± 0.1122 | 0.4965 ± 0.2200 |
| TCGA-COAD+READ | UNI2-h | **State-Aware** | **0.9080 ± 0.0965** | **0.5566 ± 0.2015** |
| TCGA-COAD+READ | CONCH | FourState-ABMIL | 0.8532 ± 0.0404 | 0.3859 ± 0.1075 |
| TCGA-COAD+READ | CONCH | **State-Aware** | **0.8876 ± 0.0473** | **0.4410 ± 0.0446** |
| TCGA-LUAD | UNI2-h | FourState-ABMIL | 0.6542 ± 0.1565 | 0.2418 ± 0.1352 |
| TCGA-LUAD | UNI2-h | **State-Aware** | **0.6725 ± 0.1746** | **0.2784 ± 0.1714** |
| TCGA-LUAD | CONCH | FourState-ABMIL | 0.5550 ± 0.1169 | 0.1529 ± 0.0603 |
| TCGA-LUAD | CONCH | **State-Aware** | **0.5725 ± 0.1285** | **0.1659 ± 0.1078** |

State-Aware was above FourState-ABMIL in both AUROC and AP for all four cohort × representation comparisons.

This experiment indicates that **four-state supervision alone does not reproduce the complete State-Aware formulation**.

```text
FourState-ABMIL
    Single MIL representation
            +
      Four-class head

State-Aware Interaction MIL
    Biomarker-specific MIL branch A
            +
    Biomarker-specific MIL branch B
            +
      Explicit interaction
            +
   Structured four-state logits
            +
  Auxiliary biomarker supervision
```

Results are available in:

[`results/fourstate_abmil_results.csv`](results/fourstate_abmil_results.csv)

---

# Component ablation

Three components were evaluated.

## No interaction

The interaction term is removed:

\[
i=0,
\]

giving

\[
q=[0,b,a,a+b].
\]

## No auxiliary supervision

The auxiliary biomarker loss weight is set to zero:

\[
\lambda_{\mathrm{aux}}=0.
\]

## Binary joint supervision

The four-state objective is replaced by direct binary joint-positive supervision while retaining biomarker-specific auxiliary losses.

---

## TCGA-COAD+READ ablation

| Variant | UNI2-h AUROC | UNI2-h AP | CONCH AUROC | CONCH AP |
|---|---:|---:|---:|---:|
| **Full State-Aware** | **0.9080 ± 0.0965** | **0.5566 ± 0.2015** | **0.8876 ± 0.0473** | **0.4410 ± 0.0446** |
| No interaction | 0.9038 ± 0.0927 | 0.5178 ± 0.1568 | 0.8281 ± 0.0703 | 0.3039 ± 0.0993 |
| No auxiliary supervision | 0.9009 ± 0.1002 | 0.4604 ± 0.1346 | 0.8611 ± 0.0516 | 0.3604 ± 0.0900 |
| Binary joint supervision | 0.8786 ± 0.0810 | 0.4542 ± 0.1835 | 0.8546 ± 0.0594 | 0.3920 ± 0.1024 |

For TCGA-COAD+READ, removing either interaction modeling or auxiliary supervision reduced AP with both representations.

Replacing four-state supervision with binary joint supervision reduced both AUROC and AP for UNI2-h and CONCH.

---

## TCGA-LUAD ablation

| Variant | UNI2-h AUROC | UNI2-h AP | CONCH AUROC | CONCH AP |
|---|---:|---:|---:|---:|
| **Full State-Aware** | 0.6725 ± 0.1746 | **0.2784 ± 0.1714** | 0.5725 ± 0.1285 | 0.1659 ± 0.1078 |
| No interaction | **0.7107 ± 0.1035** | 0.2604 ± 0.1296 | 0.5561 ± 0.1510 | **0.1882 ± 0.1128** |
| No auxiliary supervision | 0.6467 ± 0.1849 | 0.2553 ± 0.1704 | **0.5923 ± 0.1406** | 0.1665 ± 0.0895 |
| Binary joint supervision | 0.6455 ± 0.0969 | 0.2079 ± 0.1613 | 0.4872 ± 0.1093 | 0.1063 ± 0.0332 |

The interaction and auxiliary components showed representation-dependent effects in TCGA-LUAD.

In contrast, replacing structured four-state supervision with binary joint supervision reduced both AUROC and AP for both UNI2-h and CONCH.

Results are available in:

[`results/ablation_results.csv`](results/ablation_results.csv)

---

# Four-state molecular separation

The structured formulation also allows the model's joint-positive scores to be examined across the true molecular configurations.

The analysis uses out-of-fold State-Aware UNI2-h predictions.

For each cohort:

1. a Kruskal-Wallis test assesses overall score differences among states `00`, `01`, `10`, and `11`;
2. state `11` is compared separately with `00`, `01`, and `10` using one-sided Mann-Whitney U tests;
3. the three pairwise tests are corrected using Benjamini-Hochberg adjustment;
4. \(P_{\mathrm{sup}}\) reports the probability that a randomly selected joint-positive case receives a larger joint-positive score than a randomly selected case from the comparison state.

## TCGA-COAD+READ

Overall four-state difference:

\[
H=96.19,
\qquad
p=1.02\times10^{-20}.
\]

Median joint-positive probabilities:

| True state | Median \(p_{\text{joint}}\) |
|---|---:|
| `00` | 0.0011 |
| `01` | 0.1628 |
| `10` | 0.0868 |
| `11` | **0.2758** |

Pairwise comparisons:

| Comparison | \(P_{\mathrm{sup}}\) | BH-adjusted p |
|---|---:|---:|
| `11` > `00` | 0.943 | 8.90 × 10⁻¹⁶ |
| `11` > `01` | 0.637 | 0.040 |
| `11` > `10` | 0.682 | 0.040 |

The BRAF+/MSI+ group had an elevated median joint score relative to all three alternative states, with all three comparisons remaining significant after correction.

## TCGA-LUAD

Overall four-state difference:

\[
H=20.75,
\qquad
p=1.19\times10^{-4}.
\]

Median joint-positive probabilities:

| True state | Median \(p_{\text{joint}}\) |
|---|---:|
| `00` | 0.0307 |
| `01` | 0.0415 |
| `10` | 0.0590 |
| `11` | **0.0651** |

Pairwise comparisons:

| Comparison | \(P_{\mathrm{sup}}\) | BH-adjusted p |
|---|---:|---:|
| `11` > `00` | 0.713 | 6.30 × 10⁻⁵ |
| `11` > `01` | 0.653 | 0.00248 |
| `11` > `10` | 0.559 | 0.231 |

The EGFR+/TP53+ group was separated from the double-negative and TP53-only states, while its scores overlapped with the EGFR-only group.

Detailed results are available in:

[`results/four_state_separation.csv`](results/four_state_separation.csv)

---

# Spatial attention analysis

Attention visualization is included for held-out cases from both cohorts:

| Cohort | Held-out case |
|---|---|
| TCGA-COAD+READ | `TCGA-CK-6746` |
| TCGA-LUAD | `TCGA-64-1681` |

For DirectJoint, the attention distribution from the single MIL encoder is visualized directly.

For State-Aware, the two biomarker-specific attention distributions are combined using

\[
\alpha_i^{joint}
\propto
\sqrt{
\alpha_{A,i}
\alpha_{B,i}
}.
\]

The resulting distribution is normalized across tissue tiles.

For visualization only:

- attention weights are converted to percentile ranks;
- spatial smoothing is applied to the displayed heatmap;
- neither operation changes model predictions.

In the held-out TCGA-COAD+READ case, both models attended to overlapping tumor-containing areas, while State-Aware concentrated attention within more localized regions.

In the held-out TCGA-LUAD case, DirectJoint distributed attention across a wider portion of the tissue, whereas State-Aware concentrated attention within a more localized tumor-bearing region.

These visualizations examine how the prediction formulation changes spatial attention and are not used as independent biological validation.

---

# Experimental protocol

| Setting | Value |
|---|---|
| Outer evaluation | 5 fixed patient-level folds |
| Outer stratification | Four-state molecular label |
| Inner validation | 15% of outer-training patients |
| Inner stratification | Four-state molecular label |
| Training tile cap | 3,000 tiles/patient |
| Validation tiles | All available embeddings |
| Test tiles | All available embeddings |
| Slide representation dimension | 256 |
| Attention dimension | 128 |
| Dropout | 0.10 |
| Optimizer | AdamW |
| Learning rate | \(1\times10^{-4}\) |
| Weight decay | \(1\times10^{-5}\) |
| Maximum epochs | 30 |
| Gradient accumulation | 4 patient bags |
| Checkpoint criterion | Validation AP |
| Early-stopping patience | 6 epochs |
| Auxiliary-loss weight | 0.25 |
| Primary evaluation metrics | AUROC and AP |
| Reported summary | Mean ± SD across five outer folds |

---

# Repository structure

```text
StateAwareMIL/
│
├── configs/
│   ├── crc_braf_msi.yaml
│   └── luad_egfr_tp53.yaml
│
├── stateaware_mil/
│   ├── attention.py
│   ├── state_aware.py
│   ├── baselines.py
│   ├── fourstate_abmil.py
│   ├── ablations.py
│   ├── data.py
│   ├── config.py
│   ├── training.py
│   └── evaluation.py
│
├── experiments/
│   ├── run_state_aware.py
│   ├── run_baselines.py
│   ├── run_fourstate_abmil.py
│   └── run_ablations.py
│
├── analysis/
│   ├── four_state_separation.py
│   ├── plot_roc_pr.py
│   └── attention_visualization.py
│
├── results/
│   ├── main_results.csv
│   ├── fourstate_abmil_results.csv
│   ├── ablation_results.csv
│   └── four_state_separation.csv
│
├── docs/
│   ├── data_setup.md
│   └── reproduction.md
│
├── tests/
│   ├── test_data.py
│   └── test_models.py
│
├── assets/
│   └── architecture.png
│
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
└── README.md
```

The implementation is shared across cohorts. CRC and LUAD differ through configuration, biomarker labels, manifests, and embedding locations rather than through separate duplicated model implementations.

---

# Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/raajuuu1998/StateAwareMIL.git
cd StateAwareMIL

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

For WSI attention visualization:

```bash
pip install -e ".[attention]"
```

The system OpenSlide library must also be available.

---

# Data preparation

This repository does not redistribute:

- TCGA whole-slide images;
- molecular labels;
- UNI2-h model weights;
- CONCH model weights;
- extracted patient embeddings.

These resources should be obtained according to their respective access conditions and licenses.

Each experiment requires:

1. a patient-level manifest;
2. binary labels for both biomarkers;
3. outer-fold assignments;
4. one embedding bag per patient.

## CRC manifest example

```text
patient_id,BRAF,MSI,fold
TCGA-XX-XXXX,0,0,0
TCGA-XX-XXXX,0,1,1
TCGA-XX-XXXX,1,0,2
TCGA-XX-XXXX,1,1,3
```

## LUAD manifest example

```text
patient_id,EGFR,TP53,fold
TCGA-XX-XXXX,0,0,0
TCGA-XX-XXXX,0,1,1
TCGA-XX-XXXX,1,0,2
TCGA-XX-XXXX,1,1,3
```

Embedding bags are expected as PyTorch `.pt` files with shape

```text
[number_of_tiles, embedding_dimension]
```

where

```text
UNI2-h -> 1536 dimensions
CONCH  -> 512 dimensions
```

See [`docs/data_setup.md`](docs/data_setup.md) for detailed preparation instructions.

---

# Running State-Aware Interaction MIL

## CRC · BRAF/MSI · UNI2-h

```bash
python -m experiments.run_state_aware \
  --config configs/crc_braf_msi.yaml \
  --fm uni2 \
  --manifest /path/to/crc_manifest.csv \
  --embedding-dir /path/to/crc/uni2h \
  --output-dir outputs
```

## LUAD · EGFR/TP53 · CONCH

```bash
python -m experiments.run_state_aware \
  --config configs/luad_egfr_tp53.yaml \
  --fm conch \
  --manifest /path/to/luad_manifest.csv \
  --fold-file /path/to/luad_folds.csv \
  --embedding-dir /path/to/luad/conch \
  --output-dir outputs
```

---

# Running the main baselines

```bash
python -m experiments.run_baselines \
  --config configs/crc_braf_msi.yaml \
  --fm uni2 \
  --method all \
  --manifest /path/to/crc_manifest.csv \
  --embedding-dir /path/to/crc/uni2h \
  --output-dir outputs
```

Available methods:

```text
directjoint
independentpair
naivemtl
posthoc_lr
all
```

When `all` is used, IndependentPair is fitted before PostHoc-LR so that the latter can use the corresponding biomarker-specific predictions.

---

# Running FourState-ABMIL

```bash
python -m experiments.run_fourstate_abmil \
  --config configs/luad_egfr_tp53.yaml \
  --fm conch \
  --manifest /path/to/luad_manifest.csv \
  --fold-file /path/to/luad_folds.csv \
  --embedding-dir /path/to/luad/conch \
  --output-dir outputs
```

---

# Running the component ablations

```bash
python -m experiments.run_ablations \
  --config configs/crc_braf_msi.yaml \
  --fm uni2 \
  --variant all \
  --manifest /path/to/crc_manifest.csv \
  --embedding-dir /path/to/crc/uni2h \
  --output-dir outputs
```

Available variants:

```text
full
no_interaction
no_auxiliary
binary_joint
all
```

---

# Four-state statistical analysis

```bash
python analysis/four_state_separation.py \
  --predictions outputs/CRC_BRAF_MSI/StateAware/UNI2/oof_predictions.csv \
  --output-dir analysis_outputs/crc_four_state
```

Outputs include:

```text
four_state_summary.csv
four_state_global_test.csv
four_state_pairwise_tests.csv
four_state_predictions.csv
```

---

# ROC and precision-recall curves

```bash
python analysis/plot_roc_pr.py \
  --root outputs/CRC_BRAF_MSI \
  --fm UNI2 \
  --prefix crc \
  --output-dir analysis_outputs/curves
```

The precision-recall plots include the observed joint-positive prevalence as a reference level.

---

# Real-WSI attention visualization

```bash
python analysis/attention_visualization.py \
  --slide /path/to/case.svs \
  --features /path/to/case_uni2h.pt \
  --direct-checkpoint /path/to/DirectJoint/UNI2/foldX_best.pt \
  --state-checkpoint /path/to/StateAware/UNI2/foldX_best.pt \
  --output analysis_outputs/case_attention
```

---

# Reproducing the paper

A table- and figure-oriented reproduction guide is available in:

[`docs/reproduction.md`](docs/reproduction.md)

The guide maps repository commands to:

- main benchmark experiments;
- FourState-ABMIL comparisons;
- component ablations;
- four-state molecular separation;
- ROC and precision-recall curves;
- spatial attention analysis.

Paper-level numerical outputs are available in:

[`results/`](results/)

---

# Citation

If you use this implementation, please cite the accompanying manuscript.

Citation information will be updated following publication.
