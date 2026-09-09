# Data setup

The training code consumes **patient-level frozen embedding bags**. It does not download TCGA data or run UNI2-h/CONCH feature extraction.

## Manifest format

For TCGA-COAD+READ, the manifest must contain:

```csv
patient_id,BRAF,MSI,fold
TCGA-XX-0001,0,0,0
TCGA-XX-0002,1,1,1
```

For TCGA-LUAD, use:

```csv
patient_id,EGFR,TP53,fold
TCGA-YY-0001,0,1,0
TCGA-YY-0002,1,1,1
```

If `fold` is stored in a separate CSV, that file must contain exactly:

```csv
patient_id,fold
TCGA-YY-0001,0
TCGA-YY-0002,1
```

The loader normalizes patient IDs to uppercase and constructs

```text
state_code = 2 * A + B
joint      = (state_code == 3)
```

so that `00`, `01`, `10`, and `11` map to state indices 0, 1, 2, and 3.

## Embedding files

Each patient is represented by a `.pt` file named with the patient ID, for example:

```text
embeddings/uni2h/TCGA-XX-0001.pt
embeddings/conch/TCGA-XX-0001.pt
```

The file may be either:

- a tensor of shape `[n_tiles, embedding_dim]`; or
- a dictionary containing `features` or `embeddings`.

Expected embedding dimensions are:

| Representation | Dimension |
|---|---:|
| UNI2-h | 1536 |
| CONCH | 512 |

The final experiments used 224 × 224 tissue tiles at 20× magnification (0.5 µm/pixel). During training, at most 3,000 tile embeddings are sampled per patient. Validation and testing use all available tile embeddings.

## Attention visualization

The WSI attention script requires the feature `.pt` file to contain both:

```python
{
    "features": tensor_of_shape_N_by_D,
    "coords": coordinates_of_shape_N_by_2,
}
```

Coordinates must correspond to the feature ordering used to train the model.

## Data distribution

TCGA WSIs, molecular annotations, extracted embeddings, and pretrained foundation-model weights are not redistributed in this repository. Users should obtain and process them under the terms of the original data/model providers.
