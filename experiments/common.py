"""Shared experiment setup for command-line runners."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stateaware_mil.config import fm_display_name, fm_key, load_config
from stateaware_mil.data import EmbeddingStore, prepare_manifest, save_splits
from stateaware_mil.training import TrainingSettings


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Cohort YAML configuration.")
    parser.add_argument("--fm", required=True, choices=["uni2", "conch"], help="Frozen foundation-model representation.")
    parser.add_argument("--manifest", default=None, help="Override manifest CSV path from the YAML config.")
    parser.add_argument("--fold-file", default=None, help="Override optional fold CSV path from the YAML config.")
    parser.add_argument("--embedding-dir", action="append", default=None, help="Embedding directory; may be passed multiple times.")
    parser.add_argument("--output-dir", default="outputs", help="Root output directory.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--workers", type=int, default=8, help="CPU workers used to preload embedding bags.")
    parser.add_argument("--force", action="store_true", help="Re-run completed folds instead of reusing saved outputs.")
    return parser


def prepare_run(args):
    cfg = load_config(args.config)
    fm = fm_key(args.fm)
    fm_cfg = cfg["foundation_models"][fm]
    data_cfg = cfg.get("data", {})

    manifest = args.manifest or data_cfg.get("manifest")
    if manifest is None:
        raise ValueError("A manifest path is required via --manifest or config:data:manifest.")
    fold_file = args.fold_file if args.fold_file is not None else data_cfg.get("fold_file")
    embedding_dirs = args.embedding_dir or data_cfg.get("embeddings", {}).get(fm)
    if not embedding_dirs:
        raise ValueError(f"No {fm} embedding directories were provided.")

    df = prepare_manifest(
        manifest,
        cfg["biomarker_a"],
        cfg["biomarker_b"],
        fold_path=fold_file,
        expected_n=cfg.get("expected_patients"),
    )

    training_cfg = cfg.get("training", {})
    settings = TrainingSettings(
        seed=int(training_cfg.get("seed", 42)),
        hidden_dim=int(training_cfg.get("hidden_dim", 256)),
        attention_dim=int(training_cfg.get("attention_dim", 128)),
        dropout=float(training_cfg.get("dropout", 0.10)),
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
        max_epochs=int(training_cfg.get("max_epochs", 30)),
        patience=int(training_cfg.get("patience", 6)),
        accumulation_steps=int(training_cfg.get("gradient_accumulation", 4)),
        auxiliary_weight=float(training_cfg.get("auxiliary_weight", 0.25)),
        validation_fraction=float(training_cfg.get("validation_fraction", 0.15)),
        posthoc_c=float(training_cfg.get("posthoc_c", 1.0)),
        posthoc_max_iter=int(training_cfg.get("posthoc_max_iter", 2000)),
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    store = EmbeddingStore(
        df["patient_id"],
        fm=fm,
        directories=[Path(x) for x in embedding_dirs],
        input_dim=int(fm_cfg["dimension"]),
        tile_cap=training_cfg.get("train_tile_cap", 3000),
        seed=settings.seed,
    )
    store.preload(workers=args.workers)

    cohort_output = Path(args.output_dir) / cfg["output_name"]
    cohort_output.mkdir(parents=True, exist_ok=True)
    save_splits(df, cohort_output / "splits", cfg["biomarker_a"], cfg["biomarker_b"], settings.seed, settings.validation_fraction)
    return cfg, fm, fm_display_name(fm), df, store, int(fm_cfg["dimension"]), cohort_output, settings, device
