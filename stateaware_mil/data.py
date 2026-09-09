"""Patient manifests, folds, embedding bags, and deterministic tile sampling."""

from __future__ import annotations

import random
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .config import fm_key


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_manifest(
    manifest_path: str | Path,
    biomarker_a: str,
    biomarker_b: str,
    fold_path: str | Path | None = None,
    expected_n: int | None = None,
) -> pd.DataFrame:
    """Load a patient-level manifest and construct the four-state label.

    Required manifest columns are ``patient_id`` and the two biomarker labels.
    A ``fold`` column may be supplied in the manifest or in a separate fold CSV.
    """
    df = pd.read_csv(manifest_path)
    required = {"patient_id", biomarker_a, biomarker_b}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    df["patient_id"] = df["patient_id"].astype(str).str.upper().str.strip()

    if "fold" not in df.columns:
        if fold_path is None:
            raise ValueError("Manifest has no 'fold' column and no fold CSV was provided.")
        folds = pd.read_csv(fold_path)
        if not {"patient_id", "fold"}.issubset(folds.columns):
            raise ValueError("Fold CSV must contain 'patient_id' and 'fold'.")
        folds["patient_id"] = folds["patient_id"].astype(str).str.upper().str.strip()
        df = df.merge(folds[["patient_id", "fold"]], on="patient_id", how="left", validate="one_to_one")

    df[biomarker_a] = pd.to_numeric(df[biomarker_a]).astype(int)
    df[biomarker_b] = pd.to_numeric(df[biomarker_b]).astype(int)
    df["fold"] = pd.to_numeric(df["fold"]).astype(int)
    df["A"] = df[biomarker_a]
    df["B"] = df[biomarker_b]
    df["state_code"] = 2 * df["A"] + df["B"]
    df["joint"] = (df["state_code"] == 3).astype(int)

    if df["patient_id"].duplicated().any():
        dup = df.loc[df["patient_id"].duplicated(), "patient_id"].tolist()[:5]
        raise ValueError(f"Duplicate patient IDs found, e.g. {dup}")
    if df["fold"].isna().any():
        raise ValueError("At least one patient has no outer-fold assignment.")
    if not set(df["fold"].unique()).issubset({0, 1, 2, 3, 4}):
        raise ValueError("Expected outer folds numbered 0..4.")
    if expected_n is not None and len(df) != expected_n:
        raise ValueError(f"Expected {expected_n} patients but found {len(df)}.")

    return df.reset_index(drop=True)


def split_outer_fold(df: pd.DataFrame, fold: int, seed: int = 42, validation_fraction: float = 0.15):
    """Reproduce the paper's outer test fold and 15% inner validation split."""
    test = df[df["fold"] == fold].copy().reset_index(drop=True)
    dev = df[df["fold"] != fold].copy().reset_index(drop=True)
    tr_idx, va_idx = train_test_split(
        np.arange(len(dev)),
        test_size=validation_fraction,
        random_state=seed + fold,
        stratify=dev["state_code"],
    )
    train = dev.iloc[tr_idx].reset_index(drop=True)
    val = dev.iloc[va_idx].reset_index(drop=True)
    return train, val, test


def deterministic_tile_indices(patient_id: str, n_tiles: int, fold: int, epoch: int, tile_cap: int | None, seed: int = 42):
    """Match the deterministic per-patient sampling used in the final notebooks."""
    if tile_cap is None or n_tiles <= tile_cap:
        return None
    sample_seed = zlib.crc32(f"{seed}|{fold}|{epoch}|{patient_id}".encode()) & 0xFFFFFFFF
    generator = torch.Generator().manual_seed(sample_seed)
    return torch.randperm(n_tiles, generator=generator)[:tile_cap]


def load_embedding_tensor(path: str | Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "features" in obj:
            obj = obj["features"]
        elif "embeddings" in obj:
            obj = obj["embeddings"]
        else:
            raise KeyError(f"{path} contains neither 'features' nor 'embeddings'.")
    if not torch.is_tensor(obj):
        obj = torch.tensor(obj)
    return obj.float().contiguous()


class EmbeddingStore:
    """CPU cache for patient-level frozen foundation-model embeddings."""

    def __init__(
        self,
        patient_ids: Iterable[str],
        fm: str,
        directories: Iterable[str | Path],
        input_dim: int,
        tile_cap: int | None = 3000,
        seed: int = 42,
    ):
        self.patient_ids = [str(x).upper().strip() for x in patient_ids]
        self.fm = fm_key(fm)
        self.directories = [Path(x) for x in directories]
        self.input_dim = int(input_dim)
        self.tile_cap = tile_cap
        self.seed = int(seed)
        self._cache: dict[str, torch.Tensor] = {}

    def find_file(self, patient_id: str) -> Path:
        pid = patient_id.upper().strip()
        for directory in self.directories:
            exact = directory / f"{pid}.pt"
            if exact.exists():
                return exact
            hits = sorted(directory.glob(f"{pid}*.pt"))
            if hits:
                return hits[0]
        raise FileNotFoundError(f"No embedding file found for {pid} in {self.directories}")

    def _load_one(self, patient_id: str):
        x = load_embedding_tensor(self.find_file(patient_id))
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"{patient_id}: expected [N,{self.input_dim}] embeddings, got {tuple(x.shape)}")
        return patient_id, x

    def preload(self, workers: int = 8, show_progress: bool = True) -> None:
        if workers <= 1:
            iterator = self.patient_ids
            if show_progress:
                iterator = tqdm(iterator, desc=f"Loading {self.fm} embeddings")
            for pid in iterator:
                key, x = self._load_one(pid)
                self._cache[key] = x
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._load_one, pid) for pid in self.patient_ids]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc=f"Loading {self.fm} embeddings")
            for future in iterator:
                pid, x = future.result()
                self._cache[pid] = x

    def tensor(self, patient_id: str) -> torch.Tensor:
        pid = patient_id.upper().strip()
        if pid not in self._cache:
            _, self._cache[pid] = self._load_one(pid)
        return self._cache[pid]

    def bag(self, patient_id: str, fold: int, epoch: int = 0, training: bool = False, device: torch.device | str = "cpu") -> torch.Tensor:
        x = self.tensor(patient_id)
        if training:
            idx = deterministic_tile_indices(patient_id, len(x), fold, epoch, self.tile_cap, self.seed)
            if idx is not None:
                x = x[idx]
        return x.to(device, non_blocking=True)


def save_splits(df: pd.DataFrame, output_dir: str | Path, biomarker_a: str, biomarker_b: str, seed: int = 42, validation_fraction: float = 0.15) -> None:
    """Save the exact train/validation/test membership for all five outer folds."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        train, val, test = split_outer_fold(df, fold, seed, validation_fraction)
        combined = pd.concat(
            [train.assign(split="train"), val.assign(split="validation"), test.assign(split="test")],
            ignore_index=True,
        )
        combined[["patient_id", biomarker_a, biomarker_b, "state_code", "joint", "fold", "split"]].to_csv(
            root / f"fold{fold}_split.csv", index=False
        )
