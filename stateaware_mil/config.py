"""Configuration utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return cfg


def merged_config(cfg: dict[str, Any], **updates: Any) -> dict[str, Any]:
    out = deepcopy(cfg)
    for key, value in updates.items():
        if value is not None:
            out[key] = value
    return out


def fm_key(name: str) -> str:
    x = name.strip().lower().replace("-h", "").replace("_", "")
    if x in {"uni2", "uni2h"}:
        return "uni2"
    if x == "conch":
        return "conch"
    raise ValueError("Foundation model must be 'uni2' or 'conch'.")


def fm_display_name(name: str) -> str:
    return "UNI2-h" if fm_key(name) == "uni2" else "CONCH"
