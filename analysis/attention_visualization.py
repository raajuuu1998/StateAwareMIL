"""Create DirectJoint vs State-Aware real-WSI attention visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata, spearmanr

from stateaware_mil.baselines import DirectJoint
from stateaware_mil.state_aware import StateAwareInteractionMIL


def load_feature_object(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "features" not in obj or "coords" not in obj:
        raise ValueError("Feature file must be a dict containing 'features' and 'coords'.")
    features = obj["features"]
    coords = obj["coords"]
    if not torch.is_tensor(features):
        features = torch.tensor(features)
    if torch.is_tensor(coords):
        coords = coords.cpu().numpy()
    return features.float(), np.asarray(coords)


def load_checkpoint(path: Path, model, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt)
    model.eval()
    return model


def attention_weights(encoder, x):
    return encoder.attention_weights(x).detach().float().cpu().numpy()


def make_background_white(thumb):
    img = thumb.astype(np.float32) / 255.0
    max_rgb = img.max(axis=2)
    min_rgb = img.min(axis=2)
    saturation = (max_rgb - min_rgb) / np.maximum(max_rgb, 1e-6)
    brightness = img.mean(axis=2)
    background = (brightness > 0.94) & (saturation < 0.055)
    cleaned = thumb.copy()
    cleaned[background] = [255, 255, 255]
    return cleaned


def make_heat(coords, values, width, height, thumb_width, thumb_height, thumb):
    sx, sy = thumb_width / width, thumb_height / height
    px = np.clip(np.rint(coords[:, 0] * sx).astype(int), 0, thumb_width - 1)
    py = np.clip(np.rint(coords[:, 1] * sy).astype(int), 0, thumb_height - 1)
    numerator = np.zeros((thumb_height, thumb_width), dtype=np.float32)
    denominator = np.zeros_like(numerator)
    np.add.at(numerator, (py, px), values)
    np.add.at(denominator, (py, px), 1.0)

    ux, uy = np.unique(np.sort(coords[:, 0])), np.unique(np.sort(coords[:, 1]))
    dx, dy = np.diff(ux), np.diff(uy)
    spacing = np.concatenate([dx[dx > 0], dy[dy > 0]])
    step = float(np.median(spacing)) if len(spacing) else 224.0
    sigma = max(4.5, 0.65 * step * (sx + sy) / 2.0)
    num_smooth = gaussian_filter(numerator, sigma=sigma)
    den_smooth = gaussian_filter(denominator, sigma=sigma)
    heat = num_smooth / np.maximum(den_smooth, 1e-8)
    coverage = den_smooth / np.maximum(den_smooth.max(), 1e-8)
    support = coverage > 0.025

    rgb = thumb.astype(np.float32) / 255.0
    max_rgb, min_rgb = rgb.max(axis=2), rgb.min(axis=2)
    saturation = (max_rgb - min_rgb) / np.maximum(max_rgb, 1e-6)
    brightness = rgb.mean(axis=2)
    tissue = (saturation > 0.055) & (brightness < 0.97)
    final_mask = support & tissue
    heat = np.clip(gaussian_filter(heat, sigma=1.25), 0, 1)
    alpha = np.zeros((thumb_height, thumb_width), dtype=np.float32)
    alpha[final_mask] = 0.18 + 0.58 * np.power(heat[final_mask], 1.30)
    return np.ma.masked_where(~final_mask, heat), alpha, sx, sy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide", required=True, help="Diagnostic WSI (.svs or OpenSlide-compatible format).")
    parser.add_argument("--features", required=True, help="UNI2-h feature .pt containing features and coordinates.")
    parser.add_argument("--direct-checkpoint", required=True)
    parser.add_argument("--state-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output path stem without extension.")
    parser.add_argument("--input-dim", type=int, default=1536)
    parser.add_argument("--thumbnail-size", type=int, default=2200)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    features, coords = load_feature_object(Path(args.features))
    direct = load_checkpoint(Path(args.direct_checkpoint), DirectJoint(args.input_dim).to(device), device)
    state = load_checkpoint(Path(args.state_checkpoint), StateAwareInteractionMIL(args.input_dim).to(device), device)
    x = features.to(device)

    with torch.inference_mode():
        direct_attention = attention_weights(direct.encoder, x)
        attention_a = attention_weights(state.a_encoder, x)
        attention_b = attention_weights(state.b_encoder, x)

    state_attention = np.sqrt(attention_a * attention_b)
    state_attention = state_attention / np.maximum(state_attention.sum(), 1e-12)
    direct_pct = (rankdata(direct_attention, method="average") - 1) / max(len(direct_attention) - 1, 1)
    state_pct = (rankdata(state_attention, method="average") - 1) / max(len(state_attention) - 1, 1)
    rho = float(spearmanr(direct_attention, state_attention).statistic)
    k = max(1, int(np.ceil(0.10 * len(direct_attention))))
    top_direct, top_state = set(np.argsort(direct_attention)[-k:]), set(np.argsort(state_attention)[-k:])
    overlap = len(top_direct & top_state) / len(top_direct | top_state)

    slide = openslide.OpenSlide(args.slide)
    width, height = slide.dimensions
    thumb_original = np.asarray(slide.get_thumbnail((args.thumbnail_size, args.thumbnail_size)).convert("RGB"))
    thumb = make_background_white(thumb_original)
    th, tw = thumb.shape[:2]
    heat_dj, alpha_dj, sx, sy = make_heat(coords, direct_pct, width, height, tw, th, thumb_original)
    heat_sa, alpha_sa, _, _ = make_heat(coords, state_pct, width, height, tw, th, thumb_original)

    xmin, xmax = int(max(0, coords[:, 0].min() * sx)), int(min(tw, coords[:, 0].max() * sx))
    ymin, ymax = int(max(0, coords[:, 1].min() * sy)), int(min(th, coords[:, 1].max() * sy))
    padx, pady = max(20, int(0.035 * (xmax - xmin))), max(20, int(0.035 * (ymax - ymin)))
    xmin, xmax = max(0, xmin - padx), min(tw, xmax + padx)
    ymin, ymax = max(0, ymin - pady), min(th, ymax + pady)

    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad((0, 0, 0, 0))
    fig = plt.figure(figsize=(16.5, 5.15), facecolor="white")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.035], wspace=0.055)
    ax0, ax1, ax2, cax = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax0.imshow(thumb)
    ax0.set_title("Original H&E", fontsize=13, fontweight="bold", pad=8)
    ax1.imshow(thumb)
    ax1.imshow(heat_dj, cmap=cmap, vmin=0, vmax=1, alpha=alpha_dj, interpolation="bilinear")
    ax1.set_title("DirectJoint", fontsize=13, fontweight="bold", pad=8)
    ax2.imshow(thumb)
    im = ax2.imshow(heat_sa, cmap=cmap, vmin=0, vmax=1, alpha=alpha_sa, interpolation="bilinear")
    ax2.set_title("State-Aware MIL", fontsize=13, fontweight="bold", pad=8)
    for ax in [ax0, ax1, ax2]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.axis("off")
    cbar = fig.colorbar(im, cax=cax, ticks=np.linspace(0, 1, 6))
    cbar.set_label("Attention percentile", fontsize=11, labelpad=10)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    slide.close()

    metrics = {"attention_spearman": rho, "top_10_percent_jaccard": overlap, "n_patches": len(direct_attention)}
    with out.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
