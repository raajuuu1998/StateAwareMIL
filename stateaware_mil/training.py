"""Training loops reproducing the final five-fold experiments."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .ablations import StateAwareAblation
from .baselines import DirectJoint, IndependentPair, NaiveMTL, build_posthoc_lr, posthoc_features
from .data import EmbeddingStore, set_seed, split_outer_fold
from .evaluation import joint_metrics, summarize_folds
from .fourstate_abmil import FourStateABMIL
from .state_aware import StateAwareInteractionMIL


@dataclass(frozen=True)
class TrainingSettings:
    seed: int = 42
    hidden_dim: int = 256
    attention_dim: int = 128
    dropout: float = 0.10
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 30
    patience: int = 6
    accumulation_steps: int = 4
    auxiliary_weight: float = 0.25
    validation_fraction: float = 0.15
    posthoc_c: float = 1.0
    posthoc_max_iter: int = 2000


def positive_weight(y, device: torch.device) -> torch.Tensor:
    y = np.asarray(y, dtype=int)
    return torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), dtype=torch.float32, device=device)


def state_weights(y, device: torch.device) -> torch.Tensor:
    y = np.asarray(y, dtype=int)
    counts = np.maximum(np.bincount(y, minlength=4).astype(float), 1.0)
    return torch.tensor(len(y) / (4.0 * counts), dtype=torch.float32, device=device)


def build_main_model(method: str, input_dim: int, settings: TrainingSettings, device: torch.device):
    key = method.lower().replace("-", "").replace("_", "")
    common = dict(
        in_dim=input_dim,
        hidden_dim=settings.hidden_dim,
        attention_dim=settings.attention_dim,
        dropout=settings.dropout,
    )
    if key == "directjoint":
        model = DirectJoint(**common)
    elif key == "independentpair":
        model = IndependentPair(**common)
    elif key == "naivemtl":
        model = NaiveMTL(**common)
    elif key in {"stateaware", "stateawareinteractionmil"}:
        model = StateAwareInteractionMIL(**common)
    else:
        raise ValueError(f"Unsupported neural method: {method}")
    return model.to(device)


def canonical_method(method: str) -> str:
    key = method.lower().replace("-", "").replace("_", "")
    names = {
        "directjoint": "DirectJoint",
        "independentpair": "IndependentPair",
        "naivemtl": "NaiveMTL",
        "stateaware": "StateAware",
        "stateawareinteractionmil": "StateAware",
        "posthoclr": "PostHocLR",
        "fourstateabmil": "FourStateABMIL",
    }
    if key not in names:
        raise ValueError(f"Unknown method: {method}")
    return names[key]


def main_loss(method: str, out: dict[str, torch.Tensor], row, pwa, pwb, pwj, sw, aux_weight: float, device: torch.device):
    method = canonical_method(method)
    ya = torch.tensor(float(row.A), device=device)
    yb = torch.tensor(float(row.B), device=device)
    yj = torch.tensor(float(row.joint), device=device)
    ys = torch.tensor(int(row.state_code), dtype=torch.long, device=device)

    if method == "DirectJoint":
        return F.binary_cross_entropy_with_logits(out["joint_logit"], yj, pos_weight=pwj)
    if method == "IndependentPair":
        la = F.binary_cross_entropy_with_logits(out["a_logit"], ya, pos_weight=pwa)
        lb = F.binary_cross_entropy_with_logits(out["b_logit"], yb, pos_weight=pwb)
        return 0.5 * (la + lb)
    if method == "NaiveMTL":
        la = F.binary_cross_entropy_with_logits(out["a_logit"], ya, pos_weight=pwa)
        lb = F.binary_cross_entropy_with_logits(out["b_logit"], yb, pos_weight=pwb)
        lj = F.binary_cross_entropy_with_logits(out["joint_logit"], yj, pos_weight=pwj)
        return (la + lb + lj) / 3.0

    ls = F.cross_entropy(out["state_logits"].view(1, 4), ys.view(1), weight=sw)
    la = F.binary_cross_entropy_with_logits(out["a_logit"], ya, pos_weight=pwa)
    lb = F.binary_cross_entropy_with_logits(out["b_logit"], yb, pos_weight=pwb)
    return ls + aux_weight * 0.5 * (la + lb)


@torch.inference_mode()
def predict_main(model, method: str, part: pd.DataFrame, store: EmbeddingStore, device: torch.device, biomarker_a: str, biomarker_b: str) -> pd.DataFrame:
    method = canonical_method(method)
    model.eval()
    rows = []
    for r in part.itertuples(index=False):
        x = store.bag(r.patient_id, int(r.fold), training=False, device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)

        rec = {
            "patient_id": r.patient_id,
            "fold": int(r.fold),
            "A": int(r.A),
            "B": int(r.B),
            biomarker_a: int(r.A),
            biomarker_b: int(r.B),
            "state_code": int(r.state_code),
            "joint": int(r.joint),
        }
        if method == "DirectJoint":
            rec["p_joint"] = float(torch.sigmoid(out["joint_logit"]).cpu())
        elif method == "IndependentPair":
            pa = float(torch.sigmoid(out["a_logit"]).cpu())
            pb = float(torch.sigmoid(out["b_logit"]).cpu())
            rec.update({"p_A": pa, "p_B": pb, "p_joint": pa * pb})
        elif method == "NaiveMTL":
            rec.update({
                "p_A": float(torch.sigmoid(out["a_logit"]).cpu()),
                "p_B": float(torch.sigmoid(out["b_logit"]).cpu()),
                "p_joint": float(torch.sigmoid(out["joint_logit"]).cpu()),
            })
        else:
            ps = torch.softmax(out["state_logits"], dim=0).float().cpu().numpy()
            rec.update({
                "p00": float(ps[0]), "p01": float(ps[1]), "p10": float(ps[2]), "p11": float(ps[3]),
                "p_A": float(torch.sigmoid(out["a_logit"]).cpu()),
                "p_B": float(torch.sigmoid(out["b_logit"]).cpu()),
                "p_joint": float(ps[3]),
                "p_A_state": float(ps[2] + ps[3]),
                "p_B_state": float(ps[1] + ps[3]),
            })
        rows.append(rec)
        del x
    return pd.DataFrame(rows)


def _save_final_outputs(folder: Path, predictions: list[pd.DataFrame], metrics: list[dict]) -> pd.DataFrame:
    oof = pd.concat(predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(metrics).sort_values("fold").reset_index(drop=True)
    summary = summarize_folds(fold_metrics)
    pooled = pd.DataFrame([joint_metrics(oof["joint"], oof["p_joint"])])
    oof.to_csv(folder / "oof_predictions.csv", index=False)
    fold_metrics.to_csv(folder / "fold_metrics.csv", index=False)
    summary.to_csv(folder / "summary_metrics.csv", index=False)
    pooled.to_csv(folder / "pooled_oof_metrics.csv", index=False)
    return summary


def train_main_method(
    method: str,
    df: pd.DataFrame,
    store: EmbeddingStore,
    input_dim: int,
    output_root: str | Path,
    fm_folder: str,
    biomarker_a: str,
    biomarker_b: str,
    settings: TrainingSettings,
    device: torch.device,
    force: bool = False,
) -> pd.DataFrame:
    method = canonical_method(method)
    if method == "PostHocLR":
        return train_posthoc_lr(df, store, input_dim, output_root, fm_folder, biomarker_a, biomarker_b, settings, device, force)

    folder = Path(output_root) / method / fm_folder
    folder.mkdir(parents=True, exist_ok=True)
    oofs, all_metrics = [], []

    for fold in range(5):
        checkpoint = folder / f"fold{fold}_best.pt"
        pred_file = folder / f"fold{fold}_predictions.csv"
        metric_file = folder / f"fold{fold}_metrics.csv"
        if not force and checkpoint.exists() and pred_file.exists() and metric_file.exists():
            oofs.append(pd.read_csv(pred_file))
            all_metrics.append(pd.read_csv(metric_file).iloc[0].to_dict())
            continue

        set_seed(settings.seed + fold)
        train, val, test = split_outer_fold(df, fold, settings.seed, settings.validation_fraction)
        model = build_main_model(method, input_dim, settings, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        pwa = positive_weight(train["A"], device)
        pwb = positive_weight(train["B"], device)
        pwj = positive_weight(train["joint"], device)
        sw = state_weights(train["state_code"], device)
        records = list(train.itertuples(index=False))
        best_ap, best_epoch, bad = -1.0, 0, 0
        history = []

        for epoch in range(1, settings.max_epochs + 1):
            model.train()
            order = np.random.RandomState(settings.seed + fold + epoch).permutation(len(records))
            optimizer.zero_grad(set_to_none=True)
            running = 0.0

            for step, idx in enumerate(order, 1):
                row = records[idx]
                x = store.bag(row.patient_id, fold, epoch=epoch, training=True, device=device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model(x)
                    loss = main_loss(method, out, row, pwa, pwb, pwj, sw, settings.auxiliary_weight, device)
                scaler.scale(loss / settings.accumulation_steps).backward()
                if step % settings.accumulation_steps == 0 or step == len(order):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running += float(loss.detach().cpu())
                del x

            val_pred = predict_main(model, method, val, store, device, biomarker_a, biomarker_b)
            val_auc = roc_auc_score(val_pred["joint"], val_pred["p_joint"])
            val_ap = average_precision_score(val_pred["joint"], val_pred["p_joint"])
            history.append({"epoch": epoch, "train_loss": running / len(records), "val_AUROC": val_auc, "val_AP": val_ap})

            if val_ap > best_ap:
                best_ap, best_epoch, bad = val_ap, epoch, 0
                torch.save({
                    "model_state": model.state_dict(),
                    "method": method,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "best_val_ap": best_ap,
                    "settings": settings.__dict__,
                }, checkpoint)
            else:
                bad += 1
            if bad >= settings.patience:
                break

        pd.DataFrame(history).to_csv(folder / f"fold{fold}_history.csv", index=False)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        val_pred = predict_main(model, method, val, store, device, biomarker_a, biomarker_b)
        test_pred = predict_main(model, method, test, store, device, biomarker_a, biomarker_b)
        metric = joint_metrics(test_pred["joint"], test_pred["p_joint"])
        metric.update({"method": method, "fold": fold, "best_epoch": best_epoch, "best_val_AP": best_ap})
        val_pred.to_csv(folder / f"fold{fold}_val_predictions.csv", index=False)
        test_pred.to_csv(pred_file, index=False)
        pd.DataFrame([metric]).to_csv(metric_file, index=False)
        oofs.append(test_pred)
        all_metrics.append(metric)

        del model, optimizer, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _save_final_outputs(folder, oofs, all_metrics)


def _load_independent_model(input_dim: int, checkpoint: Path, settings: TrainingSettings, device: torch.device):
    model = IndependentPair(input_dim, settings.hidden_dim, settings.attention_dim, settings.dropout).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model


def train_posthoc_lr(
    df: pd.DataFrame,
    store: EmbeddingStore,
    input_dim: int,
    output_root: str | Path,
    fm_folder: str,
    biomarker_a: str,
    biomarker_b: str,
    settings: TrainingSettings,
    device: torch.device,
    force: bool = False,
) -> pd.DataFrame:
    output_root = Path(output_root)
    folder = output_root / "PostHocLR" / fm_folder
    pair_folder = output_root / "IndependentPair" / fm_folder
    folder.mkdir(parents=True, exist_ok=True)
    oofs, all_metrics = [], []

    for fold in range(5):
        pred_file = folder / f"fold{fold}_predictions.csv"
        metric_file = folder / f"fold{fold}_metrics.csv"
        model_file = folder / f"fold{fold}_model.joblib"
        if not force and pred_file.exists() and metric_file.exists() and model_file.exists():
            oofs.append(pd.read_csv(pred_file))
            all_metrics.append(pd.read_csv(metric_file).iloc[0].to_dict())
            continue

        checkpoint = pair_folder / f"fold{fold}_best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"PostHoc-LR requires IndependentPair checkpoints. Missing {checkpoint}. "
                "Run IndependentPair first."
            )
        train, val, test = split_outer_fold(df, fold, settings.seed, settings.validation_fraction)
        pair_model = _load_independent_model(input_dim, checkpoint, settings, device)
        train_pair = predict_main(pair_model, "IndependentPair", train, store, device, biomarker_a, biomarker_b)
        val_pair = predict_main(pair_model, "IndependentPair", val, store, device, biomarker_a, biomarker_b)
        test_pair = predict_main(pair_model, "IndependentPair", test, store, device, biomarker_a, biomarker_b)

        meta = build_posthoc_lr(settings.posthoc_c, settings.posthoc_max_iter, settings.seed + fold)
        meta.fit(posthoc_features(train_pair["p_A"], train_pair["p_B"]), train_pair["joint"].to_numpy())
        val_pred = val_pair.copy()
        test_pred = test_pair.copy()
        val_pred["p_joint"] = meta.predict_proba(posthoc_features(val_pair["p_A"], val_pair["p_B"]))[:, 1]
        test_pred["p_joint"] = meta.predict_proba(posthoc_features(test_pair["p_A"], test_pair["p_B"]))[:, 1]

        metric = joint_metrics(test_pred["joint"], test_pred["p_joint"])
        metric.update({"method": "PostHocLR", "fold": fold})
        train_pair.to_csv(folder / f"fold{fold}_meta_train_predictions.csv", index=False)
        val_pred.to_csv(folder / f"fold{fold}_val_predictions.csv", index=False)
        test_pred.to_csv(pred_file, index=False)
        pd.DataFrame([metric]).to_csv(metric_file, index=False)
        joblib.dump(meta, model_file)
        oofs.append(test_pred)
        all_metrics.append(metric)
        del pair_model, meta
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _save_final_outputs(folder, oofs, all_metrics)


@torch.inference_mode()
def predict_fourstate(model, part: pd.DataFrame, store: EmbeddingStore, device: torch.device, biomarker_a: str, biomarker_b: str) -> pd.DataFrame:
    model.eval()
    rows = []
    for r in part.itertuples(index=False):
        x = store.bag(r.patient_id, int(r.fold), training=False, device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        probs = torch.softmax(logits, dim=0).float().cpu().numpy()
        rows.append({
            "patient_id": r.patient_id, "fold": int(r.fold), "A": int(r.A), "B": int(r.B),
            biomarker_a: int(r.A), biomarker_b: int(r.B), "state_code": int(r.state_code), "joint": int(r.joint),
            "p00": float(probs[0]), "p01": float(probs[1]), "p10": float(probs[2]), "p11": float(probs[3]),
            "p_joint": float(probs[3]), "predicted_state": int(np.argmax(probs)),
        })
        del x
    return pd.DataFrame(rows)


def train_fourstate_abmil(
    df: pd.DataFrame,
    store: EmbeddingStore,
    input_dim: int,
    output_root: str | Path,
    fm_folder: str,
    biomarker_a: str,
    biomarker_b: str,
    settings: TrainingSettings,
    device: torch.device,
    force: bool = False,
) -> pd.DataFrame:
    folder = Path(output_root) / "FourStateABMIL" / fm_folder
    folder.mkdir(parents=True, exist_ok=True)
    oofs, all_metrics = [], []

    for fold in range(5):
        checkpoint = folder / f"fold{fold}_best.pt"
        pred_file = folder / f"fold{fold}_predictions.csv"
        metric_file = folder / f"fold{fold}_metrics.csv"
        if not force and checkpoint.exists() and pred_file.exists() and metric_file.exists():
            oofs.append(pd.read_csv(pred_file))
            all_metrics.append(pd.read_csv(metric_file).iloc[0].to_dict())
            continue

        set_seed(settings.seed + fold)
        train, val, test = split_outer_fold(df, fold, settings.seed, settings.validation_fraction)
        model = FourStateABMIL(input_dim, settings.hidden_dim, settings.attention_dim, settings.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        sw = state_weights(train["state_code"], device)
        records = list(train.itertuples(index=False))
        history, best_ap, best_epoch, bad = [], -1.0, 0, 0

        for epoch in range(1, settings.max_epochs + 1):
            model.train()
            order = np.random.RandomState(settings.seed + fold + epoch).permutation(len(records))
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            for step, idx in enumerate(order, 1):
                r = records[idx]
                x = store.bag(r.patient_id, fold, epoch=epoch, training=True, device=device)
                y = torch.tensor(int(r.state_code), dtype=torch.long, device=device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(1, 4), y.view(1), weight=sw)
                scaler.scale(loss / settings.accumulation_steps).backward()
                if step % settings.accumulation_steps == 0 or step == len(order):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running += float(loss.detach().cpu())
                del x

            val_pred = predict_fourstate(model, val, store, device, biomarker_a, biomarker_b)
            val_auc = roc_auc_score(val_pred["joint"], val_pred["p_joint"])
            val_ap = average_precision_score(val_pred["joint"], val_pred["p_joint"])
            history.append({"epoch": epoch, "train_loss": running / len(records), "val_AUROC": val_auc, "val_AP": val_ap})
            if val_ap > best_ap:
                best_ap, best_epoch, bad = val_ap, epoch, 0
                torch.save({"model_state": model.state_dict(), "method": "FourStateABMIL", "fold": fold, "best_epoch": best_epoch, "best_val_ap": best_ap, "settings": settings.__dict__}, checkpoint)
            else:
                bad += 1
            if bad >= settings.patience:
                break

        pd.DataFrame(history).to_csv(folder / f"fold{fold}_history.csv", index=False)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        val_pred = predict_fourstate(model, val, store, device, biomarker_a, biomarker_b)
        test_pred = predict_fourstate(model, test, store, device, biomarker_a, biomarker_b)
        metric = joint_metrics(test_pred["joint"], test_pred["p_joint"])
        metric.update({"method": "FourStateABMIL", "fold": fold, "best_epoch": best_epoch, "best_val_AP": best_ap})
        val_pred.to_csv(folder / f"fold{fold}_val_predictions.csv", index=False)
        test_pred.to_csv(pred_file, index=False)
        pd.DataFrame([metric]).to_csv(metric_file, index=False)
        oofs.append(test_pred)
        all_metrics.append(metric)
        del model, optimizer, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _save_final_outputs(folder, oofs, all_metrics)


def ablation_loss(variant: str, out, row, pwa, pwb, pwj, sw, aux_weight: float, device: torch.device):
    ya = torch.tensor(float(row.A), device=device)
    yb = torch.tensor(float(row.B), device=device)
    yj = torch.tensor(float(row.joint), device=device)
    ys = torch.tensor(int(row.state_code), dtype=torch.long, device=device)
    la = F.binary_cross_entropy_with_logits(out["a_logit"], ya, pos_weight=pwa)
    lb = F.binary_cross_entropy_with_logits(out["b_logit"], yb, pos_weight=pwb)
    if variant == "binary_joint":
        lj = F.binary_cross_entropy_with_logits(out["joint_logit"], yj, pos_weight=pwj)
        return lj + aux_weight * 0.5 * (la + lb)
    ls = F.cross_entropy(out["state_logits"].view(1, 4), ys.view(1), weight=sw)
    if variant == "no_auxiliary":
        return ls
    return ls + aux_weight * 0.5 * (la + lb)


@torch.inference_mode()
def predict_ablation(model, variant: str, part: pd.DataFrame, store: EmbeddingStore, device: torch.device, biomarker_a: str, biomarker_b: str) -> pd.DataFrame:
    model.eval()
    rows = []
    for r in part.itertuples(index=False):
        x = store.bag(r.patient_id, int(r.fold), training=False, device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
        if variant == "binary_joint":
            p_joint = float(torch.sigmoid(out["joint_logit"]).cpu())
            p00 = p01 = p10 = p11 = np.nan
        else:
            ps = torch.softmax(out["state_logits"], dim=0).float().cpu().numpy()
            p00, p01, p10, p11 = map(float, ps)
            p_joint = p11
        rows.append({
            "patient_id": r.patient_id, "fold": int(r.fold), "A": int(r.A), "B": int(r.B),
            biomarker_a: int(r.A), biomarker_b: int(r.B), "state_code": int(r.state_code), "joint": int(r.joint),
            "p00": p00, "p01": p01, "p10": p10, "p11": p11,
            "p_A": float(torch.sigmoid(out["a_logit"]).cpu()),
            "p_B": float(torch.sigmoid(out["b_logit"]).cpu()),
            "p_joint": p_joint,
        })
        del x
    return pd.DataFrame(rows)


def train_ablation(
    variant: str,
    df: pd.DataFrame,
    store: EmbeddingStore,
    input_dim: int,
    output_root: str | Path,
    fm_folder: str,
    biomarker_a: str,
    biomarker_b: str,
    settings: TrainingSettings,
    device: torch.device,
    force: bool = False,
) -> pd.DataFrame:
    variant = variant.lower()
    folder = Path(output_root) / "ablations" / variant / fm_folder
    folder.mkdir(parents=True, exist_ok=True)
    oofs, all_metrics = [], []

    for fold in range(5):
        checkpoint = folder / f"fold{fold}_best.pt"
        pred_file = folder / f"fold{fold}_predictions.csv"
        metric_file = folder / f"fold{fold}_metrics.csv"
        if not force and checkpoint.exists() and pred_file.exists() and metric_file.exists():
            oofs.append(pd.read_csv(pred_file))
            all_metrics.append(pd.read_csv(metric_file).iloc[0].to_dict())
            continue

        set_seed(settings.seed + fold)
        train, val, test = split_outer_fold(df, fold, settings.seed, settings.validation_fraction)
        model = StateAwareAblation(input_dim, variant, settings.hidden_dim, settings.attention_dim, settings.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        pwa = positive_weight(train["A"], device)
        pwb = positive_weight(train["B"], device)
        pwj = positive_weight(train["joint"], device)
        sw = state_weights(train["state_code"], device)
        records = list(train.itertuples(index=False))
        history, best_ap, best_epoch, bad = [], -1.0, 0, 0

        for epoch in range(1, settings.max_epochs + 1):
            model.train()
            order = np.random.RandomState(settings.seed + fold + epoch).permutation(len(records))
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            for step, idx in enumerate(order, 1):
                r = records[idx]
                x = store.bag(r.patient_id, fold, epoch=epoch, training=True, device=device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model(x)
                    loss = ablation_loss(variant, out, r, pwa, pwb, pwj, sw, settings.auxiliary_weight, device)
                scaler.scale(loss / settings.accumulation_steps).backward()
                if step % settings.accumulation_steps == 0 or step == len(order):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running += float(loss.detach().cpu())
                del x

            val_pred = predict_ablation(model, variant, val, store, device, biomarker_a, biomarker_b)
            val_auc = roc_auc_score(val_pred["joint"], val_pred["p_joint"])
            val_ap = average_precision_score(val_pred["joint"], val_pred["p_joint"])
            history.append({"epoch": epoch, "train_loss": running / len(records), "val_AUROC": val_auc, "val_AP": val_ap})
            if val_ap > best_ap:
                best_ap, best_epoch, bad = val_ap, epoch, 0
                torch.save({"model_state": model.state_dict(), "variant": variant, "fold": fold, "best_epoch": best_epoch, "best_val_ap": best_ap, "settings": settings.__dict__}, checkpoint)
            else:
                bad += 1
            if bad >= settings.patience:
                break

        pd.DataFrame(history).to_csv(folder / f"fold{fold}_history.csv", index=False)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        val_pred = predict_ablation(model, variant, val, store, device, biomarker_a, biomarker_b)
        test_pred = predict_ablation(model, variant, test, store, device, biomarker_a, biomarker_b)
        metric = joint_metrics(test_pred["joint"], test_pred["p_joint"])
        metric.update({"variant": variant, "fold": fold, "best_epoch": best_epoch, "best_val_AP": best_ap})
        val_pred.to_csv(folder / f"fold{fold}_val_predictions.csv", index=False)
        test_pred.to_csv(pred_file, index=False)
        pd.DataFrame([metric]).to_csv(metric_file, index=False)
        oofs.append(test_pred)
        all_metrics.append(metric)
        del model, optimizer, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _save_final_outputs(folder, oofs, all_metrics)
