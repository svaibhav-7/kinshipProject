import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import seaborn as sns

# -----------------------------
# Model definition (rebuild from saved config)
# -----------------------------
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=512, embedding_dim=128):
        super().__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, embedding_dim),
        )

    def forward_once(self, x):
        return self.embedding_net(x)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


def load_pairs_from_mat(mat_path: Path) -> Tuple[List[str], List[str], List[int]]:
    """Load image filename pairs and labels from a KinFaceW-II .mat file.

    Returns three parallel lists: img1_fns, img2_fns, labels (1 kin, 0 non-kin).
    """
    d = scipy.io.loadmat(str(mat_path))
    pairs = d["pairs"]
    n = pairs.shape[0]
    img1: List[str] = []
    img2: List[str] = []
    labels: List[int] = []
    for i in range(n):
        lbl = int(pairs[i, 1])  # 1=kin, 0=non-kin
        a = pairs[i, 2]
        b = pairs[i, 3]
        # Elements could be MATLAB strings or nested arrays; coerce to Python str
        def to_str(x):
            try:
                if hasattr(x, "item"):
                    x = x.item()
            except Exception:
                pass
            # Convert possible nested arrays of chars to string
            try:
                s = "".join([chr(int(c)) for c in x.ravel()])  # type: ignore
                s = s.strip()
                if len(s) > 0:
                    return s
            except Exception:
                pass
            s = str(x)
            return s.strip()

        s1 = to_str(a)
        s2 = to_str(b)
        if not s1.endswith((".jpg", ".png", ".npy")):
            s1 += ".jpg"
        if not s2.endswith((".jpg", ".png", ".npy")):
            s2 += ".jpg"
        img1.append(s1)
        img2.append(s2)
        labels.append(lbl)
    return img1, img2, labels


def build_filename_index_map(metadata_json: Path) -> Dict[str, int]:
    """Map filename -> feature index using features_with_metadata.json."""
    with open(metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    mapping: Dict[str, int] = {}
    for idx, entry in enumerate(meta):
        fn = entry.get("filename")
        if fn is not None:
            mapping[fn] = idx
            mapping[os.path.basename(fn)] = idx  # also allow basename
    return mapping


def resolve_feature_index(fn: str, fmap: Dict[str, int]) -> int | None:
    """Try to locate the feature row index for a given image filename.
    Supports sidecar naming with _combined_features.npy as used in your data.
    """
    if fn in fmap:
        return fmap[fn]
    alt = fn.replace(".jpg", "_combined_features.npy").replace(".png", "_combined_features.npy")
    if alt in fmap:
        return fmap[alt]
    # Try basename fallbacks
    base = os.path.basename(fn)
    if base in fmap:
        return fmap[base]
    alt_base = base.replace(".jpg", "_combined_features.npy").replace(".png", "_combined_features.npy")
    return fmap.get(alt_base)


def compute_scores(
    model: SiameseNetwork, feats: np.ndarray, pairs: List[Tuple[int, int]]
) -> np.ndarray:
    """Compute similarity scores per pair using negative Euclidean distance.
    Higher score => more likely kin.
    """
    device = next(model.parameters()).device
    x_idx = torch.tensor([i for i, _ in pairs], dtype=torch.long, device=device)
    y_idx = torch.tensor([j for _, j in pairs], dtype=torch.long, device=device)

    with torch.no_grad():
        feats_t = torch.from_numpy(feats).float().to(device)
        x = feats_t[x_idx]
        y = feats_t[y_idx]
        e1, e2 = model(x, y)
        dist = torch.nn.functional.pairwise_distance(e1, e2)  # small => kin
        scores = (-dist).cpu().numpy()  # invert so higher => kin
    return scores


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Select best threshold based on Youden's J (TPR - FPR) on ROC curve.
    Returns threshold and metrics at that threshold.
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    k = int(np.argmax(j))
    best_thr = thr[k]
    y_pred = (scores >= best_thr).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return float(best_thr), metrics


def plot_roc_pr_cm(y_true: np.ndarray, scores: np.ndarray, thr: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=150)
    plt.close()

    # Confusion matrix at thr
    y_pred = (scores >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-kin", "Kin"], yticklabels=["Non-kin", "Kin"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Siamese kinship model with AUC/Accuracy and plots")
    parser.add_argument("--features", type=str, default=str(Path("model_visuals_ganmodel") / "normalized_features.npy"))
    parser.add_argument("--metadata-json", type=str, default=str(Path("model_visuals_ganmodel") / "features_with_metadata.json"))
    parser.add_argument("--meta-dir", type=str, default=str(Path("KinFaceW-II") / "meta_data"))
    parser.add_argument("--model-ckpt", type=str, default=str(Path("final_siamese_model") / "siamese_kinship_all_relations.pt"))
    parser.add_argument("--out-dir", type=str, default=str(Path("final_siamese_model") / "eval"))
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    features_path = Path(args.features)
    metadata_json = Path(args.metadata_json)
    meta_dir = Path(args.meta_dir)
    model_ckpt = Path(args.model_ckpt)
    out_dir = Path(args.out_dir)

    if not features_path.is_absolute():
        features_path = project_root / features_path
    if not metadata_json.is_absolute():
        metadata_json = project_root / metadata_json
    if not meta_dir.is_absolute():
        meta_dir = project_root / meta_dir
    if not model_ckpt.is_absolute():
        model_ckpt = project_root / model_ckpt
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load features and mapping
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    feats = np.load(str(features_path))
    fmap = build_filename_index_map(metadata_json)

    # Load model
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")
    ckpt = torch.load(str(model_ckpt), map_location=device)
    # Try nested dict under key, else entire object is state dict
    state_dict = ckpt.get("model_state_dict", ckpt)
    cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    input_dim = int(cfg.get("input_dim", feats.shape[1]))
    embedding_dim = int(cfg.get("embedding_dim", 128))

    model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Collect pairs from all relation .mat files
    relation_mats = [
        "fd_pairs.mat",
        "fs_pairs.mat",
        "md_pairs.mat",
        "ms_pairs.mat",
    ]
    all_img1: List[str] = []
    all_img2: List[str] = []
    all_labels: List[int] = []

    for m in relation_mats:
        mp = meta_dir / m
        if not mp.exists():
            print(f"[WARN] Missing: {mp}")
            continue
        i1, i2, y = load_pairs_from_mat(mp)
        all_img1.extend(i1)
        all_img2.extend(i2)
        all_labels.extend(y)

    if len(all_labels) == 0:
        raise RuntimeError("No pairs loaded from meta_data. Ensure KinFaceW-II/meta_data exists with *_pairs.mat files.")

    # Map filenames to feature indices
    pairs_idx: List[Tuple[int, int]] = []
    kept_labels: List[int] = []
    dropped = 0
    for s1, s2, y in zip(all_img1, all_img2, all_labels):
        i1 = resolve_feature_index(s1, fmap)
        i2 = resolve_feature_index(s2, fmap)
        if i1 is None or i2 is None:
            dropped += 1
            continue
        pairs_idx.append((i1, i2))
        kept_labels.append(int(y))

    if len(pairs_idx) == 0:
        raise RuntimeError("No valid pairs after mapping filenames to features. Check features_with_metadata.json alignment.")

    if dropped > 0:
        print(f"[INFO] Dropped {dropped} pairs due to missing feature mapping. Kept {len(pairs_idx)}.")

    # Compute scores and metrics
    scores = compute_scores(model, feats, pairs_idx)
    y_true = np.array(kept_labels, dtype=int)

    roc_auc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))
    thr, thr_metrics = find_best_threshold(y_true, scores)

    # Aggregate summary
    summary = {
        "num_pairs_total": int(len(all_labels)),
        "num_pairs_kept": int(len(y_true)),
        "num_pairs_dropped": int(dropped),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": thr,
        **thr_metrics,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-pair CSV
    import csv
    csv_path = out_dir / "pairs_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "feat_idx_1", "feat_idx_2", "label", "score"])
        for k, ((i1, i2), y, s) in enumerate(zip(pairs_idx, y_true, scores)):
            w.writerow([k, i1, i2, int(y), float(s)])

    # Save summary JSON
    json_path = out_dir / "evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots
    plot_roc_pr_cm(y_true, scores, thr, out_dir)

    print("Evaluation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved per-pair scores to: {csv_path}")
    print(f"Saved summary JSON to: {json_path}")
    print(f"Saved plots (ROC/PR/CM) to: {out_dir}")


if __name__ == "__main__":
    main()
