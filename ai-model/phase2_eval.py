"""
phase2_eval.py
==============
Phase 2 — GPU-side, inference-only evaluation for the Agri-CNX-Edge paper.
Produces everything the paper still needs beyond Phase 1, with NO
retraining and NO weight modification:

  * Class-wise precision / recall / F1 + confusion matrices
    (paper Tables leaf/pest/fruit + Figures 10/11/12)
  * Per-image fruit-count stats: MAE / RMSE / Pearson r / R^2 /
    mean signed error + predicted-vs-ground-truth scatter
    (paper Table counting + Figure 13)
  * Ground-truth-vs-predicted box overlays for qualitative figures
    (paper Figures 14/15)

USAGE (on the college machine, from the project root folder):
    python phase2_eval.py --preflight-only     # verify setup first (fast, ~1 min)
    python phase2_eval.py                      # full evaluation (all configs)
    python phase2_eval.py --configs proposed   # subset of configs
    python phase2_eval.py --device cpu         # force CPU (slow, fallback)
    python phase2_eval.py --splits val         # val only (skip test split)

    # If the script is NOT inside the project root, point at it explicitly
    # (Linux example — no code edits needed for paths):
    python phase2_eval.py --preflight-only --root /home/CL407-33/Downloads/BTechProject-20260806T081318Z-1-001/BTechProject/models

    # Auto-stage the 4 soft-label CSVs from inside the current structure:
    python phase2_eval.py --preflight-only --stage-csvs

WHERE MISSING FILES GO (all under the project root = folder with
'Proposed Model'; this folder is the same on every machine):
    Proposed Model/outputs/{leaf,pest,fruit,yield}/*_soft_labels.csv
        <- auto-filled by --stage-csvs (copies from PDDD*/YOLO* outputs)
    Proposed Model/data/leaf/<4 class folders>      <- your leaf images
    Proposed Model/data/pest/<4 class folders>      <- copy from PDDD Pest 1/data/pest/
    Proposed Model/data/fruit/<3 class folders>     <- your fruit images
    Proposed Model/data/yield/images/{train,val}/ + labels/{train,val}/
                                                    <- your yield images + YOLO .txt
    Proposed Model/weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
                                                    <- your original DINOv3 file
    Checkpoints: already in-structure (proposed/ablA/ablB/ablC have .pth;
    frozen + ablD have none and auto-skip). Nothing to add.

REQUIREMENTS on the college machine:
    torch + torchvision (CUDA build), albumentations, Pillow,
    scikit-learn, numpy, matplotlib, tqdm (optional).
    Data + weights + soft-label CSVs at the config paths
    (i.e. the machine training ran on), plus the Proposed Model/*.py
    sources in this same folder structure.

CHECKPOINT POLICY (matches the artifacts on disk):
    best_model.pth is preferred; latest_checkpoint.pth is used as a
    documented fallback (frozen + ablation C have no best_model.pth).
    Configs with no checkpoint at all are SKIPPED with a clear warning
    (never a crash). Weights load with strict=False and every
    missing/unexpected key is logged (ablation C has no LoRA layers,
    which is expected and handled).

OUTPUTS (in phase2_results/):
    <config>/classification_<split>.{json,csv}     per-class P/R/F1
    <config>/cm_<task>_<split>.{pdf,png}           confusion matrices
    <config>/yield_per_image.csv + yield_metrics.json
    <config>/fig13_pred_vs_gt.{pdf,png}            count scatter
    <config>/overlays/bbox_*.png                   GT-vs-pred overlays
    <config>/RUN_REPORT.txt                        what ran, what was skipped
    phase2_summary.json                            machine-readable roll-up
    phase2_captions.txt                            draft captions for the paper

HONESTY NOTES (also stated in the paper):
    * Classification metrics are reported separately for the val split
      (best-epoch convention used so far) and the held-out test split.
    * Fruit classification is evaluated on GT-box ROIs (diagnostic: it
      measures the fruit head, not end-to-end detection+classification).
    * Yield counting is evaluated on the 400-image yield val set.
"""

import argparse
import csv
import gc
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Proposed Model"))

# --------------------------------------------------------------------------
# Root handling (Linux-safe: never hardcode drive letters or home dirs).
# The project root is the folder that directly contains "Proposed Model".
# Override with --root if the script lives elsewhere.
# --------------------------------------------------------------------------
EXPECTED_TOP = ["Proposed Model", "Ablation", "Ablation Frozen Backbone",
                "PDDD Fruit 1", "PDDD Fruit 3 and PDDD Leaf 1", "PDDD Pest 1",
                "YOLO26s apple object detection model 1",
                "YOLO26s apple object detection model 2",
                "YOLO26s apple object detection model 3"]


def build_runs(root):
    """Config registry (same relative layout on every machine)."""
    return {
        "proposed": {
            "label": "Proposed (LoRA)",
            "ckpt_dir": root / "Proposed Model" / "student" / "checkpoints",
        },
        "frozen": {
            "label": "Frozen backbone",
            "ckpt_dir": (root / "Ablation Frozen Backbone" / "outputs"
                         / "studentFrozen" / "checkpoints"),
        },
        "ablA": {
            "label": "A: No KD",
            "ckpt_dir": root / "Ablation" / "outputs" / "ablation_A_no_kd" / "checkpoints",
        },
        "ablB": {
            "label": "B: Static weights",
            "ckpt_dir": (root / "Ablation" / "outputs" / "ablation_B_static_weights"
                         / "checkpoints"),
        },
        "ablC": {
            "label": "C: No LoRA",
            "ckpt_dir": (root / "Ablation" / "outputs" / "ablation_C_no_lora"
                         / "checkpoints"),
        },
        "ablD": {
            "label": "D: No warm-up",
            "ckpt_dir": (root / "Ablation" / "outputs" / "ablation_D_no_warmup"
                         / "checkpoints"),
        },
    }


RUNS = build_runs(ROOT)


def set_root(path):
    """Point the script at another project root (e.g. Linux college path).

    Returns (ok, message). On failure, message names what was found plus
    close matches (catches renames / case changes, which break Linux).
    """
    import difflib
    global ROOT, RUNS
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        return False, f"--root is not a directory: {root}"
    if not (root / "Proposed Model").is_dir():
        children = sorted(p.name for p in root.iterdir())
        hint = difflib.get_close_matches("Proposed Model", children, n=3)
        msg = (f"'Proposed Model' not found under --root={root}. "
               f"Top-level entries: {children[:12]}"
               + (f". Did you mean: {hint}?" if hint else
                  " (check exact spelling/case — Linux is case-sensitive)"))
        return False, msg
    ROOT = root
    RUNS = build_runs(root)
    sp = str(root / "Proposed Model")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return True, f"root set to {root}"


# --------------------------------------------------------------------------
# Placement map — EVERYTHING lives inside the current folder structure
# (ROOT). Nothing is read from outside it. Destinations below mirror
# student_config.py exactly. Run with --stage-csvs to auto-copy the four
# soft-label CSVs from their in-structure homes (small files); data
# images and the DINOv3 file must be placed manually (large / external).
# --------------------------------------------------------------------------
CSV_SOURCES = {
    "Proposed Model/outputs/leaf/leaf_soft_labels.csv":
        "PDDD Fruit 3 and PDDD Leaf 1/outputs/leaf/leaf_soft_labels.csv",
    "Proposed Model/outputs/pest/pest_soft_labels.csv":
        "PDDD Fruit 3 and PDDD Leaf 1/outputs/pest/pest_soft_labels.csv",
    "Proposed Model/outputs/fruit/fruit_soft_labels.csv":
        "PDDD Fruit 3 and PDDD Leaf 1/outputs/fruit/fruit_soft_labels.csv",
    # Yield teacher matching the paper (LAB-CLAHE, conf 0.20) is model 2.
    "Proposed Model/outputs/yield/yield_soft_labels.csv":
        "YOLO26s apple object detection model 2/outputs/yield/yield_soft_labels.csv",
}

DATA_PLACEMENT = [
    ("Proposed Model/data/leaf/<Apple_Mosaic|Apple___Black_rot|Alternaria|Healthy>/",
     "your original leaf images (4 class folders)"),
    ("Proposed Model/data/pest/<xylotrechus|aphids|leafhoppers|spider_mite>/",
     "copy from PDDD Pest 1/data/pest/ inside this same structure"),
    ("Proposed Model/data/fruit/<Anthracnose|Black Rot|Healthy>/",
     "your original fruit images (3 class folders)"),
    ("Proposed Model/data/yield/images/{train,val}/ + labels/{train,val}/ (YOLO .txt)",
     "your original yield images + label files (both splits)"),
    ("Proposed Model/weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
     "your original DINOv3 file (exact name; cannot be generated)"),
]

NOT_NEEDED = ("Proposed Model/final_dataset/splits/*.txt is NOT needed "
              "(verified: student_dataset.py never reads it).")


def print_placement():
    print("\n  WHERE TO PUT MISSING FILES (all paths under ROOT):")
    for dest, src in DATA_PLACEMENT:
        print(f"    -> {dest}\n       source: {src}")
    print(f"  {NOT_NEEDED}")
    print("  Checkpoints need nothing: proposed/ablA/ablB/ablC already have "
          ".pth in-structure; frozen + ablD have none and auto-skip.\n")


def stage_csvs():
    """Copy the 4 soft-label CSVs from their in-structure homes to the
    Proposed Model/outputs/... destinations. Returns (ok, messages)."""
    msgs = []
    ok_all = True
    for dest_rel, src_rel in CSV_SOURCES.items():
        src = ROOT / src_rel
        dst = ROOT / dest_rel
        if dst.exists():
            msgs.append(f"[OK] already present: {dest_rel}")
            continue
        if not src.exists():
            msgs.append(f"[MISS] source absent in-structure: {src_rel}")
            ok_all = False
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, dst)
        msgs.append(f"[STAGED] {src_rel} -> {dest_rel}")
    return ok_all, msgs

CAPTIONS = """\
Figs. 10/11/12. Confusion matrices for leaf (4 classes), pest (4 classes), and fruit (3 classes) on the {split} split. Fruit is evaluated on ground-truth-box ROIs (head diagnostic). Counts + row-normalised rates shown.

Fig. 13. Predicted versus ground-truth apple counts on the 400-image yield validation set (confidence {conf}, NMS IoU {nms}). Pearson r, R^2, and mean signed error in phase2_results/<config>/yield_metrics.json.

Figs. 14/15. Representative ground-truth (solid) versus predicted (dashed) apple boxes with per-image counts; overlays in phase2_results/<config>/overlays/.
"""


# --------------------------------------------------------------------------
# Lazy imports (so --help works even on machines without torch)
# --------------------------------------------------------------------------
def lazy_imports():
    global torch, np, plt, Patch
    global skl_cm, skl_report
    global get_config, ADCStudent, decode_detections
    global ADCDataset, adc_collate_fn, build_dataloaders, _load_yolo_boxes
    global IDX_TO_LEAF, IDX_TO_PEST, IDX_TO_FRUIT
    import numpy as _np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.patches import Rectangle as _Rect
    import torch as _torch
    from sklearn.metrics import (confusion_matrix as _cm,
                                 classification_report as _rep)
    from student_config import (get_config, IDX_TO_LEAF, IDX_TO_PEST,
                                IDX_TO_FRUIT)
    from student_model import ADCStudent, decode_detections
    from student_dataset import (ADCDataset, adc_collate_fn,
                                 build_dataloaders, _load_yolo_boxes)
    torch, np, plt = _torch, _np, _plt
    Patch = _Rect
    skl_cm, skl_report = _cm, _rep


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermilion": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
def pick_checkpoint(ckpt_dir):
    best = ckpt_dir / "best_model.pth"
    latest = ckpt_dir / "latest_checkpoint.pth"
    if best.exists():
        return best, "best"
    if latest.exists():
        return latest, "latest(fallback)"
    return None, "missing"


def preflight(device_choice):
    """Verify environment + files. Returns (ok, problems list)."""
    problems = []
    print("=" * 70)
    print("  PHASE 2 PREFLIGHT")
    print("=" * 70)
    print(f"  ROOT: {ROOT}")
    try:
        top = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    except Exception as e:  # noqa: BLE001
        problems.append(f"cannot list ROOT ({ROOT}): {e}")
        print(f"[FAIL] cannot list ROOT: {e}")
        return False, problems
    print(f"  top-level dirs: {top}")
    if "Proposed Model" not in top:
        problems.append("'Proposed Model' missing under ROOT "
                        "(wrong --root or renamed folder?)")
        print("[FAIL] 'Proposed Model' not found under ROOT.")
        return False, problems
    try:
        lazy_imports()
        print(f"[OK] torch {torch.__version__}")
    except Exception as e:  # noqa: BLE001
        problems.append(f"imports failed: {e}")
        print(f"[FAIL] imports: {e}")
        return False, problems
    if torch.cuda.is_available():
        print(f"[OK] CUDA: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        msg = "CUDA not available; will run on CPU (slow)."
        if device_choice == "cuda":
            problems.append(msg)
            print(f"[FAIL] {msg}")
            return False, problems
        print(f"[WARN] {msg}")

    try:
        cfg = get_config()
    except Exception as e:  # noqa: BLE001
        problems.append(f"get_config() failed: {e}")
        print(f"[FAIL] config: {e}")
        return False, problems

    critical = [
        ("DINOv3 weights", cfg.paths.dinov3_weights),
        ("Leaf soft CSV", cfg.paths.leaf_soft_csv),
        ("Pest soft CSV", cfg.paths.pest_soft_csv),
        ("Fruit soft CSV", cfg.paths.fruit_soft_csv),
        ("Yield soft CSV", cfg.paths.yield_soft_csv),
        ("Yield train imgs", cfg.paths.yield_img_dir),
        ("Yield train labels", cfg.paths.yield_label_dir),
    ]
    for name, p in critical:
        ok = Path(p).exists()
        print(f"[{'OK' if ok else 'MISS'}] {name}: {p}")
        if not ok:
            problems.append(f"missing {name}: {p}")

    n_ckpt = 0
    for key, meta in RUNS.items():
        ckpt, kind = pick_checkpoint(meta["ckpt_dir"])
        print(f"[{'OK' if ckpt else 'SKIP'}] {key:<9} {meta['label']:<20} "
              f"{ckpt.name if ckpt else '-'} ({kind})")
        if ckpt:
            n_ckpt += 1
    if n_ckpt == 0:
        problems.append("no evaluable checkpoints found anywhere")
    print("=" * 70)
    return len(problems) == 0, problems


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_model(cfg, ckpt_path, device):
    model = ADCStudent(cfg).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device,
                      weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt.get("model", {}),
                                                strict=False)
    model.eval()
    info = {"epoch": int(ckpt.get("epoch", -1)),
            "best_val": ckpt.get("best_val"),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected)}
    return model, info


# --------------------------------------------------------------------------
# Classification evaluation (val and/or test split)
# --------------------------------------------------------------------------
def eval_classification(model, loader, device):
    yt_leaf, yp_leaf, yt_pest, yp_pest = [], [], [], []
    yt_fruit, yp_fruit = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            roi_boxes = batch["roi_boxes"].to(device, non_blocking=True)
            roi_img_idx = batch["roi_img_idx"].to(device, non_blocking=True)
            out = model(images, gt_boxes=roi_boxes,
                        gt_box_img_idx=roi_img_idx)
            m = batch["leaf_label"] != -1
            if m.any():
                yt_leaf.extend(batch["leaf_label"][m].tolist())
                yp_leaf.extend(out["leaf_logits"][m].argmax(1).cpu().tolist())
            m = batch["pest_label"] != -1
            if m.any():
                yt_pest.extend(batch["pest_label"][m].tolist())
                yp_pest.extend(out["pest_logits"][m].argmax(1).cpu().tolist())
            if out["fruit_logits"].shape[0] > 0:
                ri = out["box_img_idx"].cpu()
                fh = batch["fruit_label"][ri]
                mf = fh != -1
                if mf.any():
                    yt_fruit.extend(fh[mf].tolist())
                    yp_fruit.extend(
                        out["fruit_logits"][mf].argmax(1).cpu().tolist())
    return {"leaf": (yt_leaf, yp_leaf), "pest": (yt_pest, yp_pest),
            "fruit": (yt_fruit, yp_fruit)}


def save_classification(outdir, split, results, idx_maps):
    """Writes JSON + CSV reports and CM figures. Returns summary dict."""
    summary = {"split": split, "n": {}, "accuracy": {}, "macro_f1": {}}
    for task, idx_map in idx_maps.items():
        yt, yp = results[task]
        names = [idx_map[i] for i in sorted(idx_map)]
        labels = sorted(idx_map)
        rep = skl_report(yt, yp, labels=labels, target_names=names,
                         output_dict=True, zero_division=0)
        cm = skl_cm(yt, yp, labels=labels)
        acc = float(np.mean(np.array(yt) == np.array(yp))) if yt else 0.0
        summary["n"][task] = len(yt)
        summary["accuracy"][task] = round(acc, 4)
        summary["macro_f1"][task] = round(rep["macro avg"]["f1-score"], 4)
        summary[task] = {
            n: {"precision": round(rep[n]["precision"], 4),
                "recall": round(rep[n]["recall"], 4),
                "f1": round(rep[n]["f1-score"], 4),
                "support": int(rep[n]["support"])}
            for n in names
        }
        with open(outdir / f"classification_{task}_{split}.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "precision", "recall", "f1", "support"])
            for n in names:
                d = summary[task][n]
                w.writerow([n, d["precision"], d["recall"], d["f1"],
                            d["support"]])
        # Figure (counts + row-normalised heatmap side by side)
        fig, axes = plt.subplots(1, 2, figsize=(7.28, 2.9),
                                 layout="constrained")
        im0 = axes[0].imshow(cm, cmap="Blues", vmin=0)
        axes[0].set_title("Counts")
        with np.errstate(invalid="ignore", divide="ignore"):
            cmn = cm.astype(float) / cm.sum(1, keepdims=True)
            cmn = np.nan_to_num(cmn)
        im1 = axes[1].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Row-normalised")
        for ax, mat, fmt in [(axes[0], cm, "d"), (axes[1], cmn, ".2f")]:
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_yticklabels(names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            thr = mat.max() / 2 if mat.max() else 0.5
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, format(mat[i, j], fmt),
                            ha="center", va="center", fontsize=6.5,
                            color="white" if mat[i, j] > thr else "black")
        fig.colorbar(im0, ax=axes[0], shrink=0.85, label="Count")
        fig.colorbar(im1, ax=axes[1], shrink=0.85, label="Rate")
        fig.suptitle(f"{task.capitalize()} confusion matrix "
                     f"({split}, n={len(yt)}, acc={acc * 100:.2f}%)")
        fig.savefig(outdir / f"cm_{task}_{split}.pdf", bbox_inches="tight",
                    pad_inches=0.02)
        fig.savefig(outdir / f"cm_{task}_{split}.png", dpi=600,
                    bbox_inches="tight", pad_inches=0.02, facecolor="white")
        plt.close(fig)
    with open(outdir / f"classification_{split}.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------
# Yield counting evaluation (val set, per-image CSV + stats + scatter)
# --------------------------------------------------------------------------
def eval_yield(model, cfg, device, outdir):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from PIL import Image

    img_dir = Path(cfg.paths.yield_img_dir).parent.parent / "images" / "val"
    lbl_dir = Path(cfg.paths.yield_label_dir).parent.parent / "labels" / "val"
    if not img_dir.exists():
        print(f"  [WARN] yield val dir missing: {img_dir}; skipping")
        return None
    tf = A.Compose([
        A.Resize(cfg.model.image_size, cfg.model.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    rows = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        gt = _load_yolo_boxes(str(lbl_dir / (p.stem + ".txt")))
        img = np.array(Image.open(p).convert("RGB"))
        t = tf(image=img)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
            boxes, _, _ = decode_detections(
                out["raw_det"], cfg.model.conf_threshold, cfg.model.nms_iou)
        rows.append((p.name, len(gt), len(boxes)))
    if not rows:
        print("  [WARN] no yield val images found; skipping")
        return None
    gt = np.array([r[1] for r in rows], dtype=float)
    pr = np.array([r[2] for r in rows], dtype=float)
    err = pr - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    pearson = float(np.corrcoef(gt, pr)[0, 1]) if len(rows) > 2 else 0.0
    if not np.isfinite(pearson):
        pearson = 0.0
    ss_res = float(np.sum((gt - pr) ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if not np.isfinite(r2):
        r2 = 0.0
    metrics = {"n_images": len(rows), "mae": round(mae, 4),
               "rmse": round(rmse, 4), "pearson": round(pearson, 4),
               "r2": round(r2, 4),
               "mean_signed_error": round(float(err.mean()), 4)}
    with open(outdir / "yield_per_image.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "gt_count", "pred_count"])
        w.writerows(rows)
    with open(outdir / "yield_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Scatter (paper Fig 13)
    fig, ax = plt.subplots(figsize=(3.54, 3.0), layout="constrained")
    mx = int(max(gt.max(), pr.max())) + 1
    ax.scatter(gt, pr, s=8, alpha=0.45, color=OI["blue"],
               edgecolors="none", label="Images")
    ax.plot([0, mx], [0, mx], color=OI["vermilion"], linewidth=1.0,
            linestyle="--", label="Ideal")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_xlabel("Ground-truth count")
    ax.set_ylabel("Predicted count")
    ax.set_title(f"Pred vs GT (n={len(rows)}, r={pearson:.3f}, "
                 f"$R^2$={r2:.3f})")
    ax.legend(frameon=True, edgecolor="black")
    ax.grid(True)
    fig.savefig(outdir / "fig13_pred_vs_gt.pdf", bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(outdir / "fig13_pred_vs_gt.png", dpi=600,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    return metrics


# --------------------------------------------------------------------------
# Qualitative overlays (paper Figs 14/15)
# --------------------------------------------------------------------------
def save_overlays(model, cfg, device, outdir, k=8):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from PIL import Image

    img_dir = Path(cfg.paths.yield_img_dir).parent.parent / "images" / "val"
    lbl_dir = Path(cfg.paths.yield_label_dir).parent.parent / "labels" / "val"
    if not img_dir.exists():
        return 0
    tf = A.Compose([
        A.Resize(cfg.model.image_size, cfg.model.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    odir = outdir / "overlays"
    odir.mkdir(parents=True, exist_ok=True)
    done = 0
    for p in sorted(img_dir.iterdir()):
        if done >= k:
            break
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        gt = _load_yolo_boxes(str(lbl_dir / (p.stem + ".txt")))
        img_np = np.array(Image.open(p).convert("RGB"))
        t = tf(image=img_np)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
            boxes, scores, _ = decode_detections(
                out["raw_det"], cfg.model.conf_threshold, cfg.model.nms_iou)
        vis = (t[0].cpu().numpy() * std + mean).clip(0, 1).transpose(1, 2, 0)
        H, W = vis.shape[:2]
        fig, ax = plt.subplots(figsize=(3.54, 3.54), layout="constrained")
        ax.imshow(vis)
        for b in np.asarray(gt):
            cx, cy, w, h = b
            ax.add_patch(Patch(((cx - w / 2) * W, (cy - h / 2) * H), w * W,
                               h * H, linewidth=1.5, edgecolor="#00C000",
                               facecolor="none", label="GT"))
        for b in boxes.cpu().numpy():
            cx, cy, w, h = b
            ax.add_patch(Patch(((cx - w / 2) * W, (cy - h / 2) * H), w * W,
                               h * H, linewidth=1.2, edgecolor="#D55E00",
                               linestyle="--", facecolor="none",
                               label="Pred"))
        handles, labels = ax.get_legend_handles_labels()
        seen, uh, ul = set(), [], []
        for h_, l_ in zip(handles, labels):
            if l_ not in seen:
                seen.add(l_)
                uh.append(h_)
                ul.append(l_)
        ax.legend(uh, ul, frameon=True, edgecolor="black", fontsize=7,
                  loc="upper right")
        ax.set_title(f"{p.name}  GT:{len(gt)} Pred:{len(boxes)}")
        ax.axis("off")
        fig.savefig(odir / f"bbox_{done:02d}_{p.stem}.png", dpi=300,
                    bbox_inches="tight", pad_inches=0.02, facecolor="white")
        plt.close(fig)
        done += 1
    return done


# --------------------------------------------------------------------------
# Per-config driver
# --------------------------------------------------------------------------
def run_config(key, meta, cfg, device, out_root, splits, batch, workers,
               n_overlays):
    from torch.utils.data import DataLoader

    outdir = out_root / key
    outdir.mkdir(parents=True, exist_ok=True)
    report = {"key": key, "label": meta["label"]}
    ckpt_path, kind = pick_checkpoint(meta["ckpt_dir"])
    if ckpt_path is None:
        report["status"] = "skipped(no checkpoint)"
        print(f"[{key}] SKIP — no best/latest checkpoint in "
              f"{meta['ckpt_dir']}")
        (outdir / "RUN_REPORT.txt").write_text(
            "SKIPPED: no best_model.pth or latest_checkpoint.pth found.\n",
            encoding="utf-8")
        return report
    print(f"[{key}] checkpoint: {ckpt_path.name} ({kind})")
    try:
        model, info = load_model(cfg, ckpt_path, device)
    except Exception as e:  # noqa: BLE001
        report["status"] = f"load-failed: {e}"
        print(f"[{key}] LOAD FAILED: {e}")
        (outdir / "RUN_REPORT.txt").write_text(f"LOAD FAILED: {e}\n",
                                               encoding="utf-8")
        return report
    report.update({"status": f"evaluated({kind})", **info})
    print(f"[{key}] loaded (epoch {info['epoch']}, "
          f"missing={info['missing_keys']}, unexpected={info['unexpected_keys']})")

    idx_maps = {"leaf": IDX_TO_LEAF, "pest": IDX_TO_PEST,
                "fruit": IDX_TO_FRUIT}
    pin = cfg.data.pin_memory and device.type == "cuda"
    try:
        if "val" in splits:
            _, val_loader = build_dataloaders(cfg)
            # enforce safe batch/workers without touching global cfg file
            val_loader = DataLoader(
                val_loader.dataset, batch_size=batch, shuffle=False,
                num_workers=workers, pin_memory=pin,
                collate_fn=adc_collate_fn)
            res = eval_classification(model, val_loader, device)
            s = save_classification(outdir, "val", res, idx_maps)
            report["val"] = s
            print(f"[{key}] val: " + " ".join(
                f"{t}={s['accuracy'][t] * 100:.2f}%"
                for t in ["leaf", "pest", "fruit"]))
        if "test" in splits:
            test_ds = ADCDataset(cfg, split="test")
            test_loader = DataLoader(
                test_ds, batch_size=batch, shuffle=False,
                num_workers=workers, pin_memory=pin,
                collate_fn=adc_collate_fn)
            res = eval_classification(model, test_loader, device)
            s = save_classification(outdir, "test", res, idx_maps)
            report["test"] = s
            print(f"[{key}] test: " + " ".join(
                f"{t}={s['accuracy'][t] * 100:.2f}%"
                for t in ["leaf", "pest", "fruit"]))
        ym = eval_yield(model, cfg, device, outdir)
        if ym:
            report["yield"] = ym
            print(f"[{key}] yield: MAE {ym['mae']} RMSE {ym['rmse']} "
                  f"r {ym['pearson']} R2 {ym['r2']} (n={ym['n_images']})")
        n_ov = save_overlays(model, cfg, device, outdir, k=n_overlays)
        report["overlays"] = n_ov
        print(f"[{key}] overlays: {n_ov}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            report["status"] += "|OOM — rerun with --batch 4 (or 2)"
            print(f"[{key}] OOM — rerun with a smaller --batch.")
        else:
            report["status"] += f"|runtime-error: {e}"
            print(f"[{key}] RUNTIME ERROR: {e}")
    finally:
        (outdir / "RUN_REPORT.txt").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Phase 2 GPU-side evaluation")
    ap.add_argument("--out", default="phase2_results")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=["val", "test"],
                    choices=["val", "test"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--num-overlays", type=int, default=8)
    ap.add_argument("--root", default=None,
                    help="Project root (folder containing 'Proposed Model'). "
                         "Defaults to the script's own folder. Example Linux: "
                         "--root /home/CL407-33/Downloads/BTechProject-20260806T081318Z-1-001/BTechProject/models")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--stage-csvs", action="store_true",
                    help="Copy the 4 soft-label CSVs from their in-structure "
                         "homes into Proposed Model/outputs/... then continue.")
    args = ap.parse_args()

    if args.root:
        ok_root, msg_root = set_root(args.root)
        print(msg_root)
        if not ok_root:
            print("PREFLIGHT FAILED:")
            print(f"  - {msg_root}")
            sys.exit(2)

    if args.stage_csvs:
        ok_stage, msgs = stage_csvs()
        for m in msgs:
            print(f"  {m}")
        if not ok_stage:
            print("  [--stage-csvs incomplete: see MISS lines above]")

    ok, problems = preflight(args.device if args.device != "auto" else "auto")
    if args.preflight_only:
        if not ok:
            print("PREFLIGHT FAILED:")
            for pr in problems:
                print(f"  - {pr}")
            print_placement()
            sys.exit(2)
        print("PREFLIGHT PASSED — ready for full run.")
        return
    if not ok:
        print("PREFLIGHT found blocking problems "
              "(re-run with --preflight-only for details):")
        for pr in problems:
            print(f"  - {pr}")
        print_placement()
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != "y":
            sys.exit(2)

    lazy_imports()
    apply_style()
    if args.device == "cuda" or (args.device == "auto"
                                 and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        warnings.warn("running on CPU — expect slow inference")
    print(f"Device: {device}")
    torch.manual_seed(42)

    cfg = get_config()
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    keys = [k for k in RUNS if (args.configs is None or k in args.configs)]
    rollup = {}
    for key in keys:
        rollup[key] = run_config(key, RUNS[key], cfg, device, out_root,
                                 args.splits, args.batch, args.workers,
                                 args.num_overlays)
    with open(out_root / "phase2_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(rollup, f, indent=2, default=str)
    with open(out_root / "phase2_captions.txt", "w",
              encoding="utf-8") as f:
        f.write(CAPTIONS.format(split="+".join(args.splits),
                                conf=cfg.model.conf_threshold,
                                nms=cfg.model.nms_iou))
    print("\n" + "=" * 70)
    print("  PHASE 2 COMPLETE — per-config status:")
    for key in keys:
        print(f"    {key:<9} {rollup[key].get('status')}")
    print(f"  Results in: {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
