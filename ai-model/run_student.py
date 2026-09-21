"""
run_student.py
==============
Entry point for AgroDistill-Cascade (ADC) student training + ONNX export.

USAGE:
  # Train from scratch
  python run_student.py

  # Resume from latest checkpoint
  python run_student.py --resume

  # Export best model to ONNX only (skip training)
  python run_student.py --export-only

  # Override specific config values
  python run_student.py --epochs 50 --batch 4

OUTPUTS (in outputs/student/):
  checkpoints/best_model.pth          best validation checkpoint
  checkpoints/latest_checkpoint.pth   rolling safety checkpoint
  training_history.json               per-epoch loss/metric log
  adc_student.onnx                    deployment model

REQUIRES (all in same folder):
  student_config.py
  student_backbone.py
  student_model.py
  student_dataset.py
  student_loss.py
  student_train.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# ── Dependency check ──────────────────────────────────────────────────────────
_REQUIRED = ["torch", "torchvision", "albumentations", "PIL", "sklearn", "numpy"]
for pkg in _REQUIRED:
    try:
        __import__(pkg if pkg != "PIL" else "PIL.Image")
    except ImportError:
        print(f"[ERROR] Missing package: {pkg}")
        print("        Run: pip install torch torchvision albumentations "
              "Pillow scikit-learn numpy")
        sys.exit(1)

from student_config import ADCConfig, get_config
from student_model import ADCStudent
from student_train import ADCTrainer, export_onnx


# ===========================================================================
#  ARGUMENT PARSER
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train / export AgroDistill-Cascade (ADC) student model")

    # Mode
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest_checkpoint.pth")
    p.add_argument("--export-only", action="store_true",
                   help="Skip training; load best_model.pth and export ONNX")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a specific checkpoint to resume from or export")

    # Config overrides (most common ones)
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch",       type=int,   default=None,
                   help="Per-GPU batch size (effective = batch × accum_steps)")
    p.add_argument("--accum",       type=int,   default=None,
                   help="Gradient accumulation steps")
    p.add_argument("--lr-lora",     type=float, default=None)
    p.add_argument("--lr-heads",    type=float, default=None)
    p.add_argument("--freeze-ep",   type=int,   default=None,
                   help="Number of Phase-1 (LoRA frozen) epochs")
    p.add_argument("--patience",    type=int,   default=None)
    p.add_argument("--workers",     type=int,   default=None)

    # Paths
    p.add_argument("--output-dir",  type=str,   default=None,
                   help="Override student output directory")

    return p.parse_args()


# ===========================================================================
#  CONFIG BUILDER
# ===========================================================================

def build_config(args) -> ADCConfig:
    cfg = get_config()

    # Apply CLI overrides
    if args.epochs     is not None: cfg.train.epochs              = args.epochs
    if args.batch      is not None: cfg.train.batch_size          = args.batch
    if args.accum      is not None: cfg.train.accum_steps         = args.accum
    if args.lr_lora    is not None: cfg.train.lr_lora             = args.lr_lora
    if args.lr_heads   is not None: cfg.train.lr_heads            = args.lr_heads
    if args.freeze_ep  is not None: cfg.train.freeze_backbone_epochs = args.freeze_ep
    if args.patience   is not None: cfg.train.patience            = args.patience
    if args.workers    is not None: cfg.data.num_workers          = args.workers
    if args.output_dir is not None:
        cfg.paths.student_output_dir = args.output_dir
        cfg.paths.checkpoint_dir     = str(Path(args.output_dir) / "checkpoints")
        cfg.paths.onnx_path          = str(Path(args.output_dir) / "adc_student.onnx")

    cfg.make_dirs()
    return cfg


# ===========================================================================
#  PRE-FLIGHT CHECKS
# ===========================================================================

def preflight(cfg: ADCConfig, export_only: bool = False):
    print("\n" + "=" * 70)
    print("  AgroDistill-Cascade (ADC) — Student Model")
    print("=" * 70)

    tc = cfg.train
    mc = cfg.model
    pc = cfg.paths

    print(f"\n  Model config:")
    print(f"    Image size     : {mc.image_size}×{mc.image_size}")
    print(f"    Neck channels  : {mc.neck_channels}")
    print(f"    Grid size      : {mc.grid_size}×{mc.grid_size}")
    print(f"    ROI output     : {mc.roi_output_size}×{mc.roi_output_size}")
    print(f"    Classes        : leaf={mc.num_leaf_classes} "
          f"pest={mc.num_pest_classes} "
          f"fruit={mc.num_fruit_classes} "
          f"yield={mc.num_yield_classes}")

    print(f"\n  Training config:")
    print(f"    Epochs         : {tc.epochs}  "
          f"(Phase1={tc.freeze_backbone_epochs}, "
          f"Phase2={tc.epochs - tc.freeze_backbone_epochs})")
    print(f"    Batch / accum  : {tc.batch_size} × {tc.accum_steps} "
          f"= {tc.batch_size * tc.accum_steps} effective")
    print(f"    LR LoRA        : {tc.lr_lora}")
    print(f"    LR Heads+Neck  : {tc.lr_heads}")
    print(f"    Patience       : {tc.patience}")
    print(f"    alpha (distil) : {tc.alpha}")

    print(f"\n  Paths:")
    print(f"    DINOv3 weights : {pc.dinov3_weights}")
    print(f"    Leaf soft CSV  : {pc.leaf_soft_csv}")
    print(f"    Pest soft CSV  : {pc.pest_soft_csv}")
    print(f"    Fruit soft CSV : {pc.fruit_soft_csv}")
    print(f"    Yield soft CSV : {pc.yield_soft_csv}")
    print(f"    Output dir     : {pc.student_output_dir}")
    print(f"    ONNX path      : {pc.onnx_path}")

    # GPU info
    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU : {gpu} ({vram:.1f} GB VRAM)")
    else:
        print("\n  [WARN] CUDA not available — training on CPU will be slow.")

    # Validate critical paths (warns but does not exit — let trainer handle)
    if not export_only:
        errors = []
        if not os.path.exists(pc.dinov3_weights):
            errors.append(f"DINOv3 weights missing: {pc.dinov3_weights}")
        for csv_label, csv_path in [
            ("Leaf soft",  pc.leaf_soft_csv),
            ("Pest soft",  pc.pest_soft_csv),
            ("Fruit soft", pc.fruit_soft_csv),
            ("Yield soft", pc.yield_soft_csv),
        ]:
            if not os.path.exists(csv_path):
                errors.append(f"{csv_label} CSV missing: {csv_path}")

        if errors:
            print("\n  [WARN] Missing files (training may fail):")
            for e in errors:
                print(f"    ✗ {e}")
        else:
            print("\n  [OK] All critical files found.")

    print("=" * 70 + "\n")


# ===========================================================================
#  EXPORT ONLY
# ===========================================================================

def run_export(cfg: ADCConfig, checkpoint_path: str = None):
    """Load best checkpoint and export to ONNX."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("        Train the model first or pass --checkpoint path/to/ckpt.pth")
        sys.exit(1)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    model = ADCStudent(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[INFO] Checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt.get('best_val', 'N/A')})")

    export_onnx(model, cfg)
    print("[INFO] Export complete.")


# ===========================================================================
#  EVALUATION SUMMARY  (called after training)
# ===========================================================================

def print_training_summary(cfg: ADCConfig):
    hist_path = os.path.join(cfg.paths.student_output_dir,
                             "training_history.json")
    if not os.path.exists(hist_path):
        return

    with open(hist_path) as f:
        history = json.load(f)

    if not history:
        return

    # Find best epoch
    best = min(history, key=lambda r: r.get("val_loss_total", float("inf")))

    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Total epochs run   : {len(history)}")
    print(f"  Best epoch         : {best['epoch']} "
          f"(val_loss={best.get('val_loss_total', 'N/A'):.4f})")
    print(f"\n  Best epoch metrics:")

    for key in ["val_acc_leaf", "val_acc_pest", "val_acc_fruit",
                "val_loss_leaf", "val_loss_pest",
                "val_loss_fruit", "val_loss_yield"]:
        v = best.get(key)
        if v is not None:
            print(f"    {key:<25}: {v:.4f}")

    print(f"\n  Homoscedastic weights at best epoch:")
    for task, w in best.get("weights", {}).items():
        print(f"    {task:<10}: precision = {w:.4f}")

    print(f"\n  Checkpoints:")
    best_ckpt   = os.path.join(cfg.paths.checkpoint_dir, "best_model.pth")
    latest_ckpt = os.path.join(cfg.paths.checkpoint_dir,
                               "latest_checkpoint.pth")
    print(f"    Best   : {best_ckpt}  "
          f"[{'✓' if os.path.exists(best_ckpt) else '✗'}]")
    print(f"    Latest : {latest_ckpt}  "
          f"[{'✓' if os.path.exists(latest_ckpt) else '✗'}]")
    print(f"    ONNX   : {cfg.paths.onnx_path}  "
          f"[{'✓' if os.path.exists(cfg.paths.onnx_path) else '✗'}]")
    print("=" * 70)

    print("\n  NEXT STEPS:")
    print("  1. Use best_model.pth for inference / further evaluation.")
    print("  2. Use adc_student.onnx for mobile/edge deployment.")
    print("  3. Run confusion matrix visualisation (visualizingCode.txt).")
    print("  4. Report per-class F1 + mAP@0.5 in paper Section 5.")


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    args = parse_args()
    cfg  = build_config(args)

    # ── Export only ───────────────────────────────────────────────────────────
    if args.export_only:
        preflight(cfg, export_only=True)
        run_export(cfg, checkpoint_path=args.checkpoint)
        return

    # ── Full training ─────────────────────────────────────────────────────────
    preflight(cfg, export_only=False)

    trainer = ADCTrainer(cfg)

    if args.resume:
        ckpt_path = args.checkpoint or os.path.join(
            cfg.paths.checkpoint_dir, "latest_checkpoint.pth")
        trainer.resume(ckpt_path)

    t_start = time.time()
    model   = trainer.train()
    elapsed = time.time() - t_start

    print(f"\n[INFO] Total training time: {elapsed/3600:.2f} hours")

    # ── Auto-export ONNX after training ──────────────────────────────────────
    best_path = os.path.join(cfg.paths.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_path):
        print("\n[INFO] Auto-exporting best model to ONNX...")
        ckpt = torch.load(best_path,
                          map_location=torch.device("cpu"),
                          weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        export_onnx(model, cfg)
    else:
        print("[WARN] best_model.pth not found — ONNX export skipped.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_training_summary(cfg)


if __name__ == "__main__":
    main()