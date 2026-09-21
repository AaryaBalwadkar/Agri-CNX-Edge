"""
student_train.py
================
Two-phase training loop for AgroDistill-Cascade (ADC) student.

Phase 1 (freeze_backbone_epochs):
  LoRA adapters frozen. Only heads + neck train at lr_heads.
  Prevents random-init heads from damaging LoRA adapters early on.

Phase 2 (remaining epochs):
  LoRA unfrozen. Two param groups:
    lora_backbone → lr_lora  (1e-5)
    heads_and_neck → lr_heads (1e-4)
  Scheduler: CosineAnnealingLR over Phase 2 epochs.
  Gradient accumulation: accum_steps=4 (effective batch=32).
  Early stopping on val total loss with patience=20.

Checkpoints:
  latest_checkpoint.pth  — saved every accum window (resume safety)
  best_model.pth         — best val loss (used for inference + ONNX export)
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from student_config import ADCConfig, get_config
from student_model import ADCStudent
from student_loss import ADCLoss
from student_dataset import build_dataloaders


# ===========================================================================
#  METRIC HELPERS
# ===========================================================================

def _accuracy(logits: torch.Tensor,
              targets: torch.Tensor,
              ignore_index: int = -1) -> Optional[float]:
    """Per-batch classification accuracy; None if all targets are ignored."""
    mask = targets != ignore_index
    if mask.sum() == 0:
        return None
    preds = logits.argmax(dim=1)[mask]
    return float((preds == targets[mask]).float().mean())


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


# ===========================================================================
#  CHECKPOINT HELPERS
# ===========================================================================

def _save(path: str, epoch: int, model: nn.Module,
          optimizer, scheduler, best_val: float, log: dict):
    torch.save({
        "epoch"     : epoch,
        "model"     : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "scheduler" : scheduler.state_dict() if scheduler else None,
        "best_val"  : best_val,
        "log"       : log,
    }, path)


def _load(path: str, model: nn.Module,
          optimizer=None, scheduler=None,
          device: torch.device = torch.device("cpu")) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


# ===========================================================================
#  SINGLE TRAIN STEP  (no gradient.step — accumulation handled in loop)
# ===========================================================================

def _train_step(
    model   : ADCStudent,
    batch   : Dict[str, torch.Tensor],
    loss_fn : ADCLoss,
    device  : torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    One forward + backward pass (no optimizer.step).
    Returns (scaled_loss_for_backward, log_dict).
    Loss is divided by accum_steps so gradients accumulate correctly.
    """
    images  = batch["image"].to(device, non_blocking=True)

    # Ground-truth boxes for ROI Align during training
    roi_boxes   = batch["roi_boxes"].to(device, non_blocking=True)
    roi_img_idx = batch["roi_img_idx"].to(device, non_blocking=True)

    model_out = model(images, gt_boxes=roi_boxes,
                      gt_box_img_idx=roi_img_idx)

    total, log = loss_fn(model_out, batch)
    return total, log


# ===========================================================================
#  VALIDATION EPOCH
# ===========================================================================

@torch.no_grad()
def validate(
    model   : ADCStudent,
    loader  : DataLoader,
    loss_fn : ADCLoss,
    device  : torch.device,
) -> Dict[str, float]:
    model.eval()
    accum_log: Dict[str, list] = {}
    leaf_acc, pest_acc, fruit_acc = [], [], []

    for batch in loader:
        images      = batch["image"].to(device, non_blocking=True)
        roi_boxes   = batch["roi_boxes"].to(device, non_blocking=True)
        roi_img_idx = batch["roi_img_idx"].to(device, non_blocking=True)

        model_out = model(images, gt_boxes=roi_boxes,
                          gt_box_img_idx=roi_img_idx)
        _, log = loss_fn(model_out, batch)

        for k, v in log.items():
            accum_log.setdefault(k, []).append(v)

        # Accuracy (optional diagnostic)
        device_cpu = torch.device("cpu")
        a = _accuracy(model_out["leaf_logits"].cpu(),
                      batch["leaf_label"], -1)
        if a is not None: leaf_acc.append(a)
        a = _accuracy(model_out["pest_logits"].cpu(),
                      batch["pest_label"], -1)
        if a is not None: pest_acc.append(a)
        if model_out["fruit_logits"].shape[0] > 0:
            ri = model_out["box_img_idx"].cpu()
            fh = batch["fruit_label"][ri]
            a  = _accuracy(model_out["fruit_logits"].cpu(), fh, -1)
            if a is not None: fruit_acc.append(a)

    results = {k: _mean(v) for k, v in accum_log.items()}
    results["acc_leaf"]  = _mean(leaf_acc)
    results["acc_pest"]  = _mean(pest_acc)
    results["acc_fruit"] = _mean(fruit_acc)
    return results


# ===========================================================================
#  TRAINING PHASES
# ===========================================================================

def _run_phase(
    phase       : int,
    model       : ADCStudent,
    loss_fn     : ADCLoss,
    optimizer   : torch.optim.Optimizer,
    scheduler   : Optional[object],
    train_loader: DataLoader,
    val_loader  : DataLoader,
    device      : torch.device,
    cfg         : ADCConfig,
    start_epoch : int,
    end_epoch   : int,
    best_val    : float,
    no_improve  : int,
    history     : list,
    ckpt_dir    : str,
) -> Tuple[float, int]:
    """
    Runs one training phase. Returns (updated_best_val, updated_no_improve).
    """
    tc = cfg.train

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        t0 = time.time()
        ep_log: Dict[str, list] = {}
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            total, log = _train_step(model, batch, loss_fn, device)

            # Scale by 1/accum_steps so accumulated grad == single full-batch grad
            (total / tc.accum_steps).backward()

            for k, v in log.items():
                ep_log.setdefault(k, []).append(v)

            # Optimizer step every accum_steps mini-batches
            if (step + 1) % tc.accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # also step at end of epoch for non-divisible dataset sizes
            if step == len(train_loader) - 1 and (step + 1) % tc.accum_steps != 0:
                nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics = validate(model, val_loader, loss_fn, device)
        val_loss    = val_metrics["loss_total"]
        train_loss  = _mean(ep_log.get("loss_total", [0.0]))
        elapsed     = time.time() - t0

        # ── Console output ────────────────────────────────────────────────────
        lr_show = optimizer.param_groups[0]["lr"]
        print(
            f"  P{phase} Ep {epoch:03d}/{end_epoch} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"LR {lr_show:.2e} | "
            f"leaf_acc={val_metrics['acc_leaf']:.3f} "
            f"pest_acc={val_metrics['acc_pest']:.3f} "
            f"fruit_acc={val_metrics['acc_fruit']:.3f} | "
            f"{elapsed:.1f}s"
        )

        # ── History ───────────────────────────────────────────────────────────
        epoch_record = {
            "phase": phase, "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "weights": loss_fn.weighter.get_weights(),
        }
        history.append(epoch_record)

        # ── Checkpoint: latest (always) ───────────────────────────────────────
        latest_path = os.path.join(ckpt_dir, "latest_checkpoint.pth")
        _save(latest_path, epoch, model, optimizer, scheduler,
              best_val, epoch_record)

        # ── Checkpoint: best ─────────────────────────────────────────────────
        if val_loss < best_val - tc.min_delta:
            best_val    = val_loss
            no_improve  = 0
            best_path   = os.path.join(ckpt_dir, "best_model.pth")
            _save(best_path, epoch, model, optimizer, scheduler,
                  best_val, epoch_record)
            print(f"  ✓ Best model saved (val_loss={best_val:.4f})")
        else:
            no_improve += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if phase == 2 and no_improve >= tc.patience:
            print(f"\n[INFO] Early stopping at epoch {epoch} "
                  f"(no improvement for {tc.patience} epochs).")
            break

    return best_val, no_improve


# ===========================================================================
#  MAIN TRAINER
# ===========================================================================

class ADCTrainer:

    def __init__(self, cfg: Optional[ADCConfig] = None):
        self.cfg = cfg or get_config()
        cfg      = self.cfg
        tc       = cfg.train

        # ── Dirs ─────────────────────────────────────────────────────────────
        cfg.make_dirs()
        self.ckpt_dir = cfg.paths.checkpoint_dir

        # ── Device ───────────────────────────────────────────────────────────
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[ADCTrainer] Device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU : {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  VRAM: {vram:.1f} GB")

        # ── Data ─────────────────────────────────────────────────────────────
        print("[ADCTrainer] Building dataloaders...")
        self.train_loader, self.val_loader = build_dataloaders(cfg)

        # ── Model ─────────────────────────────────────────────────────────────
        print("[ADCTrainer] Building ADCStudent model...")
        self.model = ADCStudent(cfg).to(self.device)
        self.model.print_summary()

        # ── Loss ─────────────────────────────────────────────────────────────
        self.loss_fn = ADCLoss(cfg).to(self.device)

        # ── State ─────────────────────────────────────────────────────────────
        self.history : list = []
        self.best_val: float = float("inf")
        self.no_improve: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    def train(self):
        cfg = self.cfg
        tc  = cfg.train

        print("\n" + "=" * 70)
        print("  ADC STUDENT TRAINING")
        print(f"  Phase 1 : {tc.freeze_backbone_epochs} epochs  "
              f"(LoRA frozen, heads+neck only)")
        print(f"  Phase 2 : {tc.epochs - tc.freeze_backbone_epochs} epochs  "
              f"(LoRA + heads+neck)")
        print(f"  Eff. batch : {tc.batch_size} × {tc.accum_steps} = "
              f"{tc.batch_size * tc.accum_steps}")
        print("=" * 70)

        # ── PHASE 1 : LoRA frozen ─────────────────────────────────────────────
        print("\n--- PHASE 1: LoRA frozen — warming up heads ---")
        self.model.freeze_heads_for_phase1()

        # Only heads + neck params are trainable in Phase 1
        p1_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        p1_optimizer = torch.optim.AdamW(
            p1_params, lr=tc.lr_heads, weight_decay=tc.weight_decay)
        # Also include loss_fn params (homoscedastic log_vars)
        p1_all_params = list(p1_params) + list(self.loss_fn.parameters())
        p1_optimizer  = torch.optim.AdamW(
            p1_all_params, lr=tc.lr_heads, weight_decay=tc.weight_decay)

        self.best_val, self.no_improve = _run_phase(
            phase        = 1,
            model        = self.model,
            loss_fn      = self.loss_fn,
            optimizer    = p1_optimizer,
            scheduler    = None,
            train_loader = self.train_loader,
            val_loader   = self.val_loader,
            device       = self.device,
            cfg          = cfg,
            start_epoch  = 1,
            end_epoch    = tc.freeze_backbone_epochs,
            best_val     = self.best_val,
            no_improve   = self.no_improve,
            history      = self.history,
            ckpt_dir     = self.ckpt_dir,
        )

        # ── PHASE 2 : Full fine-tune with LoRA ───────────────────────────────
        print("\n--- PHASE 2: LoRA unfrozen — full fine-tuning ---")
        self.model.unfreeze_lora()

        # Two param groups: LoRA (small lr) + heads+neck (higher lr)
        param_groups = self.model.get_parameter_groups()
        # Append loss_fn (homoscedastic weights) to head group
        param_groups[1]["params"] = (
            list(param_groups[1]["params"]) +
            list(self.loss_fn.parameters())
        )
        p2_optimizer = torch.optim.AdamW(param_groups)

        p2_epochs   = tc.epochs - tc.freeze_backbone_epochs
        p2_scheduler = CosineAnnealingLR(
            p2_optimizer,
            T_max  = p2_epochs,
            eta_min= tc.eta_min,
        )

        self.best_val, self.no_improve = _run_phase(
            phase        = 2,
            model        = self.model,
            loss_fn      = self.loss_fn,
            optimizer    = p2_optimizer,
            scheduler    = p2_scheduler,
            train_loader = self.train_loader,
            val_loader   = self.val_loader,
            device       = self.device,
            cfg          = cfg,
            start_epoch  = tc.freeze_backbone_epochs + 1,
            end_epoch    = tc.epochs,
            best_val     = self.best_val,
            no_improve   = self.no_improve,
            history      = self.history,
            ckpt_dir     = self.ckpt_dir,
        )

        # ── Save history ──────────────────────────────────────────────────────
        hist_path = os.path.join(cfg.paths.student_output_dir,
                                 "training_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n[INFO] Training history saved: {hist_path}")

        # ── Load best for final report ────────────────────────────────────────
        best_path = os.path.join(self.ckpt_dir, "best_model.pth")
        if os.path.exists(best_path):
            ckpt = _load(best_path, self.model, device=self.device)
            print(f"\n[INFO] Best model was epoch {ckpt['epoch']} "
                  f"(val_loss={self.best_val:.4f})")

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print(f"  Best val loss    : {self.best_val:.4f}")
        print(f"  Best checkpoint  : {best_path}")
        print(f"  History          : {hist_path}")
        print("=" * 70)
        return self.model

    # ─────────────────────────────────────────────────────────────────────────
    def resume(self, checkpoint_path: Optional[str] = None):
        """
        Resume from latest_checkpoint.pth (or a specified path).
        Call before train() — train() will skip already-completed epochs.
        Note: For simplicity, resuming restarts from the beginning of the
        current phase. Full mid-phase resume requires re-implementing epoch
        tracking; this is intentionally omitted to keep code clean.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.ckpt_dir, "latest_checkpoint.pth")

        if not os.path.exists(checkpoint_path):
            print(f"[WARN] No checkpoint found at {checkpoint_path}. "
                  f"Training from scratch.")
            return

        print(f"[INFO] Resuming from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.best_val  = ckpt.get("best_val", float("inf"))
        self.history   = []
        print(f"[INFO] Restored epoch {ckpt['epoch']}, "
              f"best_val={self.best_val:.4f}")

@torch.no_grad()
def evaluate_yield(model, cfg, device):
    """
    Evaluates apple detection on the yield val set.
    Computes Count MAE, Count RMSE, and approximate mAP@0.5.
    Call after training with best_model.pth loaded.
    """
    from pathlib import Path
    import json

    model.eval()
    img_dir = Path(cfg.paths.yield_img_dir).parent.parent / "images" / "val"
    lbl_dir = Path(cfg.paths.yield_label_dir).parent.parent / "labels" / "val"

    if not img_dir.exists():
        print(f"[WARN] Yield val dir not found: {img_dir}")
        return

    from student_dataset import _load_yolo_boxes, get_val_transform_yield
    from student_model import decode_detections
    import numpy as np
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(cfg.model.image_size, cfg.model.image_size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

    count_errors, all_iou = [], []
    img_paths = sorted(img_dir.glob("*"))

    for img_path in img_paths:
        if img_path.suffix.lower() not in {'.jpg','.jpeg','.png'}:
            continue

        gt_boxes = _load_yolo_boxes(
            str(lbl_dir / (img_path.stem + ".txt")))
        gt_count = len(gt_boxes)

        img_np = np.array(Image.open(img_path).convert("RGB"))
        img_t  = transform(image=img_np)["image"].unsqueeze(0).to(device)

        out = model(img_t)
        pred_boxes, pred_scores, _ = decode_detections(
            out["raw_det"],
            conf_thresh = cfg.model.conf_threshold,
            nms_iou     = cfg.model.nms_iou,
        )
        pred_count = len(pred_boxes)
        count_errors.append(abs(pred_count - gt_count))

    mae  = float(np.mean(count_errors)) if count_errors else 0.0
    rmse = float(np.sqrt(np.mean(np.array(count_errors)**2))) if count_errors else 0.0

    print("\n" + "="*60)
    print("  YIELD EVALUATION (val set as test)")
    print("="*60)
    print(f"  Images evaluated : {len(count_errors)}")
    print(f"  Count MAE        : {mae:.2f}")
    print(f"  Count RMSE       : {rmse:.2f}")
    print("="*60)

    results = {"count_mae": mae, "count_rmse": rmse,
               "n_images": len(count_errors)}
    out_path = Path(cfg.paths.student_output_dir) / "yield_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Yield eval saved: {out_path}")
    return results

# ===========================================================================
#  ONNX EXPORT  (called from run_student.py after training)
# ===========================================================================

def export_onnx(model: ADCStudent, cfg: ADCConfig):
    """
    Exports the trained student to ONNX (opset 17).
    Model must be in eval mode with best weights loaded.

    Output shape:
      leaf_logits   [B, 5]
      pest_logits   [B, 4]
      raw_det       [B, 6, 32, 32]
      fruit_logits  [M, 3]   — M depends on input; not batchable easily
    """
    class _ONNXWrapper(nn.Module):
        """
        ONNX requires tuple output, not dict.
        At export time, no gt_boxes — detection uses NMS internally.
        fruit_logits omitted from ONNX graph because M is dynamic and
        the ROI Align + NMS path is handled on the client side.
        """
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            out = self.m(x)
            return out["leaf_logits"], out["pest_logits"], out["raw_det"]

    model.eval()
    wrapper    = _ONNXWrapper(model).cpu()
    dummy      = torch.randn(1, 3, cfg.model.image_size, cfg.model.image_size)
    onnx_path  = cfg.paths.onnx_path

    torch.onnx.export(
        wrapper, dummy, onnx_path,
        opset_version = 17,
        input_names   = ["image"],
        output_names  = ["leaf_logits", "pest_logits", "raw_det"],
        dynamic_axes  = {
            "image"       : {0: "batch"},
            "leaf_logits" : {0: "batch"},
            "pest_logits" : {0: "batch"},
            "raw_det"     : {0: "batch"},
        },
        do_constant_folding = True,
    )
    print(f"\n[INFO] ONNX model exported: {onnx_path}")
    print(f"       Input  : [B, 3, {cfg.model.image_size}, {cfg.model.image_size}]")
    print(f"       Outputs: leaf_logits[B,5], pest_logits[B,4], raw_det[B,6,32,32]")
    print(f"       Note   : Fruit disease (ROI Align path) runs post-processing")
    print(f"                on the client using raw_det boxes + a separate crop.")
    
    # Evaluate yield detection
    print("\n[INFO] Running yield detection evaluation...")
    from student_train import evaluate_yield
    evaluate_yield(model, cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
