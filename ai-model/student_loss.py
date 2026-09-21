"""
student_loss.py
===============
All loss components for AgroDistill-Cascade (ADC) student training.

Four task losses, each blending hard-label supervision with teacher
distillation, combined by HomoscedasticUncertaintyWeighter.

Classification tasks (leaf / pest / fruit):
  L_cls = alpha * FocalLoss(student_logits, hard_label)
        + (1 - alpha) * KL(student_probs || teacher_probs)
  Teacher probs are TTA-averaged at T=1 → use directly, no temperature scaling.
  ignore_index=-1 masks images where the task does not apply.

Detection task (yield):
  L_det = lambda_box * CIoU
        + lambda_obj * BCE_objectness
        + lambda_cls * BCE_class
  Applied to ground-truth assignment AND (scaled by beta) to teacher boxes.

Total loss:
  L_total = sum_k [ exp(-log_var_k) * L_k + 0.5 * log_var_k ]
  where log_var_k is a learned scalar per task (Kendall & Gal 2018).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from student_config import ADCConfig


# ===========================================================================
#  FOCAL LOSS
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Lin et al. 2017.
    L = -(1-p_t)^gamma * log(p_t)
    Masks samples where target == ignore_index.
    """

    def __init__(self, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.ce           = nn.CrossEntropyLoss(
            reduction="none", ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, C]
        targets : [B]   — ignore_index entries are masked out
        """
        ce    = self.ce(logits, targets)         # [B]
        pt    = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce

        mask  = (targets != self.ignore_index).float()
        denom = mask.sum().clamp(min=1.0)
        return (focal * mask).sum() / denom


# ===========================================================================
#  KL DISTILLATION LOSS
# ===========================================================================

def kl_distillation_loss(
    student_logits : torch.Tensor,     # [B, C]  raw logits (no softmax)
    teacher_probs  : torch.Tensor,     # [B, C]  already-normalised probs (T=1)
    targets        : torch.Tensor,     # [B]     used only for masking
    ignore_index   : int = -1,
) -> torch.Tensor:
    """
    KL(teacher_probs || student_probs) averaged over valid (non-ignored) samples.

    Teacher CSVs store TTA-averaged probabilities at T=1 — no temperature
    scaling needed. Student log-probs computed at T=1 as well.

    KL(P||Q) = sum P * (log P - log Q)
    """
    mask  = (targets != ignore_index).float()      # [B]
    denom = mask.sum().clamp(min=1.0)

    student_log_probs = F.log_softmax(student_logits, dim=-1)  # [B, C]
    # KL per sample — sum over classes
    kl = F.kl_div(student_log_probs, teacher_probs,
                  reduction="none").sum(dim=-1)    # [B]
    return (kl * mask).sum() / denom


# ===========================================================================
#  COMBINED CLASSIFICATION DISTILLATION LOSS
# ===========================================================================

class ClassDistillationLoss(nn.Module):
    """
    L_cls = alpha * Focal(hard) + (1 - alpha) * KL(teacher_probs)
    Applied to one classification task head.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0,
                 ignore_index: int = -1):
        super().__init__()
        self.alpha        = alpha
        self.ignore_index = ignore_index
        self.focal        = FocalLoss(gamma=gamma, ignore_index=ignore_index)

    def forward(
        self,
        student_logits : torch.Tensor,   # [B, C]
        hard_targets   : torch.Tensor,   # [B]
        teacher_probs  : torch.Tensor,   # [B, C]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, focal_component, kl_component).
        """
        focal_l = self.focal(student_logits, hard_targets)
        kl_l    = kl_distillation_loss(
            student_logits, teacher_probs, hard_targets, self.ignore_index)

        loss = self.alpha * focal_l + (1.0 - self.alpha) * kl_l
        return loss, focal_l, kl_l


# ===========================================================================
#  CIoU LOSS
# ===========================================================================

def ciou_loss(
    pred_boxes : torch.Tensor,   # [N, 4]  cx,cy,w,h normalised
    tgt_boxes  : torch.Tensor,   # [N, 4]  cx,cy,w,h normalised
    eps        : float = 1e-7,
) -> torch.Tensor:
    """
    Complete IoU Loss (Zheng et al. 2020).
    L_CIoU = 1 - IoU + rho²/c² + alpha_v * v
    where:
      rho² = squared Euclidean distance between centres
      c²   = squared diagonal of smallest enclosing box
      v    = aspect ratio consistency
      alpha_v = v / (1 - IoU + v + eps)
    Returns mean over N.
    """
    # Convert cxcywh → xyxy
    px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tx1 = tgt_boxes[:, 0] - tgt_boxes[:, 2] / 2
    ty1 = tgt_boxes[:, 1] - tgt_boxes[:, 3] / 2
    tx2 = tgt_boxes[:, 0] + tgt_boxes[:, 2] / 2
    ty2 = tgt_boxes[:, 1] + tgt_boxes[:, 3] / 2

    # Intersection
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    # Areas
    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    tgt_area  = tgt_boxes[:, 2] * tgt_boxes[:, 3]
    union     = pred_area + tgt_area - inter + eps

    iou = inter / union

    # Enclosing box diagonal²
    cx1 = torch.min(px1, tx1)
    cy1 = torch.min(py1, ty1)
    cx2 = torch.max(px2, tx2)
    cy2 = torch.max(py2, ty2)
    c2  = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + eps

    # Centre distance²
    rho2 = ((pred_boxes[:, 0] - tgt_boxes[:, 0]) ** 2 +
            (pred_boxes[:, 1] - tgt_boxes[:, 1]) ** 2)

    # Aspect ratio
    v = (4.0 / (math.pi ** 2)) * (
        torch.atan(tgt_boxes[:, 2] / (tgt_boxes[:, 3] + eps)) -
        torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + eps))
    ) ** 2
    with torch.no_grad():
        alpha_v = v / (1.0 - iou + v + eps)

    ciou = 1.0 - iou + rho2 / c2 + alpha_v * v
    return ciou.mean()


# ===========================================================================
#  YOLO DETECTION LOSS
# ===========================================================================

class YOLODetectionLoss(nn.Module):
    """
    Decoupled YOLO loss operating on raw_det [B, 5+C, H, W].

    Positive cell assignment: for each target box with centre (cx,cy),
    the responsible cell is floor(cx*W), floor(cy*H).

    Applied twice per forward pass:
      1. Against ground-truth boxes
      2. Against teacher boxes (scaled by beta)

    Losses:
      L_box = lambda_box * CIoU(predicted_box, target_box)
      L_obj = lambda_obj * BCE(predicted_obj, target_obj)
      L_cls = lambda_cls * BCE(predicted_cls[pos], 1.0)
    """

    def __init__(
        self,
        lambda_box  : float = 7.5,
        lambda_obj  : float = 1.0,
        lambda_cls  : float = 0.5,
        beta        : float = 0.5,
    ):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.beta       = beta
        self.bce        = nn.BCEWithLogitsLoss(reduction="mean")

    def _assign_and_compute(
        self,
        raw_det    : torch.Tensor,   # [B, 6, H, W]
        boxes      : torch.Tensor,   # [B, N, 4]  cx,cy,w,h normalised (padded)
        box_mask   : torch.Tensor,   # [B, N]     bool valid
    ) -> torch.Tensor:
        """
        Core YOLO loss for one set of target boxes.
        Returns scalar total loss.
        """
        B, C, H, W = raw_det.shape
        device = raw_det.device

        # Split channels
        pred_box = raw_det[:, :4]              # [B, 4, H, W]
        pred_obj = raw_det[:, 4:5]             # [B, 1, H, W]
        pred_cls = raw_det[:, 5:]              # [B, C-5, H, W]

        # Build targets
        tgt_obj = torch.zeros_like(pred_obj)   # [B, 1, H, W]
        tgt_box_list, pred_box_list = [], []
        cls_pos_preds, cls_pos_tgts  = [], []

        for b in range(B):
            valid = box_mask[b]                # [N] bool
            bxs   = boxes[b][valid]            # [M, 4]
            if bxs.shape[0] == 0:
                continue

            # Cell assignment
            cx_idx = (bxs[:, 0] * W).long().clamp(0, W - 1)
            cy_idx = (bxs[:, 1] * H).long().clamp(0, H - 1)

            tgt_obj[b, 0, cy_idx, cx_idx] = 1.0

            # Box regression (at positive cells)
            p_box = pred_box[b, :, cy_idx, cx_idx].t()   # [M, 4]
            p_box_sig = torch.sigmoid(p_box)              # [M, 4]
            tgt_box_list.append(bxs)
            pred_box_list.append(p_box_sig)

            # Classification (apple = class 0, single class)
            p_cls = pred_cls[b, :, cy_idx, cx_idx].t()   # [M, num_cls]
            cls_pos_preds.append(p_cls)
            cls_pos_tgts.append(torch.ones_like(p_cls))  # all positives = 1

        # Objectness loss (all cells)
        l_obj = self.lambda_obj * self.bce(pred_obj, tgt_obj)

        # Box + cls losses (positive cells only)
        if pred_box_list:
            pred_b = torch.cat(pred_box_list)   # [Total_pos, 4]
            tgt_b  = torch.cat(tgt_box_list)
            l_box  = self.lambda_box * ciou_loss(pred_b, tgt_b)

            pred_c = torch.cat(cls_pos_preds)
            tgt_c  = torch.cat(cls_pos_tgts)
            l_cls  = self.lambda_cls * self.bce(pred_c, tgt_c)
        else:
            l_box = torch.tensor(0.0, device=device)
            l_cls = torch.tensor(0.0, device=device)

        return l_obj + l_box + l_cls

    def forward(
        self,
        raw_det         : torch.Tensor,   # [B, 6, 32, 32]
        gt_boxes        : torch.Tensor,   # [B, N_gt, 4]
        gt_mask         : torch.Tensor,   # [B, N_gt] bool
        teacher_boxes   : torch.Tensor,   # [B, N_tk, 5]  cx,cy,w,h,conf
        teacher_mask    : torch.Tensor,   # [B, N_tk] bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_det_loss, gt_loss, teacher_loss).
        """
        gt_loss = self._assign_and_compute(raw_det, gt_boxes, gt_mask)

        # Teacher boxes: use only the box coordinates (drop conf column)
        tk_boxes = teacher_boxes[:, :, :4]
        tk_loss  = self._assign_and_compute(raw_det, tk_boxes, teacher_mask)

        total = gt_loss + self.beta * tk_loss
        return total, gt_loss, tk_loss


# ===========================================================================
#  HOMOSCEDASTIC UNCERTAINTY WEIGHTER  (Kendall & Gal 2018)
# ===========================================================================

class HomoscedasticWeighter(nn.Module):
    """
    Learns one log-variance log(σ_k²) per task k.
    Weighted loss: exp(-log_var_k) * L_k + 0.5 * log_var_k

    Initialised to zeros → unit weight at training start.
    Optimised jointly with model parameters via the main AdamW step.

    Tasks: ["leaf", "pest", "fruit", "yield"]
    """

    def __init__(self, task_names: list):
        super().__init__()
        self.task_names = task_names
        self.log_vars   = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1))
            for task in task_names
        })

    def forward(
        self, task_losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        task_losses : {task_name: scalar_loss}
        Returns (total_weighted_loss, {task_name: weighted_contribution}).
        """
        total   = torch.tensor(0.0,
                               device=next(iter(task_losses.values())).device)
        details = {}
        for task, loss in task_losses.items():
            lv   = self.log_vars[task]
            prec = torch.exp(-lv)                   # 1 / σ²
            w    = prec * loss + 0.5 * lv
            total = total + w
            details[task] = w.detach()
        return total, details

    def get_weights(self) -> Dict[str, float]:
        """Returns current precision (1/σ²) per task — for logging."""
        return {
            task: float(torch.exp(-lv).item())
            for task, lv in self.log_vars.items()
        }


# ===========================================================================
#  MASTER LOSS MODULE
# ===========================================================================

class ADCLoss(nn.Module):
    """
    Unified loss for all four ADC tasks.

    forward() takes model outputs + batch labels and returns:
      total_loss  : scalar used for .backward()
      loss_dict   : per-task breakdown for logging
    """

    TASKS = ["leaf", "pest", "fruit", "yield"]

    def __init__(self, cfg: ADCConfig):
        super().__init__()
        tc = cfg.train
        dc = cfg.data

        # ── Per-task classification distillation losses ──────────────────────
        self.leaf_loss  = ClassDistillationLoss(
            alpha=tc.alpha, gamma=tc.focal_gamma, ignore_index=dc.ignore_index)
        self.pest_loss  = ClassDistillationLoss(
            alpha=tc.alpha, gamma=tc.focal_gamma, ignore_index=dc.ignore_index)
        self.fruit_loss = ClassDistillationLoss(
            alpha=tc.alpha, gamma=tc.focal_gamma, ignore_index=dc.ignore_index)

        # ── Detection distillation loss ──────────────────────────────────────
        self.det_loss   = YOLODetectionLoss(
            lambda_box = tc.lambda_box,
            lambda_obj = tc.lambda_obj,
            lambda_cls = tc.lambda_cls,
            beta       = tc.beta,
        )

        # ── Uncertainty weighter ─────────────────────────────────────────────
        self.weighter   = HomoscedasticWeighter(self.TASKS)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        model_out   : Dict[str, torch.Tensor],
        batch       : Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        model_out : output dict from ADCStudent.forward()
        batch     : collated batch dict from adc_collate_fn

        Returns (total_loss, log_dict) where log_dict has scalar floats.
        """
        device = model_out["leaf_logits"].device

        # ── 1. Classification losses ─────────────────────────────────────────
        leaf_logits  = model_out["leaf_logits"]      # [B, 5]
        pest_logits  = model_out["pest_logits"]      # [B, 4]
        fruit_logits = model_out["fruit_logits"]     # [M, 3]

        leaf_hard  = batch["leaf_label"].to(device)
        pest_hard  = batch["pest_label"].to(device)
        leaf_soft  = batch["leaf_soft"].to(device)
        pest_soft  = batch["pest_soft"].to(device)

        l_leaf, fl_leaf, kl_leaf = self.leaf_loss(
            leaf_logits, leaf_hard, leaf_soft)
        l_pest, fl_pest, kl_pest = self.pest_loss(
            pest_logits, pest_hard, pest_soft)

        # Fruit loss — logits are [M, 3], labels need to be gathered per ROI
        # Each ROI comes from a specific image; use roi_img_idx to fetch labels
        roi_img_idx  = model_out["box_img_idx"].to(device)    # [M]
        fruit_hard_b = batch["fruit_label"].to(device)        # [B]
        fruit_soft_b = batch["fruit_soft"].to(device)         # [B, 3]

        if fruit_logits.shape[0] > 0 and roi_img_idx.numel() > 0:
            # Gather label/soft-label for each ROI from the originating image
            fruit_hard_roi = fruit_hard_b[roi_img_idx]     # [M]
            fruit_soft_roi = fruit_soft_b[roi_img_idx]     # [M, 3]
            l_fruit, fl_fruit, kl_fruit = self.fruit_loss(
                fruit_logits, fruit_hard_roi, fruit_soft_roi)
        else:
            z = torch.tensor(0.0, device=device)
            l_fruit = fl_fruit = kl_fruit = z

        # ── 2. Detection loss ─────────────────────────────────────────────────
        raw_det       = model_out["raw_det"]          # [B, 6, 32, 32]
        gt_boxes      = batch["gt_boxes"].to(device)  # [B, N, 4]
        gt_mask       = batch["gt_mask"].to(device)   # [B, N]
        teacher_boxes = batch["teacher_boxes"].to(device)   # [B, K, 5]
        teacher_mask  = batch["teacher_mask"].to(device)    # [B, K]

        l_yield, l_gt_det, l_tk_det = self.det_loss(
            raw_det, gt_boxes, gt_mask, teacher_boxes, teacher_mask)

        # ── 3. Homoscedastic uncertainty weighting ────────────────────────────
        task_losses = {
            "leaf"  : l_leaf,
            "pest"  : l_pest,
            "fruit" : l_fruit,
            "yield" : l_yield,
        }
        total_loss, weighted = self.weighter(task_losses)

        # ── 4. Log dict ───────────────────────────────────────────────────────
        weights = self.weighter.get_weights()
        log = {
            # Task losses (unweighted)
            "loss_leaf"     : float(l_leaf),
            "loss_pest"     : float(l_pest),
            "loss_fruit"    : float(l_fruit),
            "loss_yield"    : float(l_yield),
            # Sub-components
            "focal_leaf"    : float(fl_leaf),
            "kl_leaf"       : float(kl_leaf),
            "focal_pest"    : float(fl_pest),
            "kl_pest"       : float(kl_pest),
            "focal_fruit"   : float(fl_fruit),
            "kl_fruit"      : float(kl_fruit),
            "det_gt"        : float(l_gt_det),
            "det_teacher"   : float(l_tk_det),
            # Learned precision weights
            "w_leaf"        : weights["leaf"],
            "w_pest"        : weights["pest"],
            "w_fruit"       : weights["fruit"],
            "w_yield"       : weights["yield"],
            # Total
            "loss_total"    : float(total_loss),
        }

        return total_loss, log