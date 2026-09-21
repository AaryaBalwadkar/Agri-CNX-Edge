"""
student_model.py
================
AgroDistill-Cascade (ADC) unified multi-task student model.

Forward pass (single image or batch):
  Input  : [B, 3, 512, 512]
  Output : {
    "leaf_logits"  : [B, 5]          image-level leaf disease
    "pest_logits"  : [B, 4]          image-level pest identification
    "raw_det"      : [B, 6, 32, 32]  raw YOLO grid (sigmoid NOT applied)
    "fruit_logits" : [M, 3]          per-apple disease (M = total detections)
    "boxes"        : [M, 4]          detected apple boxes [cx,cy,w,h] normalised
    "box_img_idx"  : [M]             which image in batch each box belongs to
    "apple_count"  : [B]             per-image apple count after NMS
  }

ROI Align bridges detection → fruit disease classification.
At training time all M boxes (from ground-truth) go through FruitHead.
At inference time M boxes come from DetectionHead + NMS.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops          # roi_align + nms

from student_config import ADCConfig
from student_backbone import ADCBackbone


# ===========================================================================
#  BUILDING BLOCKS
# ===========================================================================

class DepthwiseSepConv(nn.Module):
    """Depthwise separable conv: depthwise 3×3 → pointwise 1×1 → BN → SiLU."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                      padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, ratio: float = 0.25):
        super().__init__()
        mid = max(1, int(channels * ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck: expand → depthwise → SE → project.
    Residual only when in_ch == out_ch.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 expand: int = 4, se_ratio: float = 0.25,
                 dropout: float = 0.0):
        super().__init__()
        mid = in_ch * expand
        self.expand   = (nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
            if expand != 1 else nn.Identity())
        self.dw       = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
        self.se       = SEBlock(mid, se_ratio)
        self.project  = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.drop     = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.use_skip = (in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.expand(x)
        z = self.dw(z)
        z = self.se(z)
        z = self.project(z)
        z = self.drop(z)
        return z + x if self.use_skip else z


# ===========================================================================
#  CLASSIFICATION HEAD  (shared by Leaf, Pest, and Fruit tasks)
# ===========================================================================

class ClassificationHead(nn.Module):
    """
    f_in [B, in_ch, H, W]  →  logits [B, num_classes]   (image-level)
    OR
    crops [M, in_ch, 7, 7] →  logits [M, num_classes]   (per-ROI)

    Architecture:
      1×1 conv (in_ch → neck_ch)  +  BN  +  SiLU
      MBConvBlock (neck_ch → neck_ch)
      AdaptiveAvgPool2d(1)
      Dropout → Linear(neck_ch → num_classes)
    """

    def __init__(self, in_ch: int, num_classes: int,
                 neck_ch: int = 256, dropout: float = 0.3):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, neck_ch, 1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.SiLU(inplace=True),
        )
        self.mbconv  = MBConvBlock(neck_ch, neck_ch, expand=4, se_ratio=0.25)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.drop    = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(neck_ch, num_classes)

        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.mbconv(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


# ===========================================================================
#  FPN-LITE NECK
# ===========================================================================

class FPNLiteNeck(nn.Module):
    """
    Fuses f3 [B,384,32,32] and f4 [B,768,16,16] into P3 [B,128,32,32].

    Top-down:
      lat4 : 768→128  (1×1 proj)
      up4  : upsample lat4 to 32×32
      lat3 : 384→128  (1×1 proj)
      fuse : lat3 + up4  (element-wise sum)
      P3   : DepthwiseSepConv(128,128) to refine
    """

    def __init__(self, f3_ch: int = 384, f4_ch: int = 768,
                 neck_ch: int = 128):
        super().__init__()
        self.lat4   = nn.Sequential(
            nn.Conv2d(f4_ch, neck_ch, 1, bias=False),
            nn.BatchNorm2d(neck_ch), nn.SiLU(inplace=True))
        self.lat3   = nn.Sequential(
            nn.Conv2d(f3_ch, neck_ch, 1, bias=False),
            nn.BatchNorm2d(neck_ch), nn.SiLU(inplace=True))
        self.refine = DepthwiseSepConv(neck_ch, neck_ch)

    def forward(self, f3: torch.Tensor,
                f4: torch.Tensor) -> torch.Tensor:
        p4_up = F.interpolate(self.lat4(f4), size=f3.shape[-2:],
                              mode="nearest")
        p3    = self.lat3(f3) + p4_up
        return self.refine(p3)        # [B, 128, 32, 32]


# ===========================================================================
#  DECOUPLED YOLO DETECTION HEAD
# ===========================================================================

class DecoupledDetHead(nn.Module):
    """
    Anchor-free, single-class, decoupled YOLO detection head.
    Operates on P3 [B, neck_ch, 32, 32].

    Three separate branches (decoupled):
      reg  → [B, 4, 32, 32]   cx, cy, w, h  (sigmoid applied at loss time)
      obj  → [B, 1, 32, 32]   objectness score
      cls  → [B, C, 32, 32]   class scores

    Concatenated output: [B, 4+1+C, 32, 32]  stored as "raw_det".
    Sigmoid is NOT applied here — applied in loss and post-processing.
    """

    def __init__(self, neck_ch: int = 128, num_classes: int = 1):
        super().__init__()
        self.reg_branch = nn.Sequential(
            DepthwiseSepConv(neck_ch, neck_ch),
            nn.Conv2d(neck_ch, 4, 1),
        )
        self.obj_branch = nn.Sequential(
            DepthwiseSepConv(neck_ch, neck_ch),
            nn.Conv2d(neck_ch, 1, 1),
        )
        self.cls_branch = nn.Sequential(
            DepthwiseSepConv(neck_ch, neck_ch),
            nn.Conv2d(neck_ch, num_classes, 1),
        )
        self._init_weights()

    def _init_weights(self):
        # Bias init: prior prob=0.01 → no early objectness explosion
        prior_prob  = 0.01
        bias_value  = -math.log((1 - prior_prob) / prior_prob)
        for branch in [self.obj_branch, self.cls_branch]:
            last = list(branch.children())[-1]
            if isinstance(last, nn.Conv2d) and last.bias is not None:
                nn.init.constant_(last.bias, bias_value)
        # Reg bias = 0
        last_reg = list(self.reg_branch.children())[-1]
        if isinstance(last_reg, nn.Conv2d) and last_reg.bias is not None:
            nn.init.zeros_(last_reg.bias)

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        reg = self.reg_branch(p3)   # [B, 4, H, W]
        obj = self.obj_branch(p3)   # [B, 1, H, W]
        cls = self.cls_branch(p3)   # [B, C, H, W]
        return torch.cat([reg, obj, cls], dim=1)   # [B, 5+C, H, W]


# ===========================================================================
#  POST-PROCESSING (decode raw_det → boxes for ROI Align / inference)
# ===========================================================================

@torch.no_grad()
def decode_detections(
    raw_det      : torch.Tensor,   # [B, 6, 32, 32]
    conf_thresh  : float = 0.3,
    nms_iou      : float = 0.45,
    img_size     : int   = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decodes raw YOLO grid into apple bounding boxes.

    Returns:
      boxes      [M, 4]   normalised [cx,cy,w,h]  (M = total over batch)
      scores     [M]      confidence (obj × cls)
      img_idx    [M]      which batch image each box belongs to
    """
    B, C, H, W = raw_det.shape
    device = raw_det.device

    # Flatten grid → [B, H*W, C]
    flat  = raw_det.permute(0, 2, 3, 1).reshape(B, H * W, C)

    boxes_all  : List[torch.Tensor] = []
    scores_all : List[torch.Tensor] = []
    idx_all    : List[torch.Tensor] = []

    for b in range(B):
        pred     = flat[b]                        # [H*W, 6]
        box_pred = torch.sigmoid(pred[:, :4])     # cx,cy,w,h in [0,1]
        obj      = torch.sigmoid(pred[:, 4])
        cls      = torch.sigmoid(pred[:, 5])
        scores   = obj * cls                      # [H*W]

        mask = scores > conf_thresh
        if mask.sum() == 0:
            continue

        boxes_f  = box_pred[mask]    # [K, 4]  cx,cy,w,h
        scores_f = scores[mask]      # [K]

        # Convert to xyxy for NMS (torchvision expects xyxy)
        bxyxy = torch.stack([
            boxes_f[:, 0] - boxes_f[:, 2] / 2,
            boxes_f[:, 1] - boxes_f[:, 3] / 2,
            boxes_f[:, 0] + boxes_f[:, 2] / 2,
            boxes_f[:, 1] + boxes_f[:, 3] / 2,
        ], dim=1)

        keep = ops.nms(bxyxy, scores_f, nms_iou)
        boxes_all.append(boxes_f[keep])
        scores_all.append(scores_f[keep])
        idx_all.append(torch.full((len(keep),), b,
                                   dtype=torch.long, device=device))

    if not boxes_all:
        empty = torch.zeros(0, 4, device=device)
        return empty, torch.zeros(0, device=device), \
               torch.zeros(0, dtype=torch.long, device=device)

    return (torch.cat(boxes_all),
            torch.cat(scores_all),
            torch.cat(idx_all))


# ===========================================================================
#  ROI ALIGN WRAPPER
# ===========================================================================

def roi_align_crops(
    p3          : torch.Tensor,       # [B, neck_ch, H, W]
    boxes_cxcywh: torch.Tensor,       # [M, 4]  normalised cx,cy,w,h
    img_idx     : torch.Tensor,       # [M]     batch index per box
    output_size : int   = 7,
    spatial_scale: float = 32.0 / 512.0,
) -> torch.Tensor:
    """
    Extracts [output_size × output_size] feature crops from P3
    for each detected apple box using bilinear ROI Align.

    Returns [M, neck_ch, output_size, output_size].

    torchvision.ops.roi_align expects:
      rois = [K, 5] where col 0 is the batch index (float)
      boxes in XYXY format in the INPUT image coordinate space
      (then scaled internally by spatial_scale to P3 coordinates)
    """
    if boxes_cxcywh.shape[0] == 0:
        nc = p3.shape[1]
        return torch.zeros(0, nc, output_size, output_size, device=p3.device)

    # Convert cx,cy,w,h → x1,y1,x2,y2  (still normalised [0,1])
    cx, cy, w, h = boxes_cxcywh.unbind(dim=1)
    x1 = (cx - w / 2).clamp(0, 1)
    y1 = (cy - h / 2).clamp(0, 1)
    x2 = (cx + w / 2).clamp(0, 1)
    y2 = (cy + h / 2).clamp(0, 1)

    # Scale to absolute pixel coords in the 512×512 input space
    scale = 1.0 / spatial_scale          # = 512/32 = 16  (maps [0,1]→pixel)
    img_h = img_w = round(1.0 / spatial_scale) * p3.shape[-1]  # ≈ 512

    rois = torch.stack([
        img_idx.float(),
        x1 * img_w,
        y1 * img_h,
        x2 * img_w,
        y2 * img_h,
    ], dim=1)    # [M, 5]

    return ops.roi_align(
        p3,
        rois,
        output_size   = (output_size, output_size),
        spatial_scale = spatial_scale,
        sampling_ratio= 2,
        aligned       = True,
    )   # [M, neck_ch, output_size, output_size]


# ===========================================================================
#  FULL ADC STUDENT MODEL
# ===========================================================================

class ADCStudent(nn.Module):
    """
    AgroDistill-Cascade (ADC) unified student model.

    Single forward pass produces predictions for four tasks:
      1. Leaf disease classification   (image-level, 5 classes)
      2. Pest identification           (image-level, 4 classes)
      3. Apple detection               (per-box, 1 class + coordinates)
      4. Fruit disease classification  (per-apple via ROI Align, 3 classes)

    Training mode  (gt_boxes provided): ROI Align uses ground-truth boxes
                   so FruitHead always receives valid crops.
    Inference mode (gt_boxes=None):     ROI Align uses boxes from DetHead+NMS.
    """

    def __init__(self, cfg: ADCConfig):
        super().__init__()
        self.cfg = cfg
        mc       = cfg.model
        bc       = cfg.backbone

        # ── 1. Backbone (DINOv3 + LoRA) ──────────────────────────────────────
        self.backbone = ADCBackbone(cfg)

        # ── 2. FPN-Lite Neck ─────────────────────────────────────────────────
        self.neck = FPNLiteNeck(
            f3_ch   = bc.stage_channels[2],   # 384
            f4_ch   = bc.stage_channels[3],   # 768
            neck_ch = mc.neck_channels,        # 128
        )

        # ── 3. Task heads ─────────────────────────────────────────────────────
        # LeafHead: consumes f1 (96ch) — finest spatial detail for lesion texture
        self.leaf_head = ClassificationHead(
            in_ch       = bc.stage_channels[0],    # 96
            num_classes = mc.num_leaf_classes,      # 5
            neck_ch     = mc.head_neck_ch,
            dropout     = mc.head_dropout,
        )

        # PestHead: consumes f2 (192ch) — mid-scale arthropod morphology
        self.pest_head = ClassificationHead(
            in_ch       = bc.stage_channels[1],    # 192
            num_classes = mc.num_pest_classes,      # 4
            neck_ch     = mc.head_neck_ch,
            dropout     = mc.head_dropout,
        )

        # DetectionHead: consumes P3 (128ch, 32×32)
        self.det_head = DecoupledDetHead(
            neck_ch     = mc.neck_channels,        # 128
            num_classes = mc.num_yield_classes,    # 1 (apple)
        )

        # FruitDiseaseHead: consumes ROI-Aligned P3 crops (128ch, 7×7)
        self.fruit_head = ClassificationHead(
            in_ch       = mc.neck_channels,        # 128
            num_classes = mc.num_fruit_classes,    # 3
            neck_ch     = mc.head_neck_ch,
            dropout     = mc.head_dropout,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x        : torch.Tensor,
        gt_boxes : Optional[torch.Tensor] = None,
        gt_box_img_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        x              : [B, 3, 512, 512]
        gt_boxes       : [N_gt, 4]  cx,cy,w,h normalised — provided at training
        gt_box_img_idx : [N_gt]     batch index per gt box

        At training: pass gt_boxes so FruitHead always sees valid crops.
        At inference: leave None — boxes come from DetHead + NMS.

        Returns dict with all task outputs (see module docstring).
        """
        mc = self.cfg.model

        # ── Backbone ──────────────────────────────────────────────────────────
        feats = self.backbone(x)           # {"f1","f2","f3","f4"}
        f1, f2, f3, f4 = (feats["f1"], feats["f2"],
                           feats["f3"], feats["f4"])

        # ── Neck → P3 ─────────────────────────────────────────────────────────
        p3 = self.neck(f3, f4)             # [B, 128, 32, 32]

        # ── Leaf + Pest heads (global) ────────────────────────────────────────
        leaf_logits = self.leaf_head(f1)   # [B, 5]
        pest_logits = self.pest_head(f2)   # [B, 4]

        # ── Detection head ────────────────────────────────────────────────────
        raw_det = self.det_head(p3)        # [B, 6, 32, 32]  raw (no sigmoid)

        # ── ROI Align + FruitHead ─────────────────────────────────────────────
        if gt_boxes is not None and gt_box_img_idx is not None:
            # Training mode: use ground-truth boxes for guaranteed valid crops
            boxes_for_roi   = gt_boxes
            img_idx_for_roi = gt_box_img_idx
        else:
            # Inference mode: decode DetHead + NMS
            boxes_for_roi, _, img_idx_for_roi = decode_detections(
                raw_det,
                conf_thresh = mc.conf_threshold,
                nms_iou     = mc.nms_iou,
                img_size    = mc.image_size,
            )

        # ROI Align → per-apple feature crops
        roi_crops = roi_align_crops(
            p3,
            boxes_for_roi,
            img_idx_for_roi,
            output_size    = mc.roi_output_size,   # 7
            spatial_scale  = mc.roi_spatial_scale, # 32/512
        )   # [M, 128, 7, 7]

        # Per-apple fruit disease classification
        if roi_crops.shape[0] > 0:
            fruit_logits = self.fruit_head(roi_crops)   # [M, 3]
        else:
            fruit_logits = torch.zeros(
                0, mc.num_fruit_classes, device=x.device)

        # ── Apple count (per image in batch) ─────────────────────────────────
        B = x.shape[0]
        apple_count = torch.zeros(B, dtype=torch.long, device=x.device)
        if img_idx_for_roi.numel() > 0:
            for b in range(B):
                apple_count[b] = (img_idx_for_roi == b).sum()

        return {
            "leaf_logits"  : leaf_logits,       # [B, 5]
            "pest_logits"  : pest_logits,        # [B, 4]
            "raw_det"      : raw_det,            # [B, 6, 32, 32]
            "fruit_logits" : fruit_logits,       # [M, 3]
            "boxes"        : boxes_for_roi,      # [M, 4]
            "box_img_idx"  : img_idx_for_roi,    # [M]
            "apple_count"  : apple_count,        # [B]
        }

    # ─────────────────────────────────────────────────────────────────────────
    def get_parameter_groups(self) -> List[Dict]:
        """
        Returns two parameter groups for the AdamW optimiser:
          group 0 — LoRA adapters   : lr=lora_lr  (very small — fine-tune backbone)
          group 1 — heads + neck    : lr=heads_lr (higher — train from scratch)
        """
        tc = self.cfg.train

        lora_params  = self.backbone.get_lora_parameters()
        lora_ids     = {id(p) for p in lora_params}

        head_params  = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in lora_ids
        ]

        return [
            {"params": lora_params, "lr": tc.lr_lora,
             "weight_decay": tc.weight_decay, "name": "lora_backbone"},
            {"params": head_params, "lr": tc.lr_heads,
             "weight_decay": tc.weight_decay, "name": "heads_and_neck"},
        ]

    # ─────────────────────────────────────────────────────────────────────────
    def freeze_heads_for_phase1(self):
        """
        Phase 1 (first few epochs): freeze LoRA, train heads + neck only.
        Call unfreeze_lora() to enter Phase 2.
        """
        for p in self.backbone.get_lora_parameters():
            p.requires_grad_(False)
        print("[ADCStudent] Phase 1: LoRA frozen. Training heads + neck only.")

    def unfreeze_lora(self):
        """Phase 2: activate LoRA gradients."""
        for p in self.backbone.get_lora_parameters():
            p.requires_grad_(True)
        print("[ADCStudent] Phase 2: LoRA unfrozen. Full fine-tuning active.")

    # ─────────────────────────────────────────────────────────────────────────
    def summary(self) -> Dict:
        bb_summary = self.backbone.summary()
        head_params = sum(
            p.numel() for n, p in self.named_parameters()
            if "backbone" not in n
        )
        return {
            "backbone_total_M"   : bb_summary["total_M"],
            "backbone_lora_K"    : bb_summary["lora_K"],
            "backbone_frozen_M"  : bb_summary["frozen_M"],
            "heads_neck_total_K" : head_params / 1e3,
            "lora_efficiency"    : bb_summary["lora_pct"],
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "=" * 60)
        print("  ADCStudent — Parameter Summary")
        print("=" * 60)
        print(f"  Backbone total  : {s['backbone_total_M']:.2f} M")
        print(f"  Backbone frozen : {s['backbone_frozen_M']:.2f} M")
        print(f"  LoRA adapters   : {s['backbone_lora_K']:.1f} K  "
              f"({s['lora_efficiency']:.3f}% of backbone)")
        print(f"  Heads + Neck    : {s['heads_neck_total_K']:.1f} K")
        print("=" * 60 + "\n")