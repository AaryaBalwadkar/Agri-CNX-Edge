"""
student_config.py
=================
Single source of truth for all AgroDistill-Cascade (ADC) hyperparameters.
Every other student file imports from here — never hardcode values elsewhere.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os

# ── Project root (folder containing this file) ───────────────────────────────
BASE_DIR = Path(__file__).resolve().parent


# ===========================================================================
#  PATH CONFIG
# ===========================================================================

@dataclass
class PathConfig:
    # ── Data ─────────────────────────────────────────────────────────────────
    leaf_data_root  : str = str(BASE_DIR / "data" / "leaf")
    pest_data_root  : str = str(BASE_DIR / "data" / "pest")
    fruit_data_root : str = str(BASE_DIR / "data" / "fruit")
    yield_img_dir   : str = str(BASE_DIR / "data" / "yield" / "images" / "train")

    # ── Teacher soft labels ───────────────────────────────────────────────────
    # Classification CSVs: columns = image_path, class_name, true_label_idx,
    #                                prob_0 .. prob_C   (averaged TTA probs, T=1)
    leaf_soft_csv   : str = str(BASE_DIR / "outputs" / "leaf"  / "leaf_soft_labels.csv")
    pest_soft_csv   : str = str(BASE_DIR / "outputs" / "pest"  / "pest_soft_labels.csv")
    fruit_soft_csv  : str = str(BASE_DIR / "outputs" / "fruit" / "fruit_soft_labels.csv")
    # Detection CSV: columns = image_path, json_boxes
    #               json_boxes = [{"xyxy":[x1,y1,x2,y2], "conf":0.87}, ...]
    #               coordinates are NORMALISED [0,1]
    yield_soft_csv  : str = str(BASE_DIR / "outputs" / "yield" / "yield_soft_labels.csv")

    # ── Yield ground-truth labels (YOLO .txt, one file per image) ────────────
    yield_label_dir : str = str(BASE_DIR / "data" / "yield" / "labels" / "train")

    # ── Split files (classification tasks) ───────────────────────────────────
    # Each file: one image path per line
    leaf_train_txt  : str = str(BASE_DIR / "final_dataset" / "splits" / "train.txt")
    leaf_val_txt    : str = str(BASE_DIR / "final_dataset" / "splits" / "val.txt")
    leaf_test_txt   : str = str(BASE_DIR / "final_dataset" / "splits" / "test.txt")

    # ── Backbone weights ──────────────────────────────────────────────────────
    dinov3_repo_dir : str = str(BASE_DIR / "weights" / "dinov3")
    dinov3_weights  : str = str(BASE_DIR / "weights" / "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth")

    # ── Output ────────────────────────────────────────────────────────────────
    student_output_dir : str = str(BASE_DIR / "outputs" / "student")
    checkpoint_dir     : str = str(BASE_DIR / "outputs" / "student" / "checkpoints")
    onnx_path          : str = str(BASE_DIR / "outputs" / "student" / "adc_student.onnx")


# ===========================================================================
#  BACKBONE CONFIG
# ===========================================================================

@dataclass
class BackboneConfig:
    # DINOv3-ConvNeXt-Tiny output channel widths per stage
    # f1=96@128², f2=192@64², f3=384@32², f4=768@16²  (for 512×512 input)
    stage_channels : List[int] = field(default_factory=lambda: [96, 192, 384, 768])

    # Stages to freeze completely (0-indexed)
    frozen_stages  : List[int] = field(default_factory=lambda: [0, 1])

    # Stages to apply LoRA (0-indexed)
    lora_stages    : List[int] = field(default_factory=lambda: [2, 3])

    # LoRA hyperparameters
    lora_rank      : int   = 8
    lora_alpha     : int   = 16     # scaling = alpha / rank = 2.0
    lora_dropout   : float = 0.05

    # LoRA target modules inside ConvNeXt MLP blocks
    # These are the pointwise linear layers — only ones compatible with LoRA
    lora_target_modules : List[str] = field(default_factory=lambda: [
        "mlp.fc1", "mlp.fc2"
    ])


# ===========================================================================
#  MODEL CONFIG
# ===========================================================================

@dataclass
class ModelConfig:
    # ── Input ─────────────────────────────────────────────────────────────────
    image_size     : int = 512

    # ── FPN-Lite Neck ─────────────────────────────────────────────────────────
    neck_channels  : int = 128      # P3 output channels (f3 & f4 both projected to this)

    # ── Detection grid ────────────────────────────────────────────────────────
    # 512 / 32 (stride) = 16×16  ← wait, actually 512/16=32, so grid is 32×32
    grid_size      : int = 32       # 32×32 spatial grid from P3
    num_anchors    : int = 1        # anchor-free: one prediction per cell

    # ── ROI Align ─────────────────────────────────────────────────────────────
    roi_output_size : int  = 7      # each aligned crop = 128×7×7
    roi_spatial_scale: float = 32.0 / 512.0   # P3 is 32×32 for 512 input

    # ── Classification heads (shared MBConv architecture) ─────────────────────
    head_neck_ch   : int = 256      # internal channel width inside each head
    head_dropout   : float = 0.3

    # ── Task-specific number of classes ───────────────────────────────────────
    num_leaf_classes  : int = 4     # Apple_Mosaic, Apple_scab, Black_rot, Alternaria, Healthy
    num_pest_classes  : int = 4     # xylotrechus, aphids, leafhoppers, spider_mite
    num_fruit_classes : int = 3     # Anthracnose, Black Rot, Healthy
    num_yield_classes : int = 1     # apple (single class detection)

    # ── Head → feature map routing ────────────────────────────────────────────
    # LeafHead   consumes f1 (finest-grained, for subtle foliar lesions)
    # PestHead   consumes f2 (mid-scale arthropod morphology)
    # FruitHead  consumes ROI-aligned P3 crops (per detected apple)
    # YOLOHead   consumes P3 (multi-scale detection via FPN)
    leaf_feature_stage  : int = 0   # f1 index in backbone output list
    pest_feature_stage  : int = 1   # f2
    det_feature_stage   : int = 2   # P3 (after FPN neck, not raw f3)

    # ── YOLO detection head ───────────────────────────────────────────────────
    # Output channels per cell: [cx,cy,w,h, obj] + num_yield_classes = 6
    yolo_output_channels : int = 6  # 4 box + 1 obj + 1 cls

    # ── Detection post-processing ─────────────────────────────────────────────
    conf_threshold : float = 0.3    # min objectness×class score to keep box
    nms_iou        : float = 0.45   # NMS IoU threshold
    max_boxes      : int   = 300    # max ground-truth boxes per image (padding)


# ===========================================================================
#  DATA CONFIG
# ===========================================================================

@dataclass
class DataConfig:
    image_size     : int   = 512

    # ImageNet normalisation stats (DINOv3 was pretrained with these)
    mean : List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std  : List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # ignore_index sentinel — images without a label for a given task
    # loss functions skip samples where label == ignore_index
    ignore_index : int = -1

    # Bbox padding (pad ground-truth boxes to fixed length for batching)
    max_boxes    : int = 300

    # Min bbox visibility after augmentation (Albumentations BboxParams)
    min_bbox_visibility : float = 0.1

    # DataLoader
    num_workers  : int  = 4
    pin_memory   : bool = True

    # Train/val/test split ratios (for tasks without pre-defined splits)
    val_split    : float = 0.15
    test_split   : float = 0.15
    seed         : int   = 42


# ===========================================================================
#  TRAINING CONFIG
# ===========================================================================

@dataclass
class TrainConfig:
    # ── Duration ──────────────────────────────────────────────────────────────
    epochs         : int = 200

    # ── Two-stage training ────────────────────────────────────────────────────
    # Phase 1: freeze LoRA + backbone, train heads + neck only
    freeze_backbone_epochs : int = 5   # warm up heads before LoRA activates

    # ── Batch ─────────────────────────────────────────────────────────────────
    batch_size     : int = 8
    accum_steps    : int = 4           # effective batch = 8 × 4 = 32

    # ── Learning rates ────────────────────────────────────────────────────────
    lr_lora        : float = 1e-5      # LoRA adapter parameters in backbone
    lr_heads       : float = 1e-4      # all task heads + FPN neck
    weight_decay   : float = 1e-4

    # ── Scheduler: Cosine Annealing ───────────────────────────────────────────
    eta_min        : float = 1e-7

    # ── Gradient clipping ─────────────────────────────────────────────────────
    max_grad_norm  : float = 1.0

    # ── Distillation hyperparameters ──────────────────────────────────────────
    # Classification: L = alpha*Focal(hard) + (1-alpha)*KL(soft, T=1)
    # Note: teacher CSVs already store TTA-averaged probabilities at T=1
    # so no temperature scaling is applied in the student loss
    alpha          : float = 0.5       # hard vs soft label blend
    focal_gamma    : float = 2.0

    # Detection distillation: adds teacher box alignment loss
    # L_det_total = L_gt + beta * L_teacher
    beta           : float = 0.5

    # Detection loss weights (applied to both GT and teacher branches)
    lambda_box     : float = 7.5
    lambda_obj     : float = 1.0
    lambda_cls     : float = 0.5

    # ── Checkpointing ─────────────────────────────────────────────────────────
    save_best_only : bool  = True
    eval_every     : int   = 1         # validate every N epochs

    # ── Early stopping ────────────────────────────────────────────────────────
    patience       : int   = 20
    min_delta      : float = 1e-4

    # ── Mixed precision ───────────────────────────────────────────────────────
    # Disabled — CIoU and KL divergence unstable in FP16
    use_amp        : bool  = False


# ===========================================================================
#  TASK CLASS MAPS  (index → class name, MUST match teacher CSV column order)
# ===========================================================================

LEAF_CLASS_MAP = {
    "Apple_Mosaic"      : 0,
    "Apple___Black_rot" : 1,
    "Alternaria"        : 2,
    "Healthy"           : 3,
}

PEST_CLASS_MAP = {
    "xylotrechus" : 0,
    "aphids"      : 1,
    "leafhoppers" : 2,
    "spider_mite" : 3,
}

FRUIT_CLASS_MAP = {
    "Anthracnose" : 0,
    "Black Rot"   : 1,
    "Healthy"     : 2,
}

YIELD_CLASS_MAP = {
    "apple" : 0,
}

# Reverse maps (index → name) — used for inference output
IDX_TO_LEAF  = {v: k for k, v in LEAF_CLASS_MAP.items()}
IDX_TO_PEST  = {v: k for k, v in PEST_CLASS_MAP.items()}
IDX_TO_FRUIT = {v: k for k, v in FRUIT_CLASS_MAP.items()}


# ===========================================================================
#  MASTER CONFIG  (single object imported by all other files)
# ===========================================================================

@dataclass
class ADCConfig:
    paths    : PathConfig    = field(default_factory=PathConfig)
    backbone : BackboneConfig = field(default_factory=BackboneConfig)
    model    : ModelConfig   = field(default_factory=ModelConfig)
    data     : DataConfig    = field(default_factory=DataConfig)
    train    : TrainConfig   = field(default_factory=TrainConfig)

    def make_dirs(self):
        """Create all output directories."""
        for d in [self.paths.student_output_dir,
                  self.paths.checkpoint_dir]:
            os.makedirs(d, exist_ok=True)

    def validate(self):
        """Sanity-check critical paths exist before training starts."""
        errors = []
        critical = [
            ("DINOv3 weights",   self.paths.dinov3_weights),
            ("Leaf soft labels", self.paths.leaf_soft_csv),
            ("Pest soft labels", self.paths.pest_soft_csv),
            ("Fruit soft labels",self.paths.fruit_soft_csv),
            ("Yield soft labels",self.paths.yield_soft_csv),
        ]
        for name, path in critical:
            if not os.path.exists(path):
                errors.append(f"  MISSING {name}: {path}")
        if errors:
            raise FileNotFoundError(
                "ADCConfig.validate() failed:\n" + "\n".join(errors)
            )
        return True


# ===========================================================================
#  CONVENIENCE: build default config
# ===========================================================================

def get_config() -> ADCConfig:
    """Returns a fully initialised ADCConfig with all defaults."""
    cfg = ADCConfig()
    cfg.make_dirs()
    return cfg
