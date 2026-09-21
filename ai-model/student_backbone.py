"""
student_backbone.py
===================
DINOv3-ConvNeXt-Tiny backbone with custom LoRA adapters.
  Stages 0,1 : fully frozen (no gradients)
  Stages 2,3 : mlp.fc1 + mlp.fc2 replaced with LoRALinear (rank=8)
  Forward    : returns {"f1","f2","f3","f4"} each [B,C,H,W]

Custom LoRA — no peft dependency, works directly with torch.hub models.
"""

import math
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from student_config import ADCConfig, BackboneConfig


# ===========================================================================
#  LORA LINEAR LAYER
# ===========================================================================

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a frozen weight plus low-rank adapters.

    y = x @ W.T + (dropout(x) @ A.T @ B.T) * (alpha/rank)

    Init:  A ~ Kaiming,  B = 0  → adapter output is 0 at start of training.
    Only lora_A and lora_B have requires_grad=True.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8,
                 alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.linear  = linear
        self.rank    = rank
        self.scaling = alpha / rank
        d_in, d_out  = linear.in_features, linear.out_features

        self.lora_A    = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B    = nn.Parameter(torch.zeros(d_out, rank))
        self.lora_drop = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze original linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.lora_drop(x) @ self.lora_A.t() @ self.lora_B.t()
        return base + lora * self.scaling

    def extra_repr(self) -> str:
        return (f"in={self.linear.in_features}, out={self.linear.out_features}, "
                f"rank={self.rank}, scale={self.scaling:.2f}")


# ===========================================================================
#  HELPER: introspect backbone to find LoRA target modules
# ===========================================================================

def _find_linear_modules(backbone: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Returns [(full_name, module)] for every nn.Linear in the backbone."""
    results = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Linear):
            results.append((name, module))
    return results


def _is_lora_stage(name: str, lora_stages: List[int]) -> bool:
    """True if module name belongs to one of the LoRA stages."""
    return any(re.search(rf'\bstages?[_\.]?{s}\b|stage{s}', name, re.I)
               or f'stages.{s}.' in name
               or f'stage{s}.' in name
               for s in lora_stages)


def _is_frozen_stage(name: str, frozen_stages: List[int]) -> bool:
    return any(f'stages.{s}.' in name or f'stage{s}.' in name
               for s in frozen_stages)


def _is_lora_target_module(name: str, targets: List[str]) -> bool:
    return any(name.endswith(t) for t in targets)


def _match_modules_flexibly(
    backbone    : nn.Module,
    lora_stages : List[int],
    targets     : List[str],
) -> List[Tuple[str, nn.Linear]]:
    """
    Finds Linear layers in lora_stages matching any of targets.
    Falls back to keyword search if exact stage pattern not found.
    """
    candidates = _find_linear_modules(backbone)
    matched = [
        (name, mod) for name, mod in candidates
        if _is_lora_stage(name, lora_stages) and _is_lora_target_module(name, targets)
    ]

    if not matched:
        # Diagnostic: print all linear names to help user debug
        print("[WARN] No LoRA targets found with standard ConvNeXt naming.")
        print("       All nn.Linear modules in backbone:")
        for name, _ in candidates[:30]:
            print(f"         {name}")
        if len(candidates) > 30:
            print(f"         ... and {len(candidates)-30} more")
        print("       Update lora_stages / lora_target_modules in student_config.py")

    return matched


# ===========================================================================
#  APPLY LoRA
# ===========================================================================

def apply_lora(backbone: nn.Module, cfg: BackboneConfig) -> int:
    """
    Replaces matching nn.Linear layers with LoRALinear in-place.
    Returns the number of layers replaced.
    """
    targets = _match_modules_flexibly(backbone, cfg.lora_stages,
                                       cfg.lora_target_modules)
    n = 0
    for full_name, _ in targets:
        parts     = full_name.split(".")
        attr      = parts[-1]
        parent    = backbone
        for p in parts[:-1]:
            parent = getattr(parent, p)

        original = getattr(parent, attr)
        if not isinstance(original, nn.Linear):
            continue

        lora_layer = LoRALinear(
            original,
            rank    = cfg.lora_rank,
            alpha   = cfg.lora_alpha,
            dropout = cfg.lora_dropout,
        )
        setattr(parent, attr, lora_layer)
        n += 1
        print(f"  [LoRA] {full_name}  "
              f"({original.in_features}→{original.out_features})")

    return n


# ===========================================================================
#  FREEZE NON-LoRA BACKBONE PARAMETERS
# ===========================================================================

def freeze_non_lora(backbone: nn.Module, lora_stages: List[int]) -> int:
    """
    Freezes all backbone parameters that are NOT LoRA adapter weights.
    LoRA A/B matrices are left trainable (set by LoRALinear.__init__).
    """
    n_frozen = 0
    for name, param in backbone.named_parameters():
        # Skip LoRA adapter params — they manage requires_grad themselves
        if "lora_A" in name or "lora_B" in name:
            continue
        param.requires_grad_(False)
        n_frozen += 1
    return n_frozen


# ===========================================================================
#  FEATURE SHAPE NORMALISATION
# ===========================================================================

CONVNEXT_TINY_CHANNELS = [96, 192, 384, 768]  # widths at stages 1-4


def _to_bchw(feat: torch.Tensor, expected_c: int) -> torch.Tensor:
    """
    Accepts [B,C,H,W] or [B,H,W,C] or [B,N,C] and returns [B,C,H,W].
    """
    if feat.ndim == 4:
        if feat.shape[1] == expected_c:          # already [B,C,H,W]
            return feat.contiguous()
        elif feat.shape[-1] == expected_c:        # [B,H,W,C] channels-last
            return feat.permute(0, 3, 1, 2).contiguous()
    elif feat.ndim == 3:                          # [B,N,C] token sequence
        B, N, C = feat.shape
        H = W = int(math.isqrt(N))
        assert H * W == N, f"Token sequence {N} is not a perfect square."
        feat = feat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return feat
    raise ValueError(
        f"Cannot convert feature of shape {feat.shape} "
        f"to [B,{expected_c},H,W]."
    )


# ===========================================================================
#  ADC BACKBONE MODULE
# ===========================================================================

class ADCBackbone(nn.Module):
    """
    DINOv3-ConvNeXt-Tiny with LoRA on stages 2 & 3.

    forward(x: [B,3,512,512]) → {
        "f1": [B,  96, 128, 128],
        "f2": [B, 192,  64,  64],
        "f3": [B, 384,  32,  32],
        "f4": [B, 768,  16,  16],
    }
    """

    def __init__(self, cfg: ADCConfig):
        super().__init__()
        self.cfg  = cfg
        self.bcfg = cfg.backbone

        # ── 1. Load DINOv3 backbone ──────────────────────────────────────────
        print("[ADCBackbone] Loading DINOv3-ConvNeXt-Tiny...")
        print(f"  Repo    : {cfg.paths.dinov3_repo_dir}")
        print(f"  Weights : {cfg.paths.dinov3_weights}")

        import re
        import timm as _timm

        self.backbone = _timm.create_model(
            'convnext_tiny', pretrained=False,
            features_only=True, out_indices=(0, 1, 2, 3)
        )

        ckpt = torch.load(cfg.paths.dinov3_weights, map_location='cpu', weights_only=False)
        raw  = ckpt if not isinstance(ckpt, dict) else (
            ckpt.get('model') or ckpt.get('state_dict') or ckpt
        )

        model_keys = set(self.backbone.state_dict().keys())
        remapped_raw = {}

        for old_key, value in raw.items():
            k = old_key

            # 1. Strip leading prefixes (e.g., teacher.backbone., model.backbone.)
            match = re.search(r'\b(downsample_layers|stages|stem|norm)\.', k)
            if match:
                k = k[match.start():]

            # 2. Try underscore format (timm features_only format)
            k_under = re.sub(r'^downsample_layers\.0\.(\d+)(.*)', r'stem_\1\2', k)
            
            def _ds_under(m):
                return f'stages_{m.group(1)}.downsample.{m.group(2)}{m.group(3)}'
            k_under = re.sub(r'^downsample_layers\.([1-9]\d*)\.(\d+)(.*)', _ds_under, k_under)

            def _blk_under(m):
                s = m.group(3)
                s = re.sub(r'^dwconv\.', 'conv_dw.', s)
                s = re.sub(r'^pwconv1\.', 'mlp.fc1.', s)
                s = re.sub(r'^pwconv2\.', 'mlp.fc2.', s)
                return f'stages_{m.group(1)}.blocks.{m.group(2)}.{s}'
            k_under = re.sub(r'^stages\.(\d+)\.(\d+)\.(.*)', _blk_under, k_under)

            # 3. Try standard dot format as fallback
            k_dot = re.sub(r'^downsample_layers\.0\.(\d+)(.*)', r'stem.\1\2', k)
            def _ds_dot(m):
                return f'stages.{int(m.group(1))}.downsample.{m.group(2)}{m.group(3)}'
            k_dot = re.sub(r'^downsample_layers\.([1-9]\d*)\.(\d+)(.*)', _ds_dot, k_dot)
            def _blk_dot(m):
                s = m.group(3)
                s = re.sub(r'^dwconv\.', 'conv_dw.', s)
                s = re.sub(r'^pwconv1\.', 'mlp.fc1.', s)
                s = re.sub(r'^pwconv2\.', 'mlp.fc2.', s)
                return f'stages.{m.group(1)}.blocks.{m.group(2)}.{s}'
            k_dot = re.sub(r'^stages\.(\d+)\.(\d+)\.(.*)', _blk_dot, k_dot)

            # Match against backbone keys
            if k_under in model_keys:
                remapped_raw[k_under] = value
            elif k_dot in model_keys:
                remapped_raw[k_dot] = value
            elif k in model_keys:
                remapped_raw[k] = value

        # 3. Load state dict into backbone
        missing, unexpected = self.backbone.load_state_dict(remapped_raw, strict=False)
        print(f"[ADCBackbone] DINOv3 weights loaded: {len(remapped_raw)}/{len(model_keys)} backbone keys matched.")

        if len(missing) > 0:
            print(f"[WARN] {len(missing)} missing keys in backbone (e.g. {missing[:3]})")

        self.backbone.eval()

        # ── 2. Apply LoRA (stages 2,3 → mlp.fc1, mlp.fc2) ──────────────────
        print(f"[ADCBackbone] Applying LoRA "
              f"(rank={self.bcfg.lora_rank}, alpha={self.bcfg.lora_alpha}) "
              f"to stages {self.bcfg.lora_stages}...")
        n_lora = apply_lora(self.backbone, self.bcfg)
        if n_lora == 0:
            print("[WARN] LoRA was not applied to any layer. "
                  "Check student_config.py: lora_stages and lora_target_modules.")

        # ── 3. Freeze everything except LoRA adapters ────────────────────────
        n_frozen = freeze_non_lora(self.backbone, self.bcfg.lora_stages)
        print(f"[ADCBackbone] Frozen {n_frozen} parameter tensors. "
              f"LoRA adapters remain trainable.")

        # ── 4. Print summary ─────────────────────────────────────────────────
        s = self.summary()
        print(f"[ADCBackbone] Params — Total: {s['total_M']:.2f}M  "
              f"| Trainable(LoRA): {s['lora_K']:.1f}K "
              f"| Frozen: {s['frozen_M']:.2f}M")
        print(f"[ADCBackbone] LoRA efficiency: {s['lora_pct']:.3f}% of params are LoRA")
        
    # ─────────────────────────────────────────────────────────────────────────
    def train(self, mode: bool = True):
        """
        During training:
          - LoRA layers in stages 2,3 → training mode (dropout active)
          - Everything else           → eval mode (LayerNorm stats frozen)
        """
        super().train(mode)
        if mode:
            # Selectively set only LoRA-containing modules to train mode
            for name, module in self.backbone.named_modules():
                if isinstance(module, LoRALinear):
                    module.train(True)
                elif (_is_frozen_stage(name, self.bcfg.frozen_stages)
                      or not re.search(r'stages\.\d+\.', name)):
                    module.eval()
        return self

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x → [B, 3, H, W]
        Returns dict with keys "f1".."f4" in [B, C, H, W] format.
        """
        raw = self.backbone(x)   # timm features_only returns [f1,f2,f3,f4] as [B,C,H,W]

        # raw is a tuple/list of 4 feature tensors, one per stage
        assert len(raw) == 4, (
            f"Expected 4 intermediate feature maps, got {len(raw)}. "
            "Verify backbone supports get_intermediate_layers(n=4)."
        )

        keys = ["f1", "f2", "f3", "f4"]
        features: Dict[str, torch.Tensor] = {}
        for key, feat, expected_c in zip(keys, raw, CONVNEXT_TINY_CHANNELS):
            features[key] = _to_bchw(feat, expected_c)

        return features

    # ─────────────────────────────────────────────────────────────────────────
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """LoRA adapter params only — goes into the low-LR optimiser group."""
        return [
            p for n, p in self.named_parameters()
            if p.requires_grad and ("lora_A" in n or "lora_B" in n)
        ]

    def summary(self) -> Dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora      = sum(p.numel() for p in self.get_lora_parameters())
        return {
            "total_M"  : total / 1e6,
            "frozen_M" : (total - trainable) / 1e6,
            "lora_K"   : lora / 1e3,
            "lora_pct" : lora / total * 100,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def diagnostic(self):
        """
        Prints all parameters and their requires_grad status.
        Call this after __init__ if LoRA is not being applied correctly.
        """
        print("\n[ADCBackbone] Parameter diagnostic:")
        for name, param in self.named_parameters():
            status = "TRAINABLE" if param.requires_grad else "frozen"
            print(f"  {status:10s}  {name:80s}  {list(param.shape)}")
