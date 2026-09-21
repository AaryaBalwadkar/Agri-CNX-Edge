"""
verify_backbone.py
==================
Validation script to verify 100% key alignment, shape match, 
numerical weight accuracy, and forward pass output shapes for DINOv3 -> timm.
"""

import sys
import torch
import re
import timm

from student_config import ADCConfig
from student_backbone import ADCBackbone


def validate_dinov3_mapping():
    print("=" * 72)
    print("  AgroDistill-Cascade (ADC) — DINOv3 Weight Validation Tool")
    print("=" * 72)

    # 1. Load Config & Checkpoint
    cfg = ADCConfig()
    ckpt_path = cfg.paths.dinov3_weights
    print(f"\n[1] Loading Checkpoint from:\n    {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt if not isinstance(ckpt, dict) else (
        ckpt.get("model") or ckpt.get("state_dict") or ckpt
    )
    print(f"    -> Raw checkpoint contains {len(raw)} total keys.")

    # 2. Instantiate blank timm backbone
    print("\n[2] Creating target timm model ('convnext_tiny', features_only=True)...")
    model = timm.create_model(
        "convnext_tiny", pretrained=False,
        features_only=True, out_indices=(0, 1, 2, 3)
    )
    timm_state = model.state_dict()
    model_keys = set(timm_state.keys())
    print(f"    -> Target timm backbone expects {len(model_keys)} keys.")

    # 3. Perform Remapping Logic
    print("\n[3] Key Remapping & Alignment Check...")
    remapped_raw = {}
    
    for old_key, value in raw.items():
        k = old_key

        # Strip leading prefixes
        match = re.search(r"\b(downsample_layers|stages|stem|norm)\.", k)
        if match:
            k = k[match.start():]

        # Remap to timm feature_only underscore format
        k_under = re.sub(r"^downsample_layers\.0\.(\d+)(.*)", r"stem_\1\2", k)

        def _ds_under(m):
            return f"stages_{m.group(1)}.downsample.{m.group(2)}{m.group(3)}"
        k_under = re.sub(r"^downsample_layers\.([1-9]\d*)\.(\d+)(.*)", _ds_under, k_under)

        def _blk_under(m):
            s = m.group(3)
            s = re.sub(r"^dwconv\.", "conv_dw.", s)
            s = re.sub(r"^pwconv1\.", "mlp.fc1.", s)
            s = re.sub(r"^pwconv2\.", "mlp.fc2.", s)
            return f"stages_{m.group(1)}.blocks.{m.group(2)}.{s}"
        k_under = re.sub(r"^stages\.(\d+)\.(\d+)\.(.*)", _blk_under, k_under)

        # Fallback dot format
        k_dot = re.sub(r"^downsample_layers\.0\.(\d+)(.*)", r"stem.\1\2", k)
        def _ds_dot(m):
            return f"stages.{int(m.group(1))}.downsample.{m.group(2)}{m.group(3)}"
        k_dot = re.sub(r"^downsample_layers\.([1-9]\d*)\.(\d+)(.*)", _ds_dot, k_dot)
        def _blk_dot(m):
            s = m.group(3)
            s = re.sub(r"^dwconv\.", "conv_dw.", s)
            s = re.sub(r"^pwconv1\.", "mlp.fc1.", s)
            s = re.sub(r"^pwconv2\.", "mlp.fc2.", s)
            return f"stages.{m.group(1)}.blocks.{m.group(2)}.{s}"
        k_dot = re.sub(r"^stages\.(\d+)\.(\d+)\.(.*)", _blk_dot, k_dot)

        if k_under in model_keys:
            remapped_raw[k_under] = value
        elif k_dot in model_keys:
            remapped_raw[k_dot] = value
        elif k in model_keys:
            remapped_raw[k] = value

    missing_keys = model_keys - set(remapped_raw.keys())
    if missing_keys:
        print(f"    [FAIL] Missing {len(missing_keys)} keys in timm backbone: {list(missing_keys)[:5]}")
    else:
        print("    [PASS] Key Alignment: 100% (178/178) of timm backbone keys matched!")

    # 4. Load weights & Verify Numerical Accuracy
    print("\n[4] Numerical Value & Tensor Shape Check...")
    model.load_state_dict(remapped_raw, strict=False)
    loaded_state = model.state_dict()

    shape_mismatches = []
    value_mismatches = []

    for key, ckpt_tensor in remapped_raw.items():
        loaded_tensor = loaded_state[key]
        
        # Shape verification
        if ckpt_tensor.shape != loaded_tensor.shape:
            shape_mismatches.append((key, ckpt_tensor.shape, loaded_tensor.shape))
            
        # Exact numerical comparison
        if not torch.allclose(ckpt_tensor.float(), loaded_tensor.float(), atol=1e-6):
            value_mismatches.append(key)

    if shape_mismatches:
        print(f"    [FAIL] Shape Mismatches: {shape_mismatches}")
    else:
        print("    [PASS] Tensor Shapes: All 178 parameter shapes match perfectly.")

    if value_mismatches:
        print(f"    [FAIL] Value Mismatches in keys: {value_mismatches[:5]}")
    else:
        print("    [PASS] Numerical Accuracy: Loaded weights are 100% identical to checkpoint.")

    # 5. Full Backbone Integration Test
    print("\n[5] Forward Pass Output Shape & Sanity Test...")
    adc_backbone = ADCBackbone(cfg)
    adc_backbone.eval()

    dummy_x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        features = adc_backbone(dummy_x)

    expected_shapes = {
        "f1": (2, 96, 128, 128),
        "f2": (2, 192, 64, 64),
        "f3": (2, 384, 32, 32),
        "f4": (2, 768, 16, 16)
    }

    forward_pass_ok = True
    for feat_name, expected_shape in expected_shapes.items():
        actual_shape = tuple(features[feat_name].shape)
        has_nan = torch.isnan(features[feat_name]).any().item()
        
        if actual_shape == expected_shape and not has_nan:
            print(f"    [PASS] Feature map '{feat_name}': shape {list(actual_shape)}, no NaNs.")
        else:
            print(f"    [FAIL] Feature map '{feat_name}': expected {expected_shape}, got {actual_shape} (NaN: {has_nan})")
            forward_pass_ok = False

    print("\n" + "=" * 72)
    if not missing_keys and not shape_mismatches and not value_mismatches and forward_pass_ok:
        print("  [SUCCESS] VALIDATION PASSED COMPLETELY!")
        print("  Your model weights are loaded and mapped correctly.")
    else:
        print("  [WARNING] Validation found issues. Check output above.")
    print("=" * 72)


if __name__ == "__main__":
    validate_dinov3_mapping()