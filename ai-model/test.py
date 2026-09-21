import torch
from student_config import get_config

cfg = get_config()
weights_path = cfg.paths.dinov3_weights

print(f"Loading checkpoint: {weights_path}")
ckpt = torch.load(weights_path, map_location="cpu")

# Handle nested state_dict wrappers
if isinstance(ckpt, dict):
    if "model" in ckpt:
        ckpt = ckpt["model"]
    elif "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

print("\n--- FIRST 15 KEYS IN YOUR DINOv3 WEIGHT FILE ---")
for i, key in enumerate(list(ckpt.keys())[:15]):
    shape = list(ckpt[key].shape) if hasattr(ckpt[key], "shape") else "N/A"
    print(f"{i+1:2d}. {key:<50} shape: {shape}")