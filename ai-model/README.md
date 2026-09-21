# ai-model — AgroDistill-Cascade (ADC) Student

Multi-task distillation student for leaf/pest/fruit classification + apple detection.

Copied from `Proposed Model/` in research repo. Only `.py` sources included — weights, data, outputs excluded (too large for GitHub).

## Files
* `student_config.py` — single source of truth, edit paths here
* `student_backbone.py` — DINOv3-ConvNeXt-Tiny + LoRA stages [2,3], rank 8
* `student_model.py` — FPN-Lite neck + 4 heads
* `student_dataset.py` — 512px, ImageNet norm, Albumentations
* `student_loss.py` — Focal + KL distill (alpha 0.5) + detection (box 7.5 / obj 1.0 / cls 0.5)
* `student_train.py` — trainer + ONNX export
* `run_student.py` — entry point
* `phase2_eval.py` — GPU eval: P/R/F1, confusion matrix, MAE/RMSE for counting
* `test.py`, `verify_backbone.py`, `convert_fruit_annotations.py` — utils

## Setup (college GPU machine)
```bash
pip install -r ../requirements.txt
```

Required files (NOT in git, place under ai-model/ mirroring original):
```
ai-model/weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
ai-model/outputs/leaf/leaf_soft_labels.csv
ai-model/outputs/pest/pest_soft_labels.csv
ai-model/outputs/fruit/fruit_soft_labels.csv
ai-model/outputs/yield/yield_soft_labels.csv
ai-model/data/leaf/, pest/, fruit/, yield/
```

Original layout expected: `Proposed Model/` = this folder. Either rename `ai-model/` to `Proposed Model/` on training machine, or edit `student_config.py` BASE_DIR paths.

## Train
```bash
python run_student.py
python run_student.py --epochs 50 --batch 4 --patience 20
python run_student.py --resume
python run_student.py --export-only
```

Outputs to `outputs/student/`: `checkpoints/best_model.pth`, `training_history.json`, `adc_student.onnx`

Copy final ONNX as `adc_student_full.onnx` into `android-app/app/src/main/assets/` for mobile.

## Eval
```bash
python phase2_eval.py --preflight-only
python phase2_eval.py --configs proposed
```
