# AgriCNXEdge — Edge AI for Apple Crop Health

> BunnieX Hackathon submission — AI-powered offline Android app for apple leaf disease, pest, fruit disease + yield counting.

**Demo Video:** [PASTE YOUR YOUTUBE LINK HERE - Public/Unlisted]  
**Source Code:** GitHub repo (this repo)  
**Devpost:** https://buuniex-hackathon.devpost.com/

## Problem
Farmers lose 20-30% apple yield to late disease/pest detection. Most AI tools need internet, are cloud-only, and don't work in orchards. Lab models are too heavy for phones.

## Solution
AgriCNXEdge runs a single distilled multi-task model fully on-device:

* Leaf classification (4): Apple_Mosaic, Apple_Black_rot, Alternaria, Healthy
* Pest classification (4): xylotrechus, aphids, leafhoppers, spider_mite
* Fruit classification (3): Anthracnose, Black Rot, Healthy
* Yield detection: apple counting boxes (YOLO-style, 32x32 anchor-free grid)

App features: CameraX live capture, gallery pick, offline inference with ONNX Runtime, boxes + labels overlay, bench mode for accuracy test, 100% offline history.

## How it works
1. Teacher models (PDDD Fruit/Leaf/Pest + YOLO26s yield) generate soft-labels
2. Student: DINOv3-ConvNeXt-Tiny backbone + FPN-Lite + 4 heads, trained with distillation
   * Frozen stages [0,1], LoRA rank 8 on stages [2,3]
   * Input 512x512, neck 128ch, ROI 7x7
3. Export to ONNX (`adc_student_full.onnx`)
4. Android app runs ONNX with NNAPI/GPU/CPU fallback, ImageNet normalize, conf 0.3, NMS 0.45

See `docs/ARCHITECTURE.md` and `ai-model/` for training code.

## Repo structure
```
bunniex-agricnxedge/
  android-app/        # AgriCNXEdge Android app (Kotlin, Compose, CameraX, ONNX Runtime)
    app/src/main/java/com/aarya/agricnxedge/
      MainActivity.kt       # UI
      camera/               # CameraManager
      ml/                   # AgriModelRunner, BenchHarness
  ai-model/           # AgroDistill-Cascade student training
    student_config.py
    student_backbone.py
    student_model.py
    student_dataset.py
    student_loss.py
    student_train.py
    run_student.py
    phase2_eval.py
  demo/               # sample images + video script
  docs/               # architecture
```

## Quickstart

### 1. Android app (optional - judges can evaluate via video + code)
```
1. Open android-app/ in Android Studio Ladybug+
2. Sync Gradle, run on physical device (minSdk 26)
3. Grant camera permission, tap Capture / Pick image
   Note: model file adc_student_full.onnx is excluded from git due to size.
   Place your local copy in android-app/app/src/main/assets/ to run.
```

### 2. AI model training / eval
```bash
cd ai-model
pip install -r ../requirements.txt

# Train
python run_student.py --epochs 200 --batch 8

# Eval only
python phase2_eval.py --preflight-only
python phase2_eval.py --configs proposed

# Export ONNX
python run_student.py --export-only
```

Weights needed: DINOv3 ConvNeXt-Tiny `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` + soft-label CSVs (see `ai-model/README.md`). Not committed due to size.

## Tech stack
`Python, PyTorch, Albumentations, scikit-learn, Ultralytics YOLO, DINOv3, LoRA, ONNX, ONNX Runtime Android 1.24.3, Kotlin, Jetpack Compose, CameraX, Material3, Gradle`

## Impact & future
Offline edge AI cuts diagnosis time from days to seconds, works without internet. Next: multilingual advisory with Featherless AI LLM, treatment recommender, iOS, pilot with 50 farmers.

## Team
Aarya Balwadkar
Eccha Bansal
Symbiosis Institute of Technology, Pune, India

## License
MIT - see LICENSE
