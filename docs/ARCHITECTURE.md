# Architecture — AgriCNXEdge / AgroDistill-Cascade

## Pipeline
```
PDDD Leaf/Fruit/Pest + Yield images
  -> Teacher models (PDDD Fruit 1, Fruit3+Leaf1, Pest1, YOLO26s x3) -> soft-label CSVs
  -> ADC Student (DINOv3-ConvNeXt-Tiny [96,192,384,768])
     Frozen [0,1] + LoRA rank8 alpha16 on [2,3] (mlp.fc1/fc2)
     -> FPN-Lite 128ch -> P3 32x32
        ├─ LeafHead  <- f1 (foliar lesions)
        ├─ PestHead  <- f2 (arthropod morphology)
        ├─ YieldHead <- P3 (cx,cy,w,h,obj,cls)
        └─ FruitHead <- ROI-Align 7x7 crops from P3
  -> ONNX 512x512 -> ONNX Runtime Android (NNAPI/GPU/CPU)
  -> Jetpack Compose UI + CameraX
```

## Key configs (student_config.py)
* image_size 512, neck 128, grid 32, anchors 1, ROI 7
* Train 200 epochs, Phase1 freeze 5, batch 8x4=32, lr_lora 1e-5, lr_heads 1e-4, cosine, patience 20
* Loss: alpha 0.5 Focal gamma2 + KL, beta 0.5 teacher box, lambda 7.5/1.0/0.5
* Infer: conf 0.3, NMS 0.45, max 300 boxes

## Why novel
Single 512px model does 4 tasks offline on phone vs 4 separate cloud models. Distillation + LoRA keeps accuracy while fitting mobile.

Add screenshots here for Devpost gallery:
* docs/architecture.png
* docs/confusion-matrix.png
* demo/*.jpg
