# Paste-ready Devpost content — BunnieX Hackathon

## Project overview tab

**Project name (60 chars):**
AgriCNXEdge - Offline Edge AI for Apple Crops

**Elevator pitch (200 chars):**
On-device YOLO + distilled AI app that detects apple leaf, pest, fruit disease & counts yield offline to help farmers act fast

**Built with (tags):**
Python, PyTorch, Kotlin, Jetpack Compose, CameraX, ONNX, ONNX Runtime, DINOv3, LoRA, Ultralytics YOLO, OpenCV, Albumentations, scikit-learn, Gradle, Firebase

**Try it out links:**
1. https://github.com/[YOUR-USERNAME]/bunniex-agricnxedge
2. [YOUR DRIVE APK LINK / DEMO LINK]

**Image gallery:** upload 4-5 from demo/ + 1 architecture diagram (3:2, <5MB)

**Video demo link:** https://www.youtube.com/watch?v=[YOUR-ID] (must be Public or Unlisted, embedding ON)

---

## Project details tab — About the project (Markdown)

```markdown
## Inspiration
Apple farmers in Himachal/Kashmir lose 20-30% yield spotting scab, rot, mites too late. Cloud AI fails without network. We wanted a fully offline doctor for orchards.

## What it does
AgriCNXEdge Android app: point camera at leaf/fruit -> instant disease/pest + severity + apple count. Works in airplane mode. Features: live CameraX, gallery import, box overlays, bench mode, history.

## How we built it
Teachers on PDDD + YOLO26s -> soft labels -> single ADC student (DINOv3-ConvNeXt-Tiny + FPN-Lite + 4 heads, LoRA r8) 512px, distilled (alpha 0.5). Exported to ONNX, run with ONNX Runtime Android 1.24.3 + NNAPI. UI in Kotlin Compose Material3.

Stack: Python, PyTorch, ONNX, Kotlin, Compose, CameraX, Albumentations.

Demo: [YOUTUBE LINK]
Code: https://github.com/[YOUR-USERNAME]/bunniex-agricnxedge
APK: [DRIVE LINK]

## Challenges we ran into
Large DINOv3 weights for mobile, 16KB page-size Play compliance, YOLO txt conversion, offline 512px latency, balancing 4-task loss.

## Accomplishments that we're proud of
Single model for 4 tasks, <1s on-device, fully offline, bench harness + phase2 eval with P/R/F1 + MAE for counting.

## What we learned
Distillation, LoRA fine-tuning, ONNX quantization, CameraX + ONNX threading, edge UX for farmers.

## What's next for AgriCNXEdge
Treatment advisory via Featherless AI LLM, Telugu/Hindi voice, fertilizer dose, iOS, 50-farmer pilot, mAP + F1 publication.
```

## Submit tab
Tick: I agree to Official Rules + Terms of Service -> Submit project. You can edit till Sep 22 5pm IST.
