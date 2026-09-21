# android-app — AgriCNXEdge

Offline Android app (Kotlin + Compose + CameraX + ONNX Runtime 1.24.3).

Copied from `AgriCNXEdge/` — only source + gradle files. `build/`, `.gradle/`, `.idea/`, `local.properties`, `.onnx` excluded.

## Structure
```
android-app/
  build.gradle.kts
  settings.gradle.kts
  gradle.properties
  gradle/
  gradlew / gradlew.bat
  app/
    build.gradle.kts (compileSdk 36, minSdk 26, onnxruntime-android 1.24.3)
    src/main/
      AndroidManifest.xml (Camera permission)
      java/com/aarya/agricnxedge/
        MainActivity.kt
        camera/Cameramanager.kt
        ml/Agrimodelrunner.kt (512x512, Mean [0.485,0.456,0.406], conf 0.3, NMS 0.45)
        ml/BenchHarness.kt
```

## Run (for reviewers - build locally if needed)
1. Place your local `adc_student_full.onnx` in `app/src/main/assets/` (excluded from git)
2. Open `android-app/` in Android Studio
3. Let Gradle sync (needs internet first time)
4. Run on real ARM64 device, allow Camera
5. Tap Camera to capture or Gallery to pick leaf/fruit image

No prebuilt APK shared - video demo + source code is the submission.

Model labels must match `student_config.py`:
Leaf: Apple_Mosaic, Apple___Black_rot, Alternaria, Healthy
Pest: xylotrechus, aphids, leafhoppers, spider_mite
Fruit: Anthracnose, Black Rot, Healthy
