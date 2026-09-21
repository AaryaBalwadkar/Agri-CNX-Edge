plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.aarya.agricnxedge"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.aarya.agricnxedge"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        ndk { abiFilters.addAll(listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")) }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // ── 16 KB page-size workaround ────────────────────────────────
    packaging {
        jniLibs {
            // true = extract .so files to disk on install.
            // Combined with extractNativeLibs="true" in the manifest,
            // this lets the OS linker align libs at runtime on 4 KB devices.
            useLegacyPackaging = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)

    // ── CameraX ────────────────────────────────────────────────────────────
    val cameraxVersion = "1.3.4"   // Latest stable as of mid-2025
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)    // Camera2 backend
    implementation(libs.androidx.camera.lifecycle)  // bindToLifecycle()
    implementation(libs.androidx.camera.view)       // PreviewView widget

//    // ── PyTorch Mobile Lite ────────────────────────────────────────────
//    val pytorchVersion = "2.1.0"
//    implementation(libs.pytorch.android.lite)
//    implementation(libs.pytorch.android.torchvision.lite)

    // ── ADD: ONNX Runtime 1.24.3 (16 KB ELF — Play Store compliant) ─────
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
}