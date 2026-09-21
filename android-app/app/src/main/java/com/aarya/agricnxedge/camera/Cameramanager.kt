package com.aarya.agricnxedge.camera

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.aarya.agricnxedge.ui.theme.CameraPreviewBg
import com.aarya.agricnxedge.ui.theme.SageGreen
import kotlinx.coroutines.delay
import java.util.concurrent.Executors


// ─────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────

private const val TAG              = "AgriCNX-Camera"
private const val TARGET_SIZE      = 512           // ML model input dimension

// ─────────────────────────────────────────────────────────────────
//  Bitmap helper
// ─────────────────────────────────────────────────────────────────

/**
 * Scales any [Bitmap] to exactly [TARGET_SIZE]×[TARGET_SIZE] pixels using
 * bilinear filtering. This is the format expected by the on-device ML model.
 */
fun Bitmap.resizeForModel(): Bitmap {
    // Gallery (ImageDecoder) can return HARDWARE bitmaps which crash
    // getPixels() inside the runner. Normalise to software ARGB_8888
    // here so camera AND gallery feed the identical tensor path as bench
    // (bench uses BitmapFactory → always software).
    val software: Bitmap = if (this.config == Bitmap.Config.HARDWARE) {
        this.copy(Bitmap.Config.ARGB_8888, false) ?: this
    } else this
    val scaled = Bitmap.createScaledBitmap(software, TARGET_SIZE, TARGET_SIZE, /* filter= */ true)
    // createScaledBitmap preserves HARDWARE config; force software for inference.
    val out = if (scaled.config == Bitmap.Config.HARDWARE) {
        scaled.copy(Bitmap.Config.ARGB_8888, false)?.also {
            if (it !== scaled) scaled.recycle()
        } ?: scaled
    } else scaled
    // Recycle intermediate software copy only if it is neither the
    // caller's bitmap nor the returned bitmap (createScaledBitmap may
    // return its input when sizes already match).
    if (software !== this && software !== out) software.recycle()
    if (scaled !== out && scaled !== software && scaled !== this) scaled.recycle()
    return out
}

// ─────────────────────────────────────────────────────────────────
//  CameraState  (hoisted into MainScreen, passed down as lambdas)
// ─────────────────────────────────────────────────────────────────

/**
 * Holds the live [ImageCapture] use-case once the camera is bound.
 * Kept in the composition tree via [remember] in [com.aarya.agricnxedge.MainScreen] so the
 * capture reference survives recompositions.
 */
class CameraState {
    var imageCapture: ImageCapture? by mutableStateOf(null)
        internal set
}

@Composable
fun rememberCameraState(): CameraState = remember { CameraState() }

// ─────────────────────────────────────────────────────────────────
//  Permission-aware camera composable  (top-level entry point)
// ─────────────────────────────────────────────────────────────────

/**
 * Renders a live camera feed inside [modifier].  Handles Android runtime
 * camera permission transparently:
 *
 * • If permission is already granted → bind camera immediately.
 * • If not yet requested           → show a rationale card and request.
 * • If permanently denied          → show an instructional card.
 *
 * @param cameraState        Shared [CameraState]; [ImageCapture] is populated
 *                           once the camera is successfully bound.
 * @param modifier           Applied to the root container (typically fills the
 *                           top 70% of the screen).
 */
@Composable
fun CameraPreviewWithPermission(
    cameraState: CameraState,
    modifier:    Modifier = Modifier,
) {
    val context        = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var permissionGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    var permissionDeniedPermanently by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            permissionGranted           = true
            permissionDeniedPermanently = false
        } else {
            permissionGranted           = false
            permissionDeniedPermanently = true
        }
    }

    // ── Lifecycle observer: re-check permission on every ON_RESUME ──
    // Fires when the user returns from the Android Settings page.
    // DisposableEffect ensures the observer is removed when this
    // composable leaves the tree, preventing memory leaks.
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                val granted = ContextCompat.checkSelfPermission(
                    context, Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED

                permissionGranted = granted
                if (granted) permissionDeniedPermanently = false
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    // Request on first composition if not already granted
    LaunchedEffect(Unit) {
        if (!permissionGranted) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    Box(modifier = modifier) {
        when {
            permissionGranted          -> CameraPreviewContent(
                cameraState = cameraState,
                modifier    = Modifier.fillMaxSize(),
            )
            permissionDeniedPermanently -> PermissionDeniedCard(modifier = Modifier.fillMaxSize())
            else                        -> PermissionRationaleCard(
                modifier  = Modifier.fillMaxSize(),
                onRequest = { permissionLauncher.launch(Manifest.permission.CAMERA) },
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Live camera feed
// ─────────────────────────────────────────────────────────────────

/**
 * Binds a CameraX [Preview] and [ImageCapture] use-case to the current
 * lifecycle owner using [ProcessCameraProvider].  The [PreviewView] is
 * embedded via [AndroidView].
 */
@Composable
private fun CameraPreviewContent(
    cameraState: CameraState,
    modifier:    Modifier = Modifier,
) {
    val context        = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // Single-thread executor for ImageCapture callbacks
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    // Keep a reference to PreviewView so we can inspect it if needed
    val previewView = remember { PreviewView(context) }

    // Bind use-cases once
    LaunchedEffect(previewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // ── Use-case: stream to PreviewView ───────────────
            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            // ── Use-case: single-frame capture ────────────────
            val imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            // Store reference so LowerThirdControls can trigger capture
            cameraState.imageCapture = imageCapture

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture,
                )
                Log.i(TAG, "CameraX use-cases bound successfully.")
            } catch (exc: Exception) {
                Log.e(TAG, "Camera bind failed: ${exc.message}", exc)
            }

        }, ContextCompat.getMainExecutor(context))
    }

    // Cleanup executor when the composable leaves the tree
    DisposableEffect(Unit) {
        onDispose { cameraExecutor.shutdown() }
    }

    Box(modifier = modifier.background(CameraPreviewBg)) {
        // ── Live PreviewView ──────────────────────────────────
        AndroidView(
            factory  = { previewView },
            modifier = Modifier.fillMaxSize(),
        )

        // ── Scan frame overlay ────────────────────────────────
        ScanFrameOverlay(
            modifier = Modifier
                .fillMaxSize()
                .padding(32.dp),
        )

        // ── Bottom gradient fade ──────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(72.dp)
                .align(Alignment.BottomCenter)
                .background(
                    brush = Brush.verticalGradient(
                        colors = listOf(
                            Color.Transparent,
                            Color(0xFF1A1C18).copy(alpha = 0.70f),
                        )
                    )
                )
        )

        // ── Live indicator ────────────────────────────────────
        LiveIndicatorBadge(modifier = Modifier
            .align(Alignment.TopEnd)
            .padding(16.dp)
        )
    }
}

// ─────────────────────────────────────────────────────────────────
//  Capture + resize  (called from LowerThirdControls)
// ─────────────────────────────────────────────────────────────────

/**
 * Instructs [ImageCapture] to grab the current frame, converts the
 * resulting [ImageProxy] to a [Bitmap], resizes it to 512×512, then
 * invokes [onBitmapReady] on the calling thread.
 *
 * All heavy work happens on the supplied [executor]; [onBitmapReady] is
 * posted back to the main thread for safe UI access.
 */
fun CameraState.captureAndResize(
    onBitmapReady: (Bitmap) -> Unit,
    onError:       (String) -> Unit = { Log.e(TAG, it) },
) {
    val capture = imageCapture ?: run {
        onError("ImageCapture not initialised – camera may still be binding.")
        return
    }

    capture.takePicture(
        Executors.newSingleThreadExecutor(),
        object : ImageCapture.OnImageCapturedCallback() {

            override fun onCaptureSuccess(image: ImageProxy) {
                try {
                    // toBitmap() is available in camera-core 1.3+.
                    // The sensor frame can carry EXIF-style rotation
                    // (imageInfo.rotationDegrees, usually 90); toBitmap()
                    // does NOT apply it, so an unrotated bitmap would feed
                    // the model a sideways orchard. Upright first.
                    val raw       = image.toBitmap()
                    val rotation  = image.imageInfo.rotationDegrees
                    val upright   =
                        if (rotation == 0) raw
                        else Bitmap.createBitmap(
                            raw, 0, 0, raw.width, raw.height,
                            Matrix().apply { postRotate(rotation.toFloat()) },
                            true,
                        ).also { if (it != raw) raw.recycle() }
                    val resized = upright.resizeForModel()
                    val sensorW = image.width
                    val sensorH = image.height
                    image.close()

                    Log.i(
                        TAG,
                        "Image Captured upright (rotation=${rotation}deg) and " +
                                "Resized to ${TARGET_SIZE}x${TARGET_SIZE} | " +
                                "sensor=${sensorW}x${sensorH} | " +
                                "resized=${resized.width}x${resized.height}"
                    )

                    // Hand the bitmap off on the main thread
                    Handler(Looper.getMainLooper())
                        .post { onBitmapReady(resized) }

                } catch (e: Exception) {
                    image.close()
                    onError("Bitmap conversion failed: ${e.message}")
                }
            }

            override fun onError(exception: ImageCaptureException) {
                onError("Capture error [${exception.imageCaptureError}]: ${exception.message}")
            }
        }
    )
}

// ─────────────────────────────────────────────────────────────────
//  Permission UI: rationale card
// ─────────────────────────────────────────────────────────────────

@Composable
private fun PermissionRationaleCard(
    modifier:  Modifier = Modifier,
    onRequest: () -> Unit,
) {
    Box(
        modifier         = modifier.background(CameraPreviewBg),
        contentAlignment = Alignment.Center,
    ) {
        Card(
            modifier = Modifier
                .padding(32.dp)
                .fillMaxWidth(),
            shape  = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(
                containerColor = Color(0xFF2A2D27),
            ),
        ) {
            Column(
                modifier            = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Icon(
                    imageVector        = Icons.Outlined.CameraAlt,
                    contentDescription = null,
                    tint               = SageGreen,
                    modifier           = Modifier.size(48.dp),
                )
                Text(
                    text      = "Camera Access Required",
                    style     = MaterialTheme.typography.titleMedium,
                    color     = Color.White,
                    textAlign = TextAlign.Center,
                )
                Text(
                    text      = "Agri-CNX-Edge needs live camera access to perform real-time tree analysis using the on-device AI model.",
                    style     = MaterialTheme.typography.bodySmall,
                    color     = Color.White.copy(alpha = 0.65f),
                    textAlign = TextAlign.Center,
                )
                Button(
                    onClick = onRequest,
                    colors  = ButtonDefaults.buttonColors(containerColor = SageGreen),
                    shape   = RoundedCornerShape(12.dp),
                ) {
                    Text("Grant Camera Permission")
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Permission UI: permanently denied
// ─────────────────────────────────────────────────────────────────

@Composable
private fun PermissionDeniedCard(modifier: Modifier = Modifier) {
    val context = LocalContext.current

    Box(
        modifier         = modifier.background(CameraPreviewBg),
        contentAlignment = Alignment.Center,
    ) {
        Card(
            modifier = Modifier
                .padding(32.dp)
                .fillMaxWidth(),
            shape  = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF2A2D27)),
        ) {
            Column(
                modifier            = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Icon(
                    imageVector        = Icons.Outlined.Lock,
                    contentDescription = null,
                    tint               = Color(0xFFD32F2F),
                    modifier           = Modifier.size(48.dp),
                )
                Text(
                    text      = "Camera Permission Denied",
                    style     = MaterialTheme.typography.titleMedium,
                    color     = Color.White,
                    textAlign = TextAlign.Center,
                )
                Text(
                    text      = "Camera access was permanently denied. Open Settings to grant it, then return to the app — it will resume automatically.",
                    style     = MaterialTheme.typography.bodySmall,
                    color     = Color.White.copy(alpha = 0.65f),
                    textAlign = TextAlign.Center,
                )
                Button(
                    onClick = {
                        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                            data  = Uri.fromParts("package", context.packageName, null)
                            flags = Intent.FLAG_ACTIVITY_NEW_TASK
                        }
                        context.startActivity(intent)
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = SageGreen),
                    shape  = RoundedCornerShape(12.dp),
                ) {
                    Text("Open Settings")
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Decorative composables (scan frame + live badge)
// ─────────────────────────────────────────────────────────────────

@Composable
internal fun ScanFrameOverlay(modifier: Modifier = Modifier) {
    val bracketColor = SageGreen.copy(alpha = 0.65f)
    val stroke       = 3.dp
    val armLen       = 30.dp

    Box(modifier = modifier) {
        listOf(
            Alignment.TopStart,
            Alignment.TopEnd,
            Alignment.BottomStart,
            Alignment.BottomEnd,
        ).forEach { alignment ->
            BracketCorner(
                alignment   = alignment,
                color       = bracketColor,
                strokeWidth = stroke,
                armLength   = armLen,
            )
        }
    }
}

@Composable
private fun BoxScope.BracketCorner(
    alignment:   Alignment,
    color:       Color,
    strokeWidth: Dp,
    armLength:   Dp,
) {
    Canvas(
        modifier = Modifier
            .size(armLength + strokeWidth)
            .align(alignment)
    ) {
        val sw      = strokeWidth.toPx()
        val al      = armLength.toPx()
        val isRight  = alignment == Alignment.TopEnd    || alignment == Alignment.BottomEnd
        val isBottom = alignment == Alignment.BottomStart || alignment == Alignment.BottomEnd
        val xOrigin  = if (isRight)  size.width  else 0f
        val yOrigin  = if (isBottom) size.height else 0f
        val xDir     = if (isRight)  -1f else 1f
        val yDir     = if (isBottom) -1f else 1f

        drawLine(
            color       = color,
            start       = Offset(xOrigin, yOrigin),
            end         = Offset(xOrigin + xDir * al, yOrigin),
            strokeWidth = sw,
            cap         = StrokeCap.Round,
        )
        drawLine(
            color       = color,
            start       = Offset(xOrigin, yOrigin),
            end         = Offset(xOrigin, yOrigin + yDir * al),
            strokeWidth = sw,
            cap         = StrokeCap.Round,
        )
    }
}

/** Pulsing "LIVE" badge shown in the top-right corner of the viewfinder. */
@Composable
private fun LiveIndicatorBadge(modifier: Modifier = Modifier) {
    var visible by remember { mutableStateOf(true) }

    LaunchedEffect(Unit) {
        while (true) {
            delay(900)
            visible = !visible
        }
    }

    Surface(
        modifier = modifier,
        shape    = RoundedCornerShape(8.dp),
        color    = Color(0xCC1A1C18),
    ) {
        Row(
            modifier          = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            AnimatedVisibility(
                visible = visible,
                enter   = fadeIn(tween(300)),
                exit    = fadeOut(tween(300)),
            ) {
                Box(
                    modifier = Modifier
                        .size(7.dp)
                        .clip(CircleShape)
                        .background(Color(0xFFE53935))
                )
            }
            Spacer(Modifier.width(6.dp))
            Text(
                text  = "LIVE",
                style = MaterialTheme.typography.labelSmall,
                color = Color.White,
            )
        }
    }
}