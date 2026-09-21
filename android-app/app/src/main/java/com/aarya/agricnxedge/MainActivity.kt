package com.aarya.agricnxedge

// ─────────────────────────────────────────────────────────────────
//  Imports
// ─────────────────────────────────────────────────────────────────

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Agriculture
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aarya.agricnxedge.camera.CameraPreviewWithPermission
import com.aarya.agricnxedge.camera.captureAndResize
import com.aarya.agricnxedge.camera.rememberCameraState
import com.aarya.agricnxedge.camera.resizeForModel
import com.aarya.agricnxedge.ml.AgriModelRunner
import com.aarya.agricnxedge.ml.FRUIT_LABELS
import com.aarya.agricnxedge.ml.LEAF_LABELS
import com.aarya.agricnxedge.ml.ModelOutputs
import com.aarya.agricnxedge.ml.PEST_LABELS
import com.aarya.agricnxedge.ml.ProviderMode
import com.aarya.agricnxedge.ml.runAccuracyV2
import com.aarya.agricnxedge.ml.runPhoneBench
import com.aarya.agricnxedge.ui.theme.AgriCNXEdgeTheme
import com.aarya.agricnxedge.ui.theme.ForestGreen
import com.aarya.agricnxedge.ui.theme.SageGreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// ─────────────────────────────────────────────────────────────────
//  Logcat tag
// ─────────────────────────────────────────────────────────────────

private const val TAG = "AgriCNX-Main"

// ─────────────────────────────────────────────────────────────────
//  Activity
// ─────────────────────────────────────────────────────────────────

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AgriCNXEdgeTheme {
                MainScreen()
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  AnalysisResult UI model
// ─────────────────────────────────────────────────────────────────

private data class AnalysisResult(
    val title:     String,
    val icon:      ImageVector,
    val status:    String,
    val detail:    String,
    val chipText:  String,
    val chipColor: Color,
)

// ─────────────────────────────────────────────────────────────────
//  Placeholder results  (shown before first inference runs)
// ─────────────────────────────────────────────────────────────────

private val placeholderResults = listOf(
    AnalysisResult(
        title     = "Leaf Health",
        icon      = Icons.Outlined.Eco,
        status    = "Awaiting Scan",
        detail    = "Press 'Analyze Tree' to run the on-device leaf health model.",
        chipText  = "Not yet run",
        chipColor = Color(0xFF8B8F84),
    ),
    AnalysisResult(
        title     = "Pest Status",
        icon      = Icons.Outlined.BugReport,
        status    = "Awaiting Scan",
        detail    = "Pest detection will run automatically on the next capture.",
        chipText  = "Not yet run",
        chipColor = Color(0xFF8B8F84),
    ),
    AnalysisResult(
        title     = "Fruit Quality",
        icon      = Icons.Outlined.Stars,
        status    = "Awaiting Scan",
        detail    = "Fruit grading model is loaded and ready on-device.",
        chipText  = "Not yet run",
        chipColor = Color(0xFF8B8F84),
    ),
    AnalysisResult(
        title     = "Estimated Yield",
        icon      = Icons.Outlined.BarChart,
        status    = "Awaiting Scan",
        detail    = "Yield detection head will count fruit clusters in the next frame.",
        chipText  = "Not yet run",
        chipColor = Color(0xFF8B8F84),
    ),
)

// ─────────────────────────────────────────────────────────────────
//  ModelOutputs → AnalysisResult list  (live data mapper)
// ─────────────────────────────────────────────────────────────────

// ── Reliability gating (quick app fix, no retraining) ───────────
// Val accuracy is ~99% leaf / 100% pest / 97% fruit IN-distribution
// (close-ups + GT boxes), but the app runs every head on EVERY image.
// Orchard wide-shots make leaf/pest OOD, and fruit majority-vote over
// noisy predicted boxes hides diseased minorities. Gate on confidence
// + cross-task mismatch instead of showing argmax as confident.
private const val LEAF_MIN_CONF  = 0.65f
private const val PEST_MIN_CONF  = 0.65f
private const val FRUIT_MIN_CONF = 0.60f
private const val FRUIT_DISEASE_CONF = 0.50f
private const val HEALTHY_FRUIT = 2

private fun buildLiveResults(outputs: ModelOutputs): List<AnalysisResult> {
    val isOrchardView = outputs.yieldCount > 0

    // ── Leaf: unreliable on orchard views + low confidence ──
    val leafUncertain = outputs.leafConfidence < LEAF_MIN_CONF
    val leafCard = if (leafUncertain) {
        AnalysisResult(
            title     = "Leaf Health",
            icon      = Icons.Outlined.Eco,
            status    = "Uncertain — retake close-up",
            detail    = "Confidence ${(outputs.leafConfidence * 100).toInt()}% " +
                    "below ${ (LEAF_MIN_CONF * 100).toInt()}%. Fill frame with a " +
                    "single leaf in daylight, hold steady.",
            chipText  = "Retake",
            chipColor = Color(0xFF8B8F84),
        )
    } else {
        val suffix = if (isOrchardView)
            " Orchard view (${outputs.yieldCount} apples) — close-up " +
            "classifiers may be unreliable; confirm with a leaf close-up."
        else ""
        AnalysisResult(
            title     = "Leaf Health",
            icon      = Icons.Outlined.Eco,
            status    = LEAF_LABELS[outputs.leafClass],
            detail    = leafDetailFor(outputs.leafClass) + suffix,
            chipText  = "${(outputs.leafConfidence * 100).toInt()}% Confidence" +
                    if (isOrchardView) "*" else "",
            chipColor = if (isOrchardView) Color(0xFFFFC107)
                        else leafChipColor(outputs.leafClass),
        )
    }

    // ── Pest: same OOD problem on orchard views ──
    val pestUncertain = outputs.pestConfidence < PEST_MIN_CONF
    val pestCard = if (pestUncertain) {
        AnalysisResult(
            title     = "Pest Status",
            icon      = Icons.Outlined.BugReport,
            status    = "Uncertain — retake close-up",
            detail    = "Confidence ${(outputs.pestConfidence * 100).toInt()}% " +
                    "below ${(PEST_MIN_CONF * 100).toInt()}%. Photograph the " +
                    "insect/symptom area up close.",
            chipText  = "Retake",
            chipColor = Color(0xFF8B8F84),
        )
    } else {
        val suffix = if (isOrchardView)
            " Orchard view — pest ID needs an insect close-up; treat as provisional."
        else ""
        AnalysisResult(
            title     = "Pest Status",
            icon      = Icons.Outlined.BugReport,
            status    = PEST_LABELS[outputs.pestClass],
            detail    = pestDetailFor(outputs.pestClass) + suffix,
            chipText  = pestActionFor(outputs.pestClass) +
                    if (isOrchardView) "?" else "",
            chipColor = if (isOrchardView) Color(0xFFFFC107)
                        else pestChipColor(outputs.pestClass),
        )
    }

    // ── Fruit: majority vote hides diseased minorities (false negatives).
    // Flag ANY diseased apple above FRUIT_DISEASE_CONF instead.
    val fruitCard = buildFruitCard(outputs)

    return listOf(
        leafCard,
        pestCard,
        fruitCard,
        // ── Yield card — YOLO grid detections (conf 0.3, NMS IoU 0.45) ──
        // `outputs.yieldCount` is the number of boxes kept after NMS.
        AnalysisResult(
            title     = "Estimated Yield",
            icon      = Icons.Outlined.BarChart,
            status    = "${outputs.yieldCount} apples detected",
            detail    = "YOLO 32×32 grid decoded ${outputs.yieldCount} apple boxes " +
                    "in ${outputs.inferenceMs} ms on-device (conf 0.30, NMS IoU 0.45).",
            chipText  = "Live Detection",
            chipColor = Color(0xFF1976D2),
        ),
    )
}

/**
 * Fruit card with false-negative guard.
 * Old code: pure majority vote — 4 Healthy (55%) + 1 Black Rot (92%)
 * → "Healthy", hiding the diseased apple. For agriculture a missed
 * infection is worse than a false alarm, so ANY diseased apple above
 * FRUIT_DISEASE_CONF flips the card to diseased + "X/Y apples".
 * Low mean confidence or split votes → "Uncertain".
 */
private fun buildFruitCard(outputs: ModelOutputs): AnalysisResult {
    if (outputs.fruitClass < 0 || outputs.fruitPreds.isEmpty()) {
        return AnalysisResult(
            title     = "Fruit Quality",
            icon      = Icons.Outlined.Stars,
            status    = "No apples detected",
            detail    = "No apple boxes passed the detection threshold, " +
                    "so per-apple grading was skipped.",
            chipText  = "—",
            chipColor = Color(0xFF8B8F84),
        )
    }
    val n = outputs.fruitPreds.size
    val diseasedIdx = outputs.fruitPreds.mapIndexedNotNull { i, c ->
        if (c != HEALTHY_FRUIT && outputs.fruitConfs[i] >= FRUIT_DISEASE_CONF) i else null
    }
    val nDiseased = diseasedIdx.size
    // Diseased minority exists but majority says Healthy → flag it.
    if (nDiseased > 0 && outputs.fruitClass == HEALTHY_FRUIT) {
        val votes = diseasedIdx.groupingBy { outputs.fruitPreds[it] }.eachCount()
        val worst = votes.maxByOrNull { it.value }!!.key
        var sum = 0f
        diseasedIdx.filter { outputs.fruitPreds[it] == worst }
            .forEach { sum += outputs.fruitConfs[it] }
        val worstConf = sum / votes[worst]!!
        return AnalysisResult(
            title     = "Fruit Quality",
            icon      = Icons.Outlined.Stars,
            status    = "${FRUIT_LABELS[worst]} ($nDiseased/$n apples)",
            detail    = "Majority Healthy but $nDiseased/$n apples grade as " +
                    "${FRUIT_LABELS[worst]} — flagged for safety. " +
                    fruitDetailFor(worst),
            chipText  = "${(worstConf * 100).toInt()}% on diseased",
            chipColor = fruitChipColor(worst),
        )
    }
    // Low confidence or split vote → uncertain, don't present as fact.
    val distinct = outputs.fruitPreds.toSet().size
    if (outputs.fruitConfidence < FRUIT_MIN_CONF || (distinct > 1 && nDiseased == 0)) {
        val breakdown = outputs.fruitPreds.groupingBy { it }.eachCount()
            .entries.joinToString(", ") { "${FRUIT_LABELS[it.key]}×${it.value}" }
        return AnalysisResult(
            title     = "Fruit Quality",
            icon      = Icons.Outlined.Stars,
            status    = "Uncertain — mixed/low confidence",
            detail    = "Per-apple votes: $breakdown. Top: " +
                    "${FRUIT_LABELS[outputs.fruitClass]} " +
                    "${(outputs.fruitConfidence * 100).toInt()}%. Move closer, " +
                    "one cluster per shot.",
            chipText  = "Retake",
            chipColor = Color(0xFF8B8F84),
        )
    }
    val healthyCount = outputs.fruitPreds.count { it == HEALTHY_FRUIT }
    val suffix = if (n > 1) " ($healthyCount/$n healthy)" else ""
    return AnalysisResult(
        title     = "Fruit Quality",
        icon      = Icons.Outlined.Stars,
        status    = FRUIT_LABELS[outputs.fruitClass] + suffix,
        detail    = fruitDetailFor(outputs.fruitClass),
        chipText  = "${(outputs.fruitConfidence * 100).toInt()}% Confidence",
        chipColor = fruitChipColor(outputs.fruitClass),
    )
}

// ─────────────────────────────────────────────────────────────────
//  Main Screen
// ─────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen() {
    val context = LocalContext.current
    val scope   = rememberCoroutineScope()

    // ── Bottom sheet state ─────────────────────────────────────────
    val sheetState = rememberBottomSheetScaffoldState(
        bottomSheetState = rememberStandardBottomSheetState(
            initialValue    = SheetValue.PartiallyExpanded,
            skipHiddenState = true,
        )
    )

    // ── Camera state ───────────────────────────────────────────────
    // Hoisted here so the live viewfinder and the Analyze button
    // share the same ImageCapture reference without prop-drilling.
    val cameraState = rememberCameraState()

    // ── ML runner — lazy off-thread initialisation ─────────────────
    // AgriModelRunner's constructor:
    //   1. Copies the .onnx asset to the cache dir (disk I/O, ~50–200 ms)
    //   2. Calls OrtEnvironment.createSession() (JNI + mmap, ~200–800 ms)
    // Both steps must NOT run on the Main thread. We dispatch them to
    // Dispatchers.IO inside a LaunchedEffect so the Compose UI is never
    // blocked. The runner is null until the coroutine completes.
    var modelRunner     by remember { mutableStateOf<AgriModelRunner?>(null) }
    var modelInitError  by remember { mutableStateOf<String?>(null) }
    var isModelLoading  by remember { mutableStateOf(true) }

    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            try {
                val runner = AgriModelRunner(context)
                modelRunner    = runner
                Log.i(TAG, "AgriModelRunner initialised successfully.")
            } catch (e: Exception) {
                modelInitError = e.message
                Log.e(TAG, "AgriModelRunner init failed: ${e.message}", e)
            } finally {
                isModelLoading = false
            }
        }
    }

    // ── Inference state ────────────────────────────────────────────
    var capturedBitmap by remember { mutableStateOf<Bitmap?>(null)       }
    var modelOutputs   by remember { mutableStateOf<ModelOutputs?>(null) }
    var isAnalysing    by remember { mutableStateOf(false)               }
    var inferenceError by remember { mutableStateOf<String?>(null)       }
    var benchRunning   by remember { mutableStateOf(false)               }
    var showSettingsDialog by remember { mutableStateOf(false)         }

    // ── Derive the card list shown in the bottom sheet ─────────────
    // Switches from placeholders to live model data the moment
    // modelOutputs becomes non-null.
    val displayResults = if (modelOutputs != null) {
        buildLiveResults(modelOutputs!!)
    } else {
        placeholderResults
    }

    // ── Overall confidence badge text ──────────────────────────────
    // Don't average OOD heads: orchard views make leaf/pest unreliable,
    // and mixed fruit votes are uncertain. Surface "Check" instead of a
    // high-looking average built from false values.
    val overallConfidence = modelOutputs?.let {
        val leafOk = it.leafConfidence >= LEAF_MIN_CONF
        val pestOk = it.pestConfidence >= PEST_MIN_CONF
        val fruitOk = it.fruitClass < 0 || it.fruitConfidence >= FRUIT_MIN_CONF
        val orchardPenalty = it.yieldCount > 0
        if (!leafOk || !pestOk || !fruitOk) {
            "Check*"
        } else if (orchardPenalty) {
            // All confidences high but cross-task mismatch (orchard vs
            // close-up) → keep % but flag for close-up confirmation.
            if (it.fruitClass >= 0) {
                val avg = (it.leafConfidence + it.pestConfidence + it.fruitConfidence) / 3f
                "${(avg * 100).toInt()}% Overall*"
            } else {
                val avg = (it.leafConfidence + it.pestConfidence) / 2f
                "${(avg * 100).toInt()}% Overall*"
            }
        } else {
            if (it.fruitClass >= 0) {
                val avg = (it.leafConfidence + it.pestConfidence + it.fruitConfidence) / 3f
                "${(avg * 100).toInt()}% Overall"
            } else {
                val avg = (it.leafConfidence + it.pestConfidence) / 2f
                "${(avg * 100).toInt()}% Overall*"
            }
        }
    } ?: "Pending"

    // ── Shared analyze entry point (camera capture AND gallery) ──
    // Guards, state updates, and sheet expansion live here so both
    // image sources run the identical inference path.
    val analyzeBitmap: (Bitmap) -> Unit = { bitmap ->
        val runner = modelRunner
        if (!isAnalysing && runner != null) {
            isAnalysing    = true
            inferenceError = null
            capturedBitmap = bitmap

            scope.launch {
                try {
                    // analyzeTree() dispatches to Dispatchers.Default
                    // internally — the Main thread never blocks.
                    val outputs = runner.analyzeTree(bitmap)
                    modelOutputs = outputs
                    Log.i(TAG, "Inference complete → $outputs")

                    sheetState.bottomSheetState.expand()

                } catch (e: Exception) {
                    inferenceError = e.message
                    Log.e(TAG, "Inference failed: ${e.message}", e)

                    // Still open the sheet in dev/demo builds so the
                    // UI is testable even if the model file is absent.
                    sheetState.bottomSheetState.expand()

                } finally {
                    isAnalysing = false
                }
            }
        }
    }

    // ── Gallery fallback (professor demos, offline test photos) ────
    // Any picked image goes through the same resize + analyze path.
    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            scope.launch {
                try {
                    @Suppress("DEPRECATION")
                    val pickedRaw =
                        if (Build.VERSION.SDK_INT >= 28) {
                            val src = ImageDecoder.createSource(
                                context.contentResolver, uri)
                            // Force software ARGB_8888: default HARDWARE
                            // bitmaps crash getPixels() in the runner and
                            // would make gallery always fail while bench
                            // (BitmapFactory) keeps working.
                            ImageDecoder.decodeBitmap(src) { decoder, _, _ ->
                                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                                decoder.isMutableRequired = true
                            }
                        } else {
                            @Suppress("DEPRECATION")
                            android.provider.MediaStore.Images.Media
                                .getBitmap(context.contentResolver, uri)
                        }
                    // resizeForModel() also forces software + 512x512, so
                    // gallery now feeds the identical tensor path as bench.
                    val modelInput = pickedRaw.resizeForModel()
                    if (modelInput !== pickedRaw) pickedRaw.recycle()
                    analyzeBitmap(modelInput)
                } catch (e: Exception) {
                    inferenceError = e.message
                    Log.e(TAG, "Gallery decode failed: ${e.message}", e)
                }
            }
        }
    }

    // ── Analyze button action ──────────────────────────────────────
    val onAnalyzeClick: () -> Unit = {
        // Guard 1: ignore taps if a capture/inference is already in flight.
        // Guard 2: ignore taps while the ONNX session is still loading.
        if (!isAnalysing && modelRunner != null) {
            cameraState.captureAndResize(
                onBitmapReady = { bitmap -> analyzeBitmap(bitmap) },
                onError = { msg ->
                    inferenceError = msg
                    isAnalysing    = false
                    Log.e(TAG, "Capture failed: $msg")
                    scope.launch { sheetState.bottomSheetState.expand() }
                },
            )
        }
    }

    // ── Benchmark action (Phase 4b bench v2, UNIFORM on every phone) ──
    // Accuracy-once (bench_v2, 50 images x1) then latency-repeat
    // (bench, 10 warmup + 100 measured) per provider mode.
    val onBenchClick: () -> Unit = {
        if (!isAnalysing && !benchRunning) {
            benchRunning   = true
            inferenceError = null
            scope.launch {
                try {
                    val a1 = runAccuracyV2(context, ProviderMode.NNAPI_PREFERRED)
                    val a2 = runAccuracyV2(context, ProviderMode.CPU_ONLY)
                    val s1 = runPhoneBench(context, ProviderMode.NNAPI_PREFERRED,
                        warmupRuns = 10, measuredTotal = 100)
                    val s2 = runPhoneBench(context, ProviderMode.CPU_ONLY,
                        warmupRuns = 10, measuredTotal = 100)
                    inferenceError =
                        "Bench v2 done — acc NNAPI ${a1.csvPath} ; acc CPU ${a2.csvPath} ; " +
                        "lat NNAPI ${s1.meanMs.toInt()}ms ; lat CPU ${s2.meanMs.toInt()}ms. " +
                        "CSVs: ${s1.csvPath} ; ${s2.csvPath}"
                    Log.i(TAG, "Benchmark v2 complete → $inferenceError")
                } catch (e: Exception) {
                    inferenceError = "Benchmark failed: ${e.message}"
                    Log.e(TAG, "Benchmark failed: ${e.message}", e)
                } finally {
                    benchRunning = false
                }
                sheetState.bottomSheetState.expand()
            }
        }
    }

    // ── Determine button label ─────────────────────────────────────
    val buttonLabel = when {
        isModelLoading                   -> "Loading Model…"
        modelInitError != null           -> "Model Error"
        isAnalysing                      -> "Analysing…"
        cameraState.imageCapture == null -> "Initialising…"
        else                             -> "Analyze Tree"
    }

    // ── Button enabled guard ───────────────────────────────────────
    // All four conditions must be true before we allow a capture.
    val analyzeEnabled = modelRunner != null
            && !isModelLoading
            && !isAnalysing
            && cameraState.imageCapture != null

    // ─────────────────────────────────────────────────────────────
    //  Scaffold
    // ─────────────────────────────────────────────────────────────

    BottomSheetScaffold(
        scaffoldState       = sheetState,
        sheetPeekHeight     = 0.dp,
        sheetDragHandle     = { SheetDragHandle() },
        sheetShape          = RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp),
        sheetContainerColor = MaterialTheme.colorScheme.surface,
        sheetContent = {
            AnalysisResultsSheet(
                results           = displayResults,
                overallConfidence = overallConfidence,
                inferenceError    = inferenceError ?: modelInitError,
                photo             = capturedBitmap,
                outputs           = modelOutputs,
            )
        },
        topBar = { AgriTopBar(onSettingsClick = { showSettingsDialog = true }) },
    ) { innerPadding ->

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            // ── Top 70%: Live CameraX viewfinder ──────────────────
            // All runtime permission handling (rationale card, deep-link
            // Settings button, ON_RESUME re-check) lives inside this composable.
            // Wrapped in a Box so we can overlay the CircularProgressIndicator
            // during active inference without disrupting the viewfinder.
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(0.70f),
            ) {
                CameraPreviewWithPermission(
                    cameraState = cameraState,
                    modifier    = Modifier.fillMaxSize(),
                )

                // ── Inference overlay indicator ────────────────────
                // A semi-transparent scrim + spinner appears over the
                // live viewfinder while the ONNX session is running.
                // This gives immediate, unambiguous feedback that the
                // capture has been taken and inference is in progress.
                if (isAnalysing) {
                    Box(
                        modifier          = Modifier
                            .fillMaxSize()
                            .background(Color.Black.copy(alpha = 0.40f)),
                        contentAlignment  = Alignment.Center,
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(14.dp),
                        ) {
                            CircularProgressIndicator(
                                modifier  = Modifier.size(52.dp),
                                color     = SageGreen,
                                strokeWidth = 4.dp,
                            )
                            Text(
                                text  = "Running ONNX inference…",
                                style = MaterialTheme.typography.bodyMedium,
                                color = Color.White,
                            )
                        }
                    }
                }

                // ── Model loading overlay (first cold start only) ──
                if (isModelLoading) {
                    Box(
                        modifier         = Modifier
                            .fillMaxSize()
                            .background(Color.Black.copy(alpha = 0.55f)),
                        contentAlignment = Alignment.Center,
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(14.dp),
                        ) {
                            CircularProgressIndicator(
                                modifier    = Modifier.size(52.dp),
                                color       = SageGreen,
                                strokeWidth = 4.dp,
                            )
                            Text(
                                text  = "Loading ONNX model…",
                                style = MaterialTheme.typography.bodyMedium,
                                color = Color.White,
                            )
                        }
                    }
                }
            }

            // ── Bottom 30%: Control area ───────────────────────────
            LowerThirdControls(
                modifier        = Modifier
                    .fillMaxWidth()
                    .weight(0.30f),
                onAnalyzeClick  = onAnalyzeClick,
                onGalleryClick  = { galleryLauncher.launch("image/*") },
                analyzeEnabled  = analyzeEnabled,
                isAnalysing     = isAnalysing,
                buttonLabel     = buttonLabel,
            )
        }

        // ── Settings dialog (advanced: edge benchmark lives here) ──
        // Keeps the long bench run out of the demo path while keeping
        // its Q1 evidence one tap away for professor Q&A.
        if (showSettingsDialog) {
            AlertDialog(
                onDismissRequest = { showSettingsDialog = false },
                title   = { Text("Settings") },
                text    = {
                    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text(
                            text  = "Agri-CNX-Edge v1.0-demo · adc_student_full.onnx " +
                                    "(120.7 MB) · fully on-device, no server.",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(
                            text  = "Edge benchmark (advanced): accuracy-once + " +
                                    "latency-repeat passes per provider mode " +
                                    "(~35 min). Run the night before, never live.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        if (benchRunning) {
                            Text(
                                text  = "Benchmarking… see the report sheet for CSV paths.",
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                    }
                },
                confirmButton = {
                    TextButton(onClick = { showSettingsDialog = false }) {
                        Text("Close")
                    }
                },
                dismissButton = {
                    TextButton(
                        onClick  = { onBenchClick() },
                        enabled  = analyzeEnabled && !benchRunning,
                    ) {
                        Text(if (benchRunning) "Benchmarking…" else "Run Phone Bench")
                    }
                },
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Top App Bar
// ─────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AgriTopBar(onSettingsClick: () -> Unit) {
    TopAppBar(
        title = {
            Column {
                Text(
                    text  = "Agri-CNX-Edge",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    text  = "Precision Agriculture · Edge AI",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        },
        actions = {
            IconButton(onClick = onSettingsClick) {
                Icon(
                    imageVector        = Icons.Outlined.Settings,
                    contentDescription = "Settings",
                    tint               = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
        ),
    )
}

// ─────────────────────────────────────────────────────────────────
//  Bottom sheet drag handle
// ─────────────────────────────────────────────────────────────────

@Composable
private fun SheetDragHandle() {
    Box(
        modifier = Modifier
            .padding(top = 12.dp, bottom = 8.dp)
            .size(width = 40.dp, height = 4.dp)
            .clip(CircleShape)
            .background(MaterialTheme.colorScheme.outlineVariant)
    )
}

// ─────────────────────────────────────────────────────────────────
//  Lower-third control area
// ─────────────────────────────────────────────────────────────────

/**
 * @param analyzeEnabled  True only when the ONNX model is loaded, CameraX has
 *                        bound [ImageCapture], AND no inference is in flight.
 * @param isAnalysing     Passed down solely to drive the in-button spinner.
 * @param buttonLabel     Dynamic label passed down from [MainScreen] state.
 * @param onGalleryClick  Opens the system photo picker; the chosen image
 *                        runs the same analyze path (demo/professor use).
 */
@Composable
private fun LowerThirdControls(
    modifier:       Modifier = Modifier,
    onAnalyzeClick: () -> Unit,
    onGalleryClick: () -> Unit,
    analyzeEnabled: Boolean,
    isAnalysing:    Boolean,
    buttonLabel:    String,
) {
    Surface(
        modifier       = modifier,
        color          = MaterialTheme.colorScheme.surface,
        tonalElevation = 2.dp,
    ) {
        Column(
            modifier            = Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.SpaceEvenly,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            // ── Status indicator row ───────────────────────────────
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment     = Alignment.CenterVertically,
            ) {
                StatusChip(
                    label    = if (analyzeEnabled) "Camera Ready" else "Camera Init…",
                    dotColor = if (analyzeEnabled) Color(0xFF4CAF50) else Color(0xFFFFC107),
                )
                StatusChip(label = "GPS Active", dotColor = Color(0xFF4CAF50))
                StatusChip(label = "Edge Mode",  dotColor = Color(0xFF1976D2))
            }

            // ── Primary CTA ────────────────────────────────────────
            // The button shows a small inline CircularProgressIndicator
            // next to the label while inference is running, giving a
            // secondary feedback signal in case the camera overlay is
            // not visible (e.g. sheet is expanded over the viewfinder).
            Button(
                onClick  = onAnalyzeClick,
                enabled  = analyzeEnabled,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape  = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor         = SageGreen,
                    contentColor           = Color.White,
                    disabledContainerColor = SageGreen.copy(alpha = 0.40f),
                    disabledContentColor   = Color.White.copy(alpha = 0.50f),
                ),
                elevation = ButtonDefaults.buttonElevation(
                    defaultElevation = 4.dp,
                    pressedElevation = 8.dp,
                ),
            ) {
                if (isAnalysing) {
                    // Inline spinner replaces the leaf icon while ONNX runs
                    CircularProgressIndicator(
                        modifier    = Modifier.size(20.dp),
                        color       = Color.White,
                        strokeWidth = 2.5.dp,
                    )
                } else {
                    Icon(
                        imageVector        = Icons.Default.Agriculture,
                        contentDescription = null,
                        modifier           = Modifier.size(22.dp),
                    )
                }
                Spacer(modifier = Modifier.width(10.dp))
                Text(
                    text     = buttonLabel,
                    style    = MaterialTheme.typography.labelLarge,
                    fontSize = 16.sp,
                )
            }

            // ── Capture guidance (domain-gap honesty) ────────────
            Text(
                text      = "Fill frame with apple cluster · daylight · hold steady",
                style     = MaterialTheme.typography.labelSmall,
                color     = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )

            // ── Secondary CTA: gallery fallback ────────────────────
            // Same analyze path as capture; for professor demos and
            // offline test photos. The long benchmark moved to Settings.
            OutlinedButton(
                onClick  = onGalleryClick,
                enabled  = analyzeEnabled,
                modifier = Modifier.fillMaxWidth(),
                shape    = RoundedCornerShape(16.dp),
            ) {
                Icon(
                    imageVector        = Icons.Outlined.PhotoLibrary,
                    contentDescription = null,
                    modifier           = Modifier.size(18.dp),
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text  = "Choose from Gallery",
                    style = MaterialTheme.typography.labelLarge,
                )
            }
        }
    }
}

@Composable
private fun StatusChip(label: String, dotColor: Color) {
    Surface(
        shape          = RoundedCornerShape(50),
        color          = MaterialTheme.colorScheme.surfaceVariant,
        tonalElevation = 0.dp,
    ) {
        Row(
            modifier          = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                modifier = Modifier
                    .size(7.dp)
                    .clip(CircleShape)
                    .background(dotColor)
            )
            Spacer(Modifier.width(5.dp))
            Text(
                text  = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Analysis Results Bottom Sheet
// ─────────────────────────────────────────────────────────────────

/**
 * @param results            Either [placeholderResults] or live data built
 *                           from [ModelOutputs] via [buildLiveResults].
 * @param overallConfidence  Badge text e.g. "94% Overall" or "Pending".
 * @param inferenceError     Non-null when the last inference (or model init)
 *                           threw; shown as a warning banner so the user
 *                           knows results may be stale.
 * @param photo              The exact 512 bitmap the model saw (capture or
 *                           gallery, post-rotation fix).
 * @param outputs            Live outputs; boxes are drawn over [photo] so
 *                           detections are visible, not just counted.
 */
@Composable
private fun AnalysisResultsSheet(
    results:           List<AnalysisResult>,
    overallConfidence: String,
    inferenceError:    String?,
    photo:             Bitmap?       = null,
    outputs:           ModelOutputs? = null,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp)
            .padding(bottom = 24.dp),
    ) {
        // ── Header ─────────────────────────────────────────────────
        Row(
            modifier          = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text  = "Analysis Report",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    text  = "Tree Node #A-47 · just now",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Surface(
                shape = RoundedCornerShape(12.dp),
                color = ForestGreen,
            ) {
                Text(
                    text     = overallConfidence,
                    style    = MaterialTheme.typography.labelMedium,
                    color    = Color.White,
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                )
            }
        }

        // ── Detection photo: the exact model input with decoded ──
        // predicted boxes drawn over it (orange, paper convention).
        // Empty state tells the user what to do instead of "0 apples".
        if (photo != null && outputs != null) {
            DetectionPhoto(
                photo   = photo,
                outputs = outputs,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
            )
        }

        // ── Error banner (only visible when inference or init threw) ─
        if (inferenceError != null) {
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                shape = RoundedCornerShape(12.dp),
                color = Color(0xFFFFDAD6),
            ) {
                Row(
                    modifier          = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        imageVector        = Icons.Outlined.Warning,
                        contentDescription = null,
                        tint               = Color(0xFFBA1A1A),
                        modifier           = Modifier.size(18.dp),
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text  = "Model error: $inferenceError",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFFBA1A1A),
                    )
                }
            }
        }

        HorizontalDivider(
            modifier = Modifier.padding(bottom = 16.dp),
            color    = MaterialTheme.colorScheme.outlineVariant,
        )

        // ── 2 × 2 Results Grid ─────────────────────────────────────
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            results.chunked(2).forEach { rowItems ->
                Row(
                    modifier              = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    rowItems.forEach { result ->
                        AnalysisCard(result = result, modifier = Modifier.weight(1f))
                    }
                    if (rowItems.size == 1) Spacer(modifier = Modifier.weight(1f))
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // ── Export footer ──────────────────────────────────────────
        OutlinedButton(
            onClick  = { /* Phase 4: export report */ },
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 8.dp),
            shape  = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = SageGreen),
        ) {
            Icon(
                imageVector        = Icons.Outlined.Share,
                contentDescription = null,
                modifier           = Modifier.size(18.dp),
            )
            Spacer(Modifier.width(8.dp))
            Text("Export Full Report", style = MaterialTheme.typography.labelLarge)
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Detection photo overlay (show-your-work result header)
// ─────────────────────────────────────────────────────────────────

/**
 * Renders the exact 512 bitmap the model saw with decoded apple boxes
 * (normalized cxcywh → pixels, orange to match the paper's predicted
 * convention) plus a count chip. Zero-box state guides the user
 * (move closer / fill frame) instead of silently reporting 0.
 */
@Composable
private fun DetectionPhoto(
    photo:    Bitmap,
    outputs:  ModelOutputs,
    modifier: Modifier = Modifier,
) {
    Box(modifier = modifier.clip(RoundedCornerShape(16.dp))) {
        Image(
            bitmap            = photo.asImageBitmap(),
            contentDescription = null,
            modifier          = Modifier
                .fillMaxWidth()
                .aspectRatio(1f),
        )
        Canvas(modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f)) {
            val boxColor = Color(0xFFE56E00)
            outputs.boxes.forEach { b ->
                val left   = (b.cx - b.w / 2f) * size.width
                val top    = (b.cy - b.h / 2f) * size.height
                val width  = b.w * size.width
                val height = b.h * size.height
                drawRect(
                    color = boxColor,
                    topLeft = Offset(left, top),
                    size = Size(width, height),
                    style = Stroke(width = 5f),
                )
            }
        }
        Surface(
            shape = RoundedCornerShape(10.dp),
            color = Color.Black.copy(alpha = 0.60f),
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(10.dp),
        ) {
            Text(
                text  = if (outputs.boxes.isEmpty())
                    "No apples found — move closer"
                else
                    "${outputs.boxes.size} apples · ${outputs.inferenceMs} ms",
                style = MaterialTheme.typography.labelMedium,
                color = Color.White,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Individual Analysis Result Card
// ─────────────────────────────────────────────────────────────────

@Composable
private fun AnalysisCard(
    result:   AnalysisResult,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier  = modifier,
        shape     = RoundedCornerShape(16.dp),
        colors    = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    ) {
        Column(
            modifier            = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector        = result.icon,
                    contentDescription = null,
                    tint               = SageGreen,
                    modifier           = Modifier.size(20.dp),
                )
                Spacer(Modifier.width(6.dp))
                Text(
                    text     = result.title,
                    style    = MaterialTheme.typography.titleSmall,
                    color    = MaterialTheme.colorScheme.onSurface,
                    maxLines = 1,
                )
            }
            Text(
                text  = result.status,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text      = result.detail,
                style     = MaterialTheme.typography.bodySmall,
                color     = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines  = 3,
                textAlign = TextAlign.Start,
            )
            Surface(
                shape = RoundedCornerShape(8.dp),
                color = result.chipColor.copy(alpha = 0.15f),
            ) {
                Text(
                    text     = result.chipText,
                    style    = MaterialTheme.typography.labelSmall,
                    color    = result.chipColor,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Per-class detail strings  (UI layer only – ML stays in AgriModelRunner)
// ─────────────────────────────────────────────────────────────────

private fun leafDetailFor(cls: Int) = when (cls) {
    0    -> "Apple Mosaic virus pattern detected. Vector (aphid) control advised."
    1    -> "Apple Black Rot lesions present. Remove affected foliage; fungicide advised."
    2    -> "Alternaria leaf spot detected. Monitor spread; treat early."
    else -> "No chlorosis or necrosis detected. Foliage healthy."
}

private fun pestDetailFor(cls: Int) = when (cls) {
    0    -> "Xylotrechus (stem borer) signs. Structural branch damage risk."
    1    -> "Aphid colony activity. Mosaic-virus vector — act early."
    2    -> "Leafhopper activity. Phytoplasma/proliferation risk."
    else -> "Spider mite activity. Bronzing and defoliation risk."
}

private fun fruitDetailFor(cls: Int) = when (cls) {
    0    -> "Anthracnose lesions (Colletotrichum). Remove infected fruit."
    1    -> "Black Rot spreading outward. Segregate affected apples."
    else -> "No disease detected. Fruit healthy."
}

private fun pestActionFor(cls: Int) = when (cls) {
    0    -> "Inspect Wood"
    1    -> "Act Early"
    2    -> "Monitor Weekly"
    else -> "Treat Mites"
}

// ─────────────────────────────────────────────────────────────────
//  Per-class chip colors
// ─────────────────────────────────────────────────────────────────

private fun leafChipColor(cls: Int) = when (cls) {
    0    -> Color(0xFFFFC107)
    1    -> Color(0xFFD32F2F)
    2    -> Color(0xFFD32F2F)
    else -> Color(0xFF4CAF50)
}

private fun pestChipColor(cls: Int) = when (cls) {
    0    -> Color(0xFFD32F2F)
    1    -> Color(0xFFFFC107)
    2    -> Color(0xFFFFC107)
    else -> Color(0xFFD32F2F)
}

private fun fruitChipColor(cls: Int) = when (cls) {
    0    -> Color(0xFFD32F2F)
    1    -> Color(0xFFD32F2F)
    else -> Color(0xFF4CAF50)
}