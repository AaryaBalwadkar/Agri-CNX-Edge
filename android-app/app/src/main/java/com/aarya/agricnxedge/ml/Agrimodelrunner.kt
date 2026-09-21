package com.aarya.agricnxedge.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.EnumSet
import kotlin.math.exp

// ─────────────────────────────────────────────────────────────────
//  Constants  (mirror student_config.py / student_model.py)
// ─────────────────────────────────────────────────────────────────

private const val TAG = "AgriCNX-ONNX"

private const val MODEL_FILE = "adc_student_full.onnx"

// Input tensor dimensions
private const val INPUT_C    = 3
private const val INPUT_H    = 512
private const val INPUT_W    = 512
private const val INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W   // 786,432 floats

// YOLO detection grid (single class, anchor-free, one prediction / cell)
private const val GRID       = 32
private const val CELLS      = GRID * GRID                   // 1024
private const val CONF_THRESH = 0.3f
private const val NMS_IOU     = 0.45f

// ImageNet normalisation constants (torchvision defaults)
private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)   // R, G, B
private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)   // R, G, B

// ─────────────────────────────────────────────────────────────────
//  Label maps  (index order MUST match student_config.py class maps)
// ─────────────────────────────────────────────────────────────────

val LEAF_LABELS  = listOf("Apple_Mosaic", "Apple___Black_rot", "Alternaria", "Healthy")
val PEST_LABELS  = listOf("xylotrechus", "aphids", "leafhoppers", "spider_mite")
val FRUIT_LABELS = listOf("Anthracnose", "Black Rot", "Healthy")

// ─────────────────────────────────────────────────────────────────
//  Output data classes
// ─────────────────────────────────────────────────────────────────

/** One decoded apple box, normalised cxcywh + confidence. */
data class AppleBox(
    val cx:    Float,
    val cy:    Float,
    val w:     Float,
    val h:     Float,
    val score: Float,
)

/**
 * Fully-parsed result of one inference pass (two ONNX calls internally).
 *
 * @param fruitClass  Majority-vote fruit class over detected apples,
 *                    or -1 when no apple was detected.
 * @param inferenceMs Wall-clock milliseconds for both ONNX calls
 *                    (excludes preprocessing).
 */
data class ModelOutputs(
    val leafClass:      Int,
    val pestClass:      Int,
    val leafConfidence: Float,
    val pestConfidence: Float,
    val fruitClass:     Int,
    val fruitConfidence: Float,
    val yieldCount:     Int,
    val boxes:          List<AppleBox>,
    val fruitPreds:     List<Int>,
    val fruitConfs:     List<Float>,
    val inferenceMs:    Long,
)

/** Execution-provider strategy for the ORT session. */
enum class ProviderMode { CPU_ONLY, NNAPI_PREFERRED }

// ─────────────────────────────────────────────────────────────────
//  AgriModelRunner — ONNX Runtime (full adc_student_full.onnx graph)
// ─────────────────────────────────────────────────────────────────

/**
 * On-device inference engine for the Agri-CNX-Edge YOLO multi-task model.
 *
 * ## Two-call flow (single shared ORT session)
 * ```
 * call 1: (image)                -> leaf_logits, pest_logits, raw_det
 * CPU   : decode raw_det (sigmoid, conf 0.3, NMS IoU 0.45) -> boxes
 * call 2: (image, boxes, box_idx) -> fruit_logits  (skipped if no boxes)
 * ```
 *
 * ## Execution providers
 * NNAPI_PREFERRED registers NNAPI with USE_FP16 and falls back per-op to
 * CPU/XNNPACK automatically; CPU_ONLY skips NNAPI entirely (deterministic
 * baseline for benchmarking, and the safe mode on old drivers).
 */
class AgriModelRunner(context: Context, mode: ProviderMode = ProviderMode.NNAPI_PREFERRED) {

    val providerMode: ProviderMode = mode

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()

    private val session: OrtSession = ortEnv.createSession(
        copyAssetToCache(context, MODEL_FILE).absolutePath,
        buildSessionOptions(mode),
    )

    init {
        val ins  = session.inputNames.toList()
        val outs = session.outputNames.toList()

        Log.i(TAG, "OrtSession ready (mode=$mode).")
        Log.i(TAG, "                   Input names : $ins")
        Log.i(TAG, "                   Output names: $outs")

        require(ins == listOf("image", "boxes", "box_idx")) {
            "Model input name mismatch: expected [image, boxes, box_idx], got $ins."
        }
        require(outs == listOf("leaf_logits", "pest_logits", "raw_det", "fruit_logits")) {
            "Model output name mismatch.\n" +
                    "  Expected : [leaf_logits, pest_logits, raw_det, fruit_logits]\n" +
                    "  Got      : $outs"
        }
        Log.i(TAG, "I/O contract validated.  Model: assets/$MODEL_FILE")
    }

    // ─────────────────────────────────────────────────────────────
    //  Session options
    // ─────────────────────────────────────────────────────────────

    private fun buildSessionOptions(mode: ProviderMode): SessionOptions {
        val opts = SessionOptions()
        opts.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT)
        opts.setIntraOpNumThreads(2)
        opts.setInterOpNumThreads(2)

        if (mode == ProviderMode.NNAPI_PREFERRED) {
            try {
                opts.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                Log.i(TAG, "NNAPI EP registered (USE_FP16). Falls back per-op as needed.")
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI unavailable — CPU/XNNPACK will be used. Reason: $e")
            }
        } else {
            Log.i(TAG, "CPU_ONLY mode — NNAPI not registered.")
        }
        return opts
    }

    // ─────────────────────────────────────────────────────────────
    //  Public inference API
    // ─────────────────────────────────────────────────────────────

    /** Runs a complete inference pass; see [analyzeTreeTimed] for timing. */
    suspend fun analyzeTree(bitmap: Bitmap): ModelOutputs =
        analyzeTreeTimed(bitmap).first

    /**
     * Runs a complete inference pass and returns the outputs together with
     * the wall-clock milliseconds spent inside the two ONNX calls
     * (preprocessing excluded). Used by the UI and the benchmark harness.
     */
    suspend fun analyzeTreeTimed(bitmap: Bitmap): Pair<ModelOutputs, Long> =
        withContext(Dispatchers.Default) {
            val floatBuffer = bitmapToChwFloatBuffer(bitmap)
            val inputTensor = OnnxTensor.createTensor(
                ortEnv,
                floatBuffer,
                longArrayOf(1L, INPUT_C.toLong(), INPUT_H.toLong(), INPUT_W.toLong()),
            )

            // Dummy single box — call 1 only needs leaf/pest/raw_det;
            // fruit output of this call is discarded.
            val dummyBoxes = directFloatBuffer(floatArrayOf(0.5f, 0.5f, 0.1f, 0.1f))
            val dummyIdx   = directLongBuffer(longArrayOf(0L))
            val dummyBoxTensor = OnnxTensor.createTensor(ortEnv, dummyBoxes, longArrayOf(1L, 4L))
            val dummyIdxTensor = OnnxTensor.createTensor(ortEnv, dummyIdx, longArrayOf(1L))

            val t0 = System.nanoTime()
            val leafArray: FloatArray
            val pestArray: FloatArray
            val detArray: FloatArray
            try {
                val r1 = session.run(
                    mapOf("image" to inputTensor, "boxes" to dummyBoxTensor, "box_idx" to dummyIdxTensor))
                r1.use { result ->
                    leafArray = (result["leaf_logits"].get() as OnnxTensor).floatBuffer.toFloatArray()
                    pestArray = (result["pest_logits"].get() as OnnxTensor).floatBuffer.toFloatArray()
                    detArray = (result["raw_det"].get() as OnnxTensor).floatBuffer.toFloatArray()
                }
            } finally {
                dummyBoxTensor.close()
                dummyIdxTensor.close()
            }

            val boxes = decodeDetections(detArray)

            val fruitPreds = mutableListOf<Int>()
            val fruitConfs = mutableListOf<Float>()
            if (boxes.isNotEmpty()) {
                val n = boxes.size
                val bf = ByteBuffer.allocateDirect(n * 4 * java.lang.Float.BYTES)
                    .order(ByteOrder.nativeOrder()).asFloatBuffer()
                val bi = ByteBuffer.allocateDirect(n * java.lang.Long.BYTES)
                    .order(ByteOrder.nativeOrder()).asLongBuffer()
                boxes.forEachIndexed { i, b ->
                    bf.put(i * 4 + 0, b.cx); bf.put(i * 4 + 1, b.cy)
                    bf.put(i * 4 + 2, b.w); bf.put(i * 4 + 3, b.h)
                    bi.put(i, 0L)
                }
                bf.rewind(); bi.rewind()
                val boxTensor = OnnxTensor.createTensor(
                    ortEnv, bf, longArrayOf(n.toLong(), 4L))
                val idxTensor = OnnxTensor.createTensor(
                    ortEnv, bi, longArrayOf(n.toLong()))
                try {
                    val r2 = session.run(
                        mapOf("image" to inputTensor, "boxes" to boxTensor, "box_idx" to idxTensor))
                    r2.use { result ->
                        val fl = (result["fruit_logits"].get() as OnnxTensor)
                            .floatBuffer.toFloatArray()
                        for (m in 0 until n) {
                            val row = fl.copyOfRange(m * 3, m * 3 + 3)
                            val probs = softmax(row)
                            val cls = argmax(probs)
                            fruitPreds.add(cls)
                            fruitConfs.add(probs[cls])
                        }
                    }
                } finally {
                    boxTensor.close()
                    idxTensor.close()
                    inputTensor.close()
                }
            } else {
                inputTensor.close()
            }
            val ms = (System.nanoTime() - t0) / 1_000_000L

            val leafProbs = softmax(leafArray)
            val pestProbs = softmax(pestArray)
            val leafClass = argmax(leafProbs)
            val pestClass = argmax(pestProbs)

            // Majority vote over per-apple fruit predictions (-1 if none).
            val fruitClass: Int
            val fruitConf:  Float
            if (fruitPreds.isEmpty()) {
                fruitClass = -1
                fruitConf  = 0f
            } else {
                val votes = fruitPreds.groupingBy { it }.eachCount()
                val best  = votes.maxByOrNull { it.value }!!.key
                fruitClass = best
                var sum = 0f; var cnt = 0
                fruitPreds.forEachIndexed { i, c ->
                    if (c == best) { sum += fruitConfs[i]; cnt++ }
                }
                fruitConf = if (cnt > 0) sum / cnt else 0f
            }

            val fruitBreakdown = if (fruitPreds.isEmpty()) "none" else
                fruitPreds.groupingBy { it }.eachCount().entries.joinToString(", ") {
                    "${FRUIT_LABELS[it.key]}x${it.value}"
                } + " confs=" + fruitConfs.joinToString(",", "[", "]") {
                    "%d".format((it * 100).toInt())
                }
            Log.i(TAG, "Inference OK (${ms}ms) -> " +
                    "${LEAF_LABELS[leafClass]}(${(leafProbs[leafClass] * 100).toInt()}%) | " +
                    "${PEST_LABELS[pestClass]}(${(pestProbs[pestClass] * 100).toInt()}%) | " +
                    "apples=${boxes.size} fruit=${if (fruitClass >= 0) FRUIT_LABELS[fruitClass] else "none"} " +
                    "(${(fruitConf * 100).toInt()}%) votes=[$fruitBreakdown]")

            Pair(ModelOutputs(
                leafClass       = leafClass,
                pestClass       = pestClass,
                leafConfidence  = leafProbs[leafClass],
                pestConfidence  = pestProbs[pestClass],
                fruitClass      = fruitClass,
                fruitConfidence = fruitConf,
                yieldCount      = boxes.size,
                boxes           = boxes,
                fruitPreds      = fruitPreds,
                fruitConfs      = fruitConfs,
                inferenceMs     = ms,
            ), ms)
        }

    /** Release native ORT resources. Must be called when the runner is no longer needed. */
    fun close() {
        runCatching { session.close() }.onFailure { Log.w(TAG, "Session close: $it") }
        runCatching { ortEnv.close()  }.onFailure { Log.w(TAG, "OrtEnv close: $it")  }
        Log.i(TAG, "ONNX Runtime resources released.")
    }

    // ─────────────────────────────────────────────────────────────
    //  YOLO detection decoding (mirrors student_model.decode_detections:
    //  sigmoid, score = obj*cls > 0.3, greedy NMS IoU 0.45)
    // ─────────────────────────────────────────────────────────────

    private fun sigmoid(x: Float): Float = (1.0f / (1.0f + exp(-x.toDouble()).toFloat()))

    private fun decodeDetections(raw: FloatArray): List<AppleBox> {
        val cands = ArrayList<AppleBox>(64)
        var maxScore = 0f
        for (y in 0 until GRID) {
            for (x in 0 until GRID) {
                fun b(c: Int): Float = raw[(c * GRID + y) * GRID + x]
                val cx = sigmoid(b(0)); val cy = sigmoid(b(1))
                val w  = sigmoid(b(2)); val h  = sigmoid(b(3))
                val score = sigmoid(b(4)) * sigmoid(b(5))
                if (score > maxScore) maxScore = score
                if (score > CONF_THRESH) cands.add(AppleBox(cx, cy, w, h, score))
            }
        }
        cands.sortByDescending { it.score }
        val kept = ArrayList<AppleBox>(cands.size)
        for (c in cands) {
            var drop = false
            for (k in kept) {
                if (boxIoU(c, k) > NMS_IOU) { drop = true; break }
            }
            if (!drop) kept.add(c)
        }
        // Diagnostic: distinguishes "model scored low" (max<thresh, live
        // domain-gap) from "decode bug" (max high but kept=0). Bench and
        // live share this path, so logcat shows both.
        Log.d(TAG, "decode: maxScore=%.4f preNMS=%d kept=%d (thresh=%.2f)".format(
            maxScore, cands.size, kept.size, CONF_THRESH))
        return kept
    }

    private fun boxIoU(a: AppleBox, b: AppleBox): Float {
        val ax1 = a.cx - a.w / 2; val ay1 = a.cy - a.h / 2
        val ax2 = a.cx + a.w / 2; val ay2 = a.cy + a.h / 2
        val bx1 = b.cx - b.w / 2; val by1 = b.cy - b.h / 2
        val bx2 = b.cx + b.w / 2; val by2 = b.cy + b.h / 2
        val ix1 = maxOf(ax1, bx1); val iy1 = maxOf(ay1, by1)
        val ix2 = minOf(ax2, bx2); val iy2 = minOf(ay2, by2)
        val inter = maxOf(0f, ix2 - ix1) * maxOf(0f, iy2 - iy1)
        val union = a.w * a.h + b.w * b.h - inter
        return if (union <= 0f) 0f else inter / union
    }

    // ─────────────────────────────────────────────────────────────
    //  Preprocessing: Bitmap → CHW normalised FloatBuffer
    // ─────────────────────────────────────────────────────────────

    private fun bitmapToChwFloatBuffer(src: Bitmap): FloatBuffer {
        // Live path (camera/gallery) can hand us HARDWARE bitmaps or
        // non-512 sizes; bench path always gives 512 software bitmaps.
        // Normalise here so BOTH paths feed identical tensors.
        var bitmap = src
        if (bitmap.config == Bitmap.Config.HARDWARE) {
            bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false)
                ?: throw IllegalStateException("HARDWARE bitmap copy failed")
            Log.w(TAG, "HARDWARE bitmap converted to software for inference.")
        }
        if (bitmap.width != INPUT_W || bitmap.height != INPUT_H) {
            Log.w(TAG, "bitmapToChw: expected ${INPUT_W}x${INPUT_H} " +
                    "got ${bitmap.width}x${bitmap.height} — scaling.")
            val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_W, INPUT_H, true)
            // Only recycle the temp if we allocated a new one AND it isn't the caller's src.
            if (scaled !== bitmap && bitmap !== src) bitmap.recycle()
            bitmap = scaled
        }
        val pixelCount = INPUT_H * INPUT_W
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, INPUT_W, 0, 0, INPUT_W, INPUT_H)
        // If we allocated a temp bitmap above, release it now (never recycle caller's src).
        if (bitmap !== src) bitmap.recycle()
        val buf: FloatBuffer = ByteBuffer
            .allocateDirect(INPUT_SIZE * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        val rBase = 0
        val gBase = pixelCount
        val bBase = 2 * pixelCount
        for (i in 0 until pixelCount) {
            val px = pixels[i]
            val r = ((px shr 16) and 0xFF) * (1.0f / 255.0f)
            val g = ((px shr  8) and 0xFF) * (1.0f / 255.0f)
            val b = ( px         and 0xFF) * (1.0f / 255.0f)
            buf.put(rBase + i, (r - MEAN[0]) / STD[0])
            buf.put(gBase + i, (g - MEAN[1]) / STD[1])
            buf.put(bBase + i, (b - MEAN[2]) / STD[2])
        }
        buf.rewind()
        return buf
    }

    private fun directFloatBuffer(arr: FloatArray): FloatBuffer {
        val buf = ByteBuffer.allocateDirect(arr.size * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        buf.put(arr); buf.rewind()
        return buf
    }

    private fun directLongBuffer(arr: LongArray): LongBuffer {
        val buf = ByteBuffer.allocateDirect(arr.size * java.lang.Long.BYTES)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        buf.put(arr); buf.rewind()
        return buf
    }

    // ─────────────────────────────────────────────────────────────
    //  Postprocessing: numerically-stable softmax + raw argmax
    // ─────────────────────────────────────────────────────────────

    private fun softmax(logits: FloatArray): FloatArray {
        val n = logits.size
        var maxLogit = logits[0]
        for (i in 1 until n) {
            if (logits[i] > maxLogit) maxLogit = logits[i]
        }
        val exps   = FloatArray(n)
        var sumExp = 0.0f
        for (i in 0 until n) {
            val e   = exp((logits[i] - maxLogit).toDouble()).toFloat()
            exps[i] = e
            sumExp += e
        }
        if (sumExp < 1e-9f) sumExp = 1e-9f
        val probs = FloatArray(n)
        for (i in 0 until n) {
            probs[i] = exps[i] / sumExp
        }
        return probs
    }

    private fun argmax(probs: FloatArray): Int {
        var bestIdx = 0
        var bestVal = probs[0]
        for (i in 1 until probs.size) {
            if (probs[i] > bestVal) {
                bestVal = probs[i]
                bestIdx = i
            }
        }
        return bestIdx
    }

    // ─────────────────────────────────────────────────────────────
    //  Asset management
    // ─────────────────────────────────────────────────────────────

    private fun copyAssetToCache(context: Context, assetName: String): File {
        val dest = File(context.cacheDir, assetName)
        // Stale-cache guard: APK updates keep cacheDir, so an old .onnx
        // would otherwise be reused forever. Compare asset length and
        // re-copy when they differ (live and bench share this file).
        var assetLen = -1L
        try {
            context.assets.openFd(assetName).use { fd -> assetLen = fd.length }
        } catch (_: Exception) {
            // Compressed assets have no FD length — fall back to exists check.
        }
        val needsCopy = !dest.exists() || (assetLen > 0 && dest.length() != assetLen)
        if (needsCopy) {
            if (dest.exists() && assetLen > 0) {
                Log.i(TAG, "Model cache stale (cached=${dest.length()} " +
                        "asset=$assetLen) — re-copying '$assetName'…")
                dest.delete()
            } else {
                Log.i(TAG, "First run: copying '$assetName' from APK assets…")
            }
            context.assets.open(assetName).use { src ->
                FileOutputStream(dest).use { dst ->
                    src.copyTo(dst, bufferSize = 4 * 1024 * 1024)
                }
            }
            Log.i(TAG, "Copy complete: ${dest.absolutePath}  (${dest.length() / 1024} KB)")
        } else {
            Log.d(TAG, "Model cache hit: ${dest.absolutePath} (${dest.length() / 1024} KB)")
        }
        return dest
    }
}

// ─────────────────────────────────────────────────────────────────
//  FloatBuffer extension
// ─────────────────────────────────────────────────────────────────

private fun FloatBuffer.toFloatArray(): FloatArray {
    val arr = FloatArray(remaining())
    get(arr)
    return arr
}
