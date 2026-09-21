package com.aarya.agricnxedge.ml

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.BatteryManager
import android.os.Debug
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.sqrt

// ─────────────────────────────────────────────────────────────────
//  Phone benchmark harness (Phase 4b)
//  Method: warmup + timed runs, batch 1, NanoTime; CSV log per image;
//  battery % before/after (BatteryManager); PSS before/after;
//  provider engaged is recorded from the runner mode.
// ─────────────────────────────────────────────────────────────────

private const val BENCH_TAG = "AgriCNX-Bench"
private const val BENCH_DIR = "bench"
private const val BENCH_SIZE = 512

/** One timed inference row (also written to CSV). */
data class BenchRow(
    val image:      String,
    val gtCount:    Int,
    val leaf:       Int,
    val pest:       Int,
    val predCount:  Int,
    val nFruit:     Int,
    val msTotal:    Long,
    val provider:   String,
)

/** Aggregate report for one provider mode (also saved as JSON-ish txt). */
data class BenchSummary(
    val provider:     String,
    val deviceModel:  String,
    val nImages:      Int,
    val warmupRuns:   Int,
    val measuredRuns: Int,
    val meanMs:       Double,
    val stdMs:        Double,
    val p50Ms:        Double,
    val p95Ms:        Double,
    val countMae:     Double,
    val batteryBefore: Int,
    val batteryAfter:  Int,
    val pssBeforeKb:  Int,
    val pssAfterKb:   Int,
    val modelFileMb:  Double,
    val csvPath:      String,
)

/**
 * Runs the full on-device benchmark for one [ProviderMode].
 *
 * Test images + ground-truth counts come from `assets/bench/`
 * (manifest.csv + *.jpg bundled at build time).
 *
 * @return summary with statistics; per-image CSV at
 *         `<filesDir>/bench_<mode>_<epoch>.csv`.
 */
suspend fun runPhoneBench(
    context: Context,
    mode: ProviderMode,
    warmupRuns: Int = 100,
    measuredTotal: Int = 1000,
): BenchSummary = withContext(Dispatchers.Default) {
    val appCtx = context.applicationContext
    val bm = appCtx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    val am = appCtx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

    // ── Load manifest + bitmaps ──────────────────────────────────
    // Manifest stores full asset filenames (e.g. apple_01775.jpg);
    // a missing-extension fallback covers hand-written manifests.
    fun openAsset(name: String) = try {
        appCtx.assets.open("$BENCH_DIR/$name")
    } catch (e: Exception) {
        appCtx.assets.open("$BENCH_DIR/$name.jpg")
    }
    val gt = LinkedHashMap<String, Int>()
    openAsset("manifest.csv").bufferedReader().useLines { lines ->
        lines.drop(1).forEach { line ->
            val parts = line.trim().split(",")
            if (parts.size >= 2) gt[parts[0]] = parts[1].toIntOrNull() ?: -1
        }
    }
    require(gt.isNotEmpty()) { "assets/bench/manifest.csv is empty or missing." }
    val bitmaps: List<Pair<String, Bitmap>> = gt.keys.map { name ->
        val raw = BitmapFactory.decodeStream(openAsset(name))
            ?: throw IllegalStateException("Cannot decode asset bench/$name")
        val scaled = if (raw.width == BENCH_SIZE && raw.height == BENCH_SIZE) raw
        else Bitmap.createScaledBitmap(raw, BENCH_SIZE, BENCH_SIZE, true)
            .also { if (it != raw) raw.recycle() }
        name to scaled
    }
    // Spread measuredTotal timed inferences evenly across images.
    val reps = maxOf(1, (measuredTotal + bitmaps.size - 1) / bitmaps.size)
    Log.i(BENCH_TAG, "Loaded ${bitmaps.size} bench images, reps=$reps " +
            "(~${reps * bitmaps.size} timed inferences) for mode=$mode.")

    // ── Build runner (own session per mode) ──────────────────────
    val runner = AgriModelRunner(appCtx, mode)
    try {
        fun pssKb(): Int {
            val pid = android.os.Process.myPid()
            val info = am.getProcessMemoryInfo(intArrayOf(pid))
            return if (info.isNotEmpty()) info[0].totalPss else -1
        }
        fun battPct(): Int =
            bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)

        val modelFile = File(appCtx.cacheDir, "adc_student_full.onnx")
        val modelMb = if (modelFile.exists()) modelFile.length() / 1048576.0 else -1.0

        // ── Warmup (untimed) ─────────────────────────────────────
        repeat(warmupRuns) { i ->
            runner.analyzeTree(bitmaps[i % bitmaps.size].second)
        }
        Debug.MemoryInfo() // touch class loader before measuring (avoid first-use skew)

        // ── Measured runs ────────────────────────────────────────
        val battBefore = battPct()
        val pssBefore  = pssKb()
        val rows = ArrayList<BenchRow>()
        val times = ArrayList<Long>()
        var countAbsErr = 0.0
        var countN = 0
        var rep = 0
        repeat(reps) {
            rep++
            for ((name, bm2) in bitmaps) {
                val (out, ms) = runner.analyzeTreeTimed(bm2)
                times.add(ms)
                val g = gt[name] ?: -1
                if (g >= 0) { countAbsErr += kotlin.math.abs(out.yieldCount - g); countN++ }
                rows.add(BenchRow(name, g, out.leafClass, out.pestClass,
                    out.yieldCount, out.fruitPreds.size, ms, mode.name))
            }
        }
        val pssAfter  = pssKb()
        val battAfter = battPct()

        // ── Statistics ───────────────────────────────────────────
        val arr = times.sorted()
        fun pct(p: Double): Double =
            if (arr.isEmpty()) 0.0
            else arr[((arr.size - 1) * p).toInt()].toDouble()
        val mean = if (arr.isEmpty()) 0.0 else arr.average()
        val variance = if (arr.isEmpty()) 0.0
        else arr.map { (it - mean) * (it - mean) }.average()

        // ── CSV log ──────────────────────────────────────────────
        val ts = System.currentTimeMillis()
        val csv = File(appCtx.filesDir, "bench_${mode.name.lowercase()}_$ts.csv")
        csv.bufferedWriter().use { w ->
            w.write("image,gt_count,leaf,pest,pred_count,n_fruit_boxes,ms_total,provider\n")
            rows.forEach { r ->
                w.write("${r.image},${r.gtCount},${r.leaf},${r.pest}," +
                        "${r.predCount},${r.nFruit},${r.msTotal},${r.provider}\n")
            }
        }
        val summary = BenchSummary(
            provider     = mode.name,
            deviceModel  = android.os.Build.MODEL ?: "unknown",
            nImages      = bitmaps.size,
            warmupRuns   = warmupRuns,
            measuredRuns = times.size,
            meanMs       = mean,
            stdMs        = sqrt(variance),
            p50Ms        = pct(0.50),
            p95Ms        = pct(0.95),
            countMae     = if (countN > 0) countAbsErr / countN else -1.0,
            batteryBefore = battBefore,
            batteryAfter  = battAfter,
            pssBeforeKb  = pssBefore,
            pssAfterKb   = pssAfter,
            modelFileMb  = modelMb,
            csvPath      = csv.absolutePath,
        )
        File(appCtx.filesDir, "bench_${mode.name.lowercase()}_$ts.txt")
            .writeText(summary.toString())
        Log.i(BENCH_TAG, "DONE mode=$mode mean=${"%.1f".format(mean)}ms " +
                "p95=${"%.1f".format(pct(0.95))}ms mae=${"%.3f".format(summary.countMae)} " +
                "batt=$battBefore->$battAfter pss=$pssBefore->$pssAfter KB " +
                "csv=${csv.absolutePath}")
        summary
    } finally {
        bitmaps.forEach { (_, b) -> if (!b.isRecycled) b.recycle() }
        runCatching { runner.close() }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Bench v2: uniform accuracy-once pass over bench_v2/manifest_v2.csv
//  Same APK/model/code on every phone. Each image runs ONCE per mode.
//  manifest columns: image,task,gt_count,gt_class,split
// ─────────────────────────────────────────────────────────────────

private const val BENCH_V2_DIR = "bench_v2"

/** One accuracy row (also written to CSV). */
data class BenchAccRow(
    val image: String,
    val task: String,
    val gtCount: Int,
    val gtClass: Int,
    val leaf: Int,
    val pest: Int,
    val predCount: Int,
    val nFruit: Int,
    val fruitClass: Int,
    val msTotal: Long,
    val provider: String,
)

suspend fun runAccuracyV2(
    context: Context,
    mode: ProviderMode,
): BenchSummary = withContext(Dispatchers.Default) {
    val appCtx = context.applicationContext
    val bm = appCtx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    val am = appCtx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

    fun openV2(name: String) = appCtx.assets.open("$BENCH_V2_DIR/$name")
    data class Spec(val image: String, val task: String, val gtCount: Int, val gtClass: Int)
    val specs = ArrayList<Spec>()
    openV2("manifest_v2.csv").bufferedReader().useLines { lines ->
        lines.drop(1).forEach { line ->
            val p = line.trim().split(",")
            if (p.size >= 5) specs.add(Spec(p[0], p[1], p[2].toIntOrNull() ?: -1, p[3].toIntOrNull() ?: -1))
        }
    }
    require(specs.isNotEmpty()) { "assets/bench_v2/manifest_v2.csv is empty or missing." }
    val bitmaps: List<Triple<Spec, String, Bitmap>> = specs.map { s ->
        val raw = BitmapFactory.decodeStream(openV2(s.image))
            ?: throw IllegalStateException("Cannot decode asset bench_v2/${s.image}")
        val scaled = if (raw.width == BENCH_SIZE && raw.height == BENCH_SIZE) raw
        else Bitmap.createScaledBitmap(raw, BENCH_SIZE, BENCH_SIZE, true)
            .also { if (it != raw) raw.recycle() }
        Triple(s, s.image, scaled)
    }
    Log.i(BENCH_TAG, "Loaded ${bitmaps.size} v2 accuracy images for mode=$mode.")

    val runner = AgriModelRunner(appCtx, mode)
    try {
        fun pssKb(): Int {
            val pid = android.os.Process.myPid()
            val info = am.getProcessMemoryInfo(intArrayOf(pid))
            return if (info.isNotEmpty()) info[0].totalPss else -1
        }
        fun battPct(): Int = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        val modelFile = File(appCtx.cacheDir, "adc_student_full.onnx")
        val modelMb = if (modelFile.exists()) modelFile.length() / 1048576.0 else -1.0

        val battBefore = battPct()
        val pssBefore = pssKb()
        val rows = ArrayList<BenchAccRow>()
        val times = ArrayList<Long>()
        var yieldAbsErr = 0.0; var yieldN = 0
        var pestCorrect = 0; var pestN = 0
        for ((spec, name, bmp) in bitmaps) {
            val (out, ms) = runner.analyzeTreeTimed(bmp)
            times.add(ms)
            if (spec.task == "yield" && spec.gtCount >= 0) {
                yieldAbsErr += kotlin.math.abs(out.yieldCount - spec.gtCount); yieldN++
            }
            if (spec.task == "pest" && spec.gtClass >= 0) {
                pestN++; if (out.pestClass == spec.gtClass) pestCorrect++
            }
            rows.add(BenchAccRow(name, spec.task, spec.gtCount, spec.gtClass,
                out.leafClass, out.pestClass, out.yieldCount, out.fruitPreds.size,
                out.fruitClass, ms, mode.name))
        }
        val pssAfter = pssKb()
        val battAfter = battPct()
        val arr = times.sorted()
        fun pct(p: Double): Double =
            if (arr.isEmpty()) 0.0 else arr[((arr.size - 1) * p).toInt()].toDouble()
        val mean = if (arr.isEmpty()) 0.0 else arr.average()
        val variance = if (arr.isEmpty()) 0.0
        else arr.map { (it - mean) * (it - mean) }.average()
        val ts = System.currentTimeMillis()
        val csv = File(appCtx.filesDir, "bench_accuracy_${mode.name.lowercase()}_$ts.csv")
        csv.bufferedWriter().use { w ->
            w.write("image,task,gt_count,gt_class,leaf,pest,pred_count,n_fruit_boxes,fruit_class,ms_total,provider\n")
            rows.forEach { r ->
                w.write("${r.image},${r.task},${r.gtCount},${r.gtClass},${r.leaf},${r.pest}," +
                        "${r.predCount},${r.nFruit},${r.fruitClass},${r.msTotal},${r.provider}\n")
            }
        }
        val summary = BenchSummary(
            provider = mode.name,
            deviceModel = android.os.Build.MODEL ?: "unknown",
            nImages = bitmaps.size,
            warmupRuns = 0,
            measuredRuns = times.size,
            meanMs = mean,
            stdMs = sqrt(variance),
            p50Ms = pct(0.50),
            p95Ms = pct(0.95),
            countMae = if (yieldN > 0) yieldAbsErr / yieldN else -1.0,
            batteryBefore = battBefore,
            batteryAfter = battAfter,
            pssBeforeKb = pssBefore,
            pssAfterKb = pssAfter,
            modelFileMb = modelMb,
            csvPath = csv.absolutePath,
        )
        File(appCtx.filesDir, "bench_accuracy_${mode.name.lowercase()}_$ts.txt")
            .writeText(summary.toString() + " pestAcc=" +
                    (if (pestN > 0) pestCorrect.toDouble() / pestN else -1.0) +
                    " ($pestCorrect/$pestN)")
        Log.i(BENCH_TAG, "DONE accuracy mode=$mode mean=${"%.1f".format(mean)}ms " +
                "yieldMae=${"%.3f".format(summary.countMae)} " +
                "pestAcc=$pestCorrect/$pestN csv=${csv.absolutePath}")
        summary
    } finally {
        bitmaps.forEach { (_, _, b) -> if (!b.isRecycled) b.recycle() }
        runCatching { runner.close() }
    }
}
