// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.graphics.ImageFormat
import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap

private const val TAG = "FLIP"

/**
 * FlipTester measures the "tight camera flip" idea: open A, grab a frame, close A, open B, grab a frame, repeat.
 *
 * This is SEQUENTIAL capture. It is never classified as stereo: the two views are taken at different
 * instants. The point of this test is to put a number on how different those instants are.
 */
class FlipResult(val idA: String, val idB: String, val cycles: Int) {
    val firstFrameLatencyAms = ArrayList<Double>()   // openCamera() call → first frame delivered
    val firstFrameLatencyBms = ArrayList<Double>()
    val gapMs = ArrayList<Double>()                  // time between the A frame and the B frame (receipt clock)
    val gapTsMs = ArrayList<Double>()                // same, using sensor timestamps (only if REALTIME on both)
    val cycleMs = ArrayList<Double>()                // full A→B round trip
    val errors = ArrayList<String>()
    var timestampsComparable = false

    fun verdict(): String {
        if (gapMs.isEmpty()) return "NO USABLE CYCLES"
        val med = FrameSynchronizer.median(gapMs)
        val cyc = FrameSynchronizer.median(cycleMs)
        val rate = if (cyc > 0) 1000.0 / cyc else 0.0
        return "SEQUENTIAL FLIP: NOT SIMULTANEOUS, NOT COUNTED AS STEREO. Median gap between the two views ≈ " +
            "${"%.0f".format(med)} ms; about ${"%.2f".format(rate)} A+B pairs per second. " +
            "Usable only for static scenes on a stable support."
    }

    fun toJson(): JSONObject = JSONObject()
        .put("idA", idA).put("idB", idB).put("cycles", cycles)
        .put("timestampsComparable", timestampsComparable)
        .put("firstFrameLatencyA_ms", JSONArray(firstFrameLatencyAms))
        .put("firstFrameLatencyB_ms", JSONArray(firstFrameLatencyBms))
        .put("gapReceiptClock_ms", JSONArray(gapMs))
        .put("gapSensorTimestamps_ms", JSONArray(gapTsMs))
        .put("cycle_ms", JSONArray(cycleMs))
        .put("errors", JSONArray(errors))
        .put("verdict", verdict())

    fun summary(): String {
        val sb = StringBuilder("▶ flip $idA ↔ $idB  ($cycles cycles)\n")
        sb.append("   first-frame latency A: ${FrameSynchronizer.summarize(firstFrameLatencyAms)} ms\n")
        sb.append("   first-frame latency B: ${FrameSynchronizer.summarize(firstFrameLatencyBms)} ms\n")
        sb.append("   gap A→B (receipt clock): ${FrameSynchronizer.summarize(gapMs)} ms\n")
        if (timestampsComparable) sb.append("   gap A→B (sensor timestamps): ${FrameSynchronizer.summarize(gapTsMs)} ms\n")
        sb.append("   full cycle: ${FrameSynchronizer.summarize(cycleMs)} ms\n")
        if (errors.isNotEmpty()) sb.append("   errors: ${errors.take(5)}\n")
        sb.append("   → ${verdict()}\n")
        return sb.toString()
    }
}

class FlipTester(private val coord: CameraCoordinator, private val probe: ProbeResult) {

    private class Grab(val id: String, val latencyMs: Double, val sensorTsNs: Long, val rxNs: Long, val startNs: Long, val endNs: Long)

    fun run(a: String, b: String, cycles: Int = 6): FlipResult {
        val res = FlipResult(a, b, cycles)
        val ia = probe.byId[a]
        val ib = probe.byId[b]
        res.timestampsComparable =
            ia?.timestampSource == android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME &&
                ib?.timestampSource == android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME

        for (i in 1..cycles) {
            val ga = grab(a, res.errors)
            val gb = if (ga != null) grab(b, res.errors) else null
            if (ga == null || gb == null) {
                Diagnostics.log(TAG, "cycle $i incomplete")
                continue
            }
            res.firstFrameLatencyAms.add(ga.latencyMs)
            res.firstFrameLatencyBms.add(gb.latencyMs)
            res.gapMs.add((gb.rxNs - ga.rxNs) / 1e6)
            if (res.timestampsComparable) res.gapTsMs.add((gb.sensorTsNs - ga.sensorTsNs) / 1e6)
            res.cycleMs.add((gb.endNs - ga.startNs) / 1e6)
            Diagnostics.log(TAG, "cycle $i: A first frame ${"%.0f".format(ga.latencyMs)} ms, B first frame ${"%.0f".format(gb.latencyMs)} ms, gap ${"%.0f".format((gb.rxNs - ga.rxNs) / 1e6)} ms")
        }
        return res
    }

    /** Open → first frame → close. Returns null on any failure (failure text goes into [errors]). */
    private fun grab(id: String, errors: MutableList<String>): Grab? {
        val info = probe.byId[id]
        val size = info?.let { coord.pickSize(it, ImageFormat.YUV_420_888) }
        if (size == null) { errors.add("$id: no YUV size"); return null }

        val rec = StreamRecorder(id)
        val rig = Rig()
        val late = ConcurrentHashMap<String, String>()
        val start = SystemClock.elapsedRealtimeNanos()
        try {
            val fail = coord.bringUp(id, size, rig, rec, late)
            if (fail != null) { errors.add("$id [${fail.stage}] ${fail.msg}"); return null }

            val deadline = SystemClock.elapsedRealtime() + 4000
            while (rec.firstRxNs.get() == 0L && SystemClock.elapsedRealtime() < deadline) Thread.sleep(2)
            val rx = rec.firstRxNs.get()
            if (rx == 0L) { errors.add("$id: no frame within 4 s"); return null }
            val ts = rec.imageTimestamps().firstOrNull() ?: 0L
            val end = SystemClock.elapsedRealtimeNanos()
            return Grab(id, (rx - start) / 1e6, ts, rx, start, end)
        } catch (t: Throwable) {
            errors.add("$id: ${Diagnostics.describe(t)}")
            return null
        } finally {
            rig.close()
            Thread.sleep(150)
        }
    }
}
