// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.max

/**
 * The classification labels from the directive. They are deliberately NOT collapsed into a
 * generic "dual camera supported". CONCURRENT_NOT_REAR_PAIR is an extra informational label
 * for front+rear combinations Android claims: concurrent, but not a stereo pair.
 */
enum class PairClass(val label: String) {
    TRUE_STEREO_CANDIDATE("TRUE STEREO CANDIDATE"),
    CONCURRENT_UNCERTAIN("CONCURRENT BUT UNCERTAIN"),
    LOGICAL_MULTI_CAMERA_ONLY("LOGICAL MULTI-CAMERA ONLY"),
    UNSUPPORTED("UNSUPPORTED"),
    RESTRICTED("API-RESTRICTED / VENDOR-RESTRICTED"),
    CONCURRENT_NOT_REAR_PAIR("CONCURRENT, NOT A REAR PAIR (not stereo)")
}

/** Baseline: does ONE camera stream in our implementation? Separates "our bug" from "device limit". */
class SingleResult(val id: String) {
    var opened = false
    var streamed = false
    var images = 0
    var captures = 0
    var approxFps = 0.0
    var failure: String? = null

    fun toJson(): JSONObject = JSONObject()
        .put("id", id)
        .put("opened", opened)
        .put("streamed", streamed)
        .put("images", images)
        .put("captures", captures)
        .put("approxFps", Math.round(approxFps * 100.0) / 100.0)
        .put("failure", failure ?: JSONObject.NULL)

    fun summary(): String =
        "camera $id: opened=$opened streamed=$streamed images=$images captures=$captures " +
            "~${"%.1f".format(approxFps)} fps" + (failure?.let { "  FAIL: $it" } ?: "")
}

class PairResult(
    val mode: String,            // DIRECT_OPEN or LOGICAL_PHYSICAL_STREAMS
    val idA: String,
    val idB: String,
    val logicalId: String? = null
) {
    var claimedConcurrent: Boolean? = null
    var concurrentConfigSupported: Boolean? = null
    var concurrentConfigNote: String? = null
    var rearRear = false
    var sharedSensor = false
    var timestampsComparable = false
    var classification: PairClass = PairClass.UNSUPPORTED
    var failStage: String? = null
    var failure: String? = null
    var restricted = false
    var locusHint: String? = null
    var imagesA = 0
    var imagesB = 0
    var capturesA = 0
    var capturesB = 0
    var failedCapturesA = 0
    var failedCapturesB = 0
    var lateEvents: Map<String, String> = emptyMap()
    var statsStart: DeltaStats? = null
    var statsMid: DeltaStats? = null
    val notes = ArrayList<String>()

    fun title(): String =
        if (logicalId == null) "$idA + $idB" else "$idA + $idB (physical streams inside logical $logicalId)"

    fun toJson(): JSONObject {
        val j = JSONObject()
            .put("mode", mode)
            .put("idA", idA)
            .put("idB", idB)
            .put("logicalId", logicalId ?: JSONObject.NULL)
            .put("classification", classification.name)
            .put("classificationLabel", classification.label)
            .put("claimedConcurrentByAndroid", claimedConcurrent ?: JSONObject.NULL)
            .put("isConcurrentSessionConfigurationSupported", concurrentConfigSupported ?: JSONObject.NULL)
            .put("concurrentConfigNote", concurrentConfigNote ?: JSONObject.NULL)
            .put("rearRear", rearRear)
            .put("sharedSensor", sharedSensor)
            .put("timestampsComparable", timestampsComparable)
            .put("failStage", failStage ?: JSONObject.NULL)
            .put("failure", failure ?: JSONObject.NULL)
            .put("restrictionLocusHint", locusHint ?: JSONObject.NULL)
            .put("imagesA", imagesA).put("imagesB", imagesB)
            .put("capturesA", capturesA).put("capturesB", capturesB)
            .put("failedCapturesA", failedCapturesA).put("failedCapturesB", failedCapturesB)
            .put("approxDroppedA", max(0, capturesA - imagesA))
            .put("approxDroppedB", max(0, capturesB - imagesB))
        val late = JSONObject()
        for ((k, v) in lateEvents) late.put(k, v)
        j.put("lateEvents", late)
        j.put("deltaStartOfExposure", statsStart?.toJson() ?: JSONObject.NULL)
        j.put("deltaMidExposure", statsMid?.toJson() ?: JSONObject.NULL)
        val n = JSONArray()
        notes.forEach { n.put(it) }
        j.put("notes", n)
        return j
    }

    fun summary(): String {
        val sb = StringBuilder()
        sb.append("▶ ${title()}\n")
        sb.append("   CLASS: ${classification.label}\n")
        sb.append("   rear+rear=$rearRear  androidClaimsConcurrent=${claimedConcurrent ?: "n/a"}  sharedSensor=$sharedSensor  tsComparable=$timestampsComparable\n")
        concurrentConfigNote?.let { sb.append("   $it\n") }
        if (failure != null) sb.append("   FAIL[$failStage]: $failure\n")
        locusHint?.let { sb.append("   restriction-locus hint (heuristic): $it\n") }
        sb.append("   A: images=$imagesA captures=$capturesA failed=$failedCapturesA ~dropped=${max(0, capturesA - imagesA)}\n")
        sb.append("   B: images=$imagesB captures=$capturesB failed=$failedCapturesB ~dropped=${max(0, capturesB - imagesB)}\n")
        statsStart?.let { sb.append("   Δt (start of exposure): ${it.summary()}\n") }
        statsMid?.let { sb.append("   Δt (mid exposure):      ${it.summary()}\n") }
        if (lateEvents.isNotEmpty()) sb.append("   late camera events: $lateEvents\n")
        notes.forEach { sb.append("   note: $it\n") }
        return sb.toString()
    }
}

class TestRun(
    val singles: List<SingleResult>,
    val pairs: List<PairResult>,
    val hiddenOpen: Map<String, String>
) {
    fun toJson(): JSONObject {
        val s = JSONArray(); singles.forEach { s.put(it.toJson()) }
        val p = JSONArray(); pairs.forEach { p.put(it.toJson()) }
        val h = JSONObject(); for ((k, v) in hiddenOpen) h.put(k, v)
        return JSONObject().put("singleCameraBaselines", s).put("pairTests", p).put("hiddenPhysicalOpenAttempts", h)
    }

    fun summary(): String {
        val sb = StringBuilder("═══ PAIR TEST RESULTS ═══\n")
        sb.append("-- single-camera baselines --\n")
        singles.forEach { sb.append(it.summary()).append('\n') }
        if (hiddenOpen.isNotEmpty()) {
            sb.append("-- direct-open attempts on hidden physical IDs --\n")
            for ((k, v) in hiddenOpen) sb.append("id $k: $v\n")
        }
        sb.append("-- pairs --\n")
        pairs.forEach { sb.append(it.summary()).append('\n') }
        return sb.toString()
    }
}
