// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import org.json.JSONObject
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

private fun f2(x: Double): String = String.format(Locale.US, "%.2f", x)
private fun r3(x: Double): Double = if (x.isNaN() || x.isInfinite()) 0.0 else Math.round(x * 1000.0) / 1000.0

/**
 * Δt = |timestampA - timestampB| statistics for nearest-neighbour frame pairing.
 *
 * Reading these numbers: for two REGULAR streams that are NOT synchronised, the nearest-frame
 * offset is roughly uniform on 0..P/2 (P = frame period), so mean ≈ P/4. Truly triggered sensors
 * sit near zero. Timestamps are driver-reported values; this measures, it does not prove hardware sync.
 */
data class DeltaStats(
    val samples: Int,
    val minMs: Double,
    val maxMs: Double,
    val meanMs: Double,
    val medianMs: Double,
    val jitterMs: Double,
    val meanSignedMs: Double,
    val unmatchedA: Int,
    val unmatchedB: Int,
    val periodAms: Double,
    val periodBms: Double,
    val verdict: String
) {
    fun toJson(): JSONObject = JSONObject()
        .put("samples", samples)
        .put("minMs", r3(minMs))
        .put("maxMs", r3(maxMs))
        .put("meanMs", r3(meanMs))
        .put("medianMs", r3(medianMs))
        .put("jitterStdDevMs", r3(jitterMs))
        .put("meanSignedMs_BminusA", r3(meanSignedMs))
        .put("unmatchedA", unmatchedA)
        .put("unmatchedB", unmatchedB)
        .put("periodAms", r3(periodAms))
        .put("periodBms", r3(periodBms))
        .put("verdict", verdict)

    fun summary(): String =
        "n=$samples Δt min=${f2(minMs)} med=${f2(medianMs)} mean=${f2(meanMs)} max=${f2(maxMs)} " +
            "jitter(sd)=${f2(jitterMs)} ms | signed mean(B-A)=${f2(meanSignedMs)} ms | " +
            "unmatched A/B=$unmatchedA/$unmatchedB | period A/B=${f2(periodAms)}/${f2(periodBms)} ms\n     → $verdict"
}

object FrameSynchronizer {

    fun median(xs: List<Double>): Double {
        if (xs.isEmpty()) return 0.0
        val s = xs.sorted()
        val n = s.size
        return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2.0
    }

    fun medianPeriodMs(sortedTs: List<Long>): Double {
        if (sortedTs.size < 2) return 0.0
        val d = ArrayList<Double>()
        for (i in 1 until sortedTs.size) d.add((sortedTs[i] - sortedTs[i - 1]) / 1e6)
        return median(d)
    }

    /** min / median / max text for a list of numbers (used by the flip test). */
    fun summarize(xs: List<Double>): String {
        if (xs.isEmpty()) return "n=0"
        return "n=${xs.size} min=${f2(xs.min())} med=${f2(median(xs))} max=${f2(xs.max())}"
    }

    /** Timestamps in nanoseconds from two streams. Returns null when there is not enough data. */
    fun compute(a: List<Long>, b: List<Long>): DeltaStats? {
        val sa = a.sorted()
        val sb = b.sorted()
        if (sa.size < 2 || sb.size < 2) return null

        val pA = medianPeriodMs(sa)
        val pB = medianPeriodMs(sb)
        val halfPeriodNs = (max(pA, pB) * 1e6 / 2.0).toLong()

        val absMs = ArrayList<Double>()
        val signedMs = ArrayList<Double>()
        val unmatchedA = nearest(sa, sb, halfPeriodNs, absMs, signedMs)
        val unmatchedB = nearest(sb, sa, halfPeriodNs, null, null)
        if (absMs.isEmpty()) return null

        val mean = absMs.average()
        val variance = absMs.fold(0.0) { acc, v -> acc + (v - mean) * (v - mean) } / absMs.size
        val med = median(absMs)
        val mx = absMs.max()
        return DeltaStats(
            samples = absMs.size,
            minMs = absMs.min(),
            maxMs = mx,
            meanMs = mean,
            medianMs = med,
            jitterMs = sqrt(variance),
            meanSignedMs = signedMs.average(),
            unmatchedA = unmatchedA,
            unmatchedB = unmatchedB,
            periodAms = pA,
            periodBms = pB,
            verdict = verdict(absMs.size, med, mx, mean, pA, pB)
        )
    }

    /** For every timestamp in [from], find the nearest in [to]. Returns count with no partner within half a period. */
    private fun nearest(
        from: List<Long>,
        to: List<Long>,
        halfPeriodNs: Long,
        absMs: MutableList<Double>?,
        signedMs: MutableList<Double>?
    ): Int {
        var j = 0
        var unmatched = 0
        for (t in from) {
            while (j + 1 < to.size && abs(to[j + 1] - t) <= abs(to[j] - t)) j++
            val d = to[j] - t
            if (abs(d) > halfPeriodNs) {
                unmatched++
            } else {
                absMs?.add(abs(d) / 1e6)
                signedMs?.add(d / 1e6)
            }
        }
        return unmatched
    }

    private fun verdict(n: Int, med: Double, mx: Double, mean: Double, pA: Double, pB: Double): String {
        if (n < 5) return "INSUFFICIENT DATA"
        val p = max(pA, pB)
        val rateNote =
            if (pA > 0 && pB > 0 && abs(pA - pB) / max(pA, pB) > 0.10)
                " Streams run at different frame periods (A=${f2(pA)} ms, B=${f2(pB)} ms), so they cannot be locked frame-for-frame."
            else ""
        return when {
            med < 1.0 && mx < 3.0 ->
                "TIGHT ALIGNMENT (median <1 ms, max <3 ms): consistent with a shared trigger or clock. " +
                    "These are driver-reported sensor timestamps: evidence, not proof, of hardware sync.$rateNote"
            p > 0 && mean > 0.15 * p && mx > 0.35 * p ->
                "FREE-RUNNING: Δt is spread across the frame period (unsynchronised streams average ≈ P/4 = ${f2(p / 4)} ms). " +
                    "No evidence of hardware synchronisation.$rateNote"
            else ->
                "INTERMEDIATE / UNCLEAR: alignment is neither tight nor uniformly spread; collect a longer sample.$rateNote"
        }
    }
}
