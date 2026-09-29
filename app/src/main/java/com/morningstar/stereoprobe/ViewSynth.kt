// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Predicting the second eye from the first eye plus SENSOR metrics only.
 *
 * Physics is exact: given the camera rotation (gyro), the translation of the camera centre, the intrinsics and a depth for
 * every pixel, where that pixel lands in view B is a formula. The ONLY unknown is depth, so the learnable part of the model
 * is a depth prior, expressed as an inverse-depth field over the image:
 *
 *     invZ(u,v) = clamp( w . phi(nx,ny), 0.15, 6 ) / focusDistance
 *     phi = [1, ny, nx, blob, ny^2, nx*ny, blob*ny]      blob = exp(-(nx^2+ny^2) / (2*0.35^2))
 *
 * w[0] is a global gain. NOTE the deliberate confound: disparity = f * baseline * invZ, so baseline error and depth level
 * multiply together and cannot be told apart from one capture. w[0] therefore absorbs sensor baseline bias as well as depth level.
 *
 * Pure Kotlin (no Android / OpenCV) so it can be tested on a plain JVM.
 */
object ViewSynth {

    const val NFEAT = 7
    val FLAT_PRIOR = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    /** Camera intrinsics used for projection. */
    class Cam(val fx: Double, val fy: Double, val cx: Double, val cy: Double, val width: Double, val height: Double)

    /** One completed capture: correspondences A->B (ground truth, revealed AFTER prediction) plus the sensor metrics known BEFORE. */
    class Sample(
        val a: DoubleArray,             // interleaved x,y in A
        val b: DoubleArray,             // interleaved x,y in B (truth)
        val focusM: Double,
        val rot: DoubleArray,           // camera A -> camera B rotation from the gyro (3x3 row-major)
        val c: DoubleArray              // camera-B centre in camera-A coordinates (metres) from the accelerometer
    )

    fun features(nx: Double, ny: Double, out: DoubleArray) {
        val blob = exp(-(nx * nx + ny * ny) / (2 * 0.35 * 0.35))
        out[0] = 1.0; out[1] = ny; out[2] = nx; out[3] = blob
        out[4] = ny * ny; out[5] = nx * ny; out[6] = blob * ny
    }

    fun invZ(w: DoubleArray, focusM: Double, nx: Double, ny: Double, scratch: DoubleArray = DoubleArray(NFEAT)): Double {
        features(nx, ny, scratch)
        var g = 0.0
        for (i in 0 until NFEAT) g += w[i] * scratch[i]
        return min(6.0, max(0.15, g)) / focusM
    }

    /** Where does pixel (u,v) of A land in B, given inverse depth, rotation and camera-B centre c? Null if behind the camera. */
    fun project(u: Double, v: Double, invZ: Double, rot: DoubleArray, c: DoubleArray, cam: Cam, out: DoubleArray): Boolean {
        val rx = (u - cam.cx) / cam.fx
        val ry = (v - cam.cy) / cam.fy
        val z = 1.0 / invZ
        val x = rx * z; val y = ry * z
        // X2 = R X - R c
        val rcx = rot[0] * c[0] + rot[1] * c[1] + rot[2] * c[2]
        val rcy = rot[3] * c[0] + rot[4] * c[1] + rot[5] * c[2]
        val rcz = rot[6] * c[0] + rot[7] * c[1] + rot[8] * c[2]
        val x2 = rot[0] * x + rot[1] * y + rot[2] * z - rcx
        val y2 = rot[3] * x + rot[4] * y + rot[5] * z - rcy
        val z2 = rot[6] * x + rot[7] * y + rot[8] * z - rcz
        if (z2 <= 1e-6) return false
        out[0] = cam.fx * x2 / z2 + cam.cx
        out[1] = cam.fy * y2 / z2 + cam.cy
        return true
    }

    /** Prediction of every A point with the learned prior. */
    fun predictPoints(s: Sample, w: DoubleArray, cam: Cam): DoubleArray {
        val n = s.a.size / 2
        val out = DoubleArray(n * 2)
        val scratch = DoubleArray(NFEAT)
        val p = DoubleArray(2)
        for (i in 0 until n) {
            val u = s.a[2 * i]; val v = s.a[2 * i + 1]
            val nx = (u - cam.width / 2) / (cam.width / 2)
            val ny = (v - cam.height / 2) / (cam.height / 2)
            val ok = project(u, v, invZ(w, s.focusM, nx, ny, scratch), s.rot, s.c, cam, p)
            out[2 * i] = if (ok) p[0] else u
            out[2 * i + 1] = if (ok) p[1] else v
        }
        return out
    }

    /** Baseline predictors that use NO depth: copy A, or apply the gyro rotation alone (infinite-depth homography). */
    fun identityPoints(s: Sample): DoubleArray = s.a.copyOf()

    fun rotationOnlyPoints(s: Sample, cam: Cam): DoubleArray {
        val n = s.a.size / 2
        val out = DoubleArray(n * 2)
        for (i in 0 until n) {
            val rx = (s.a[2 * i] - cam.cx) / cam.fx
            val ry = (s.a[2 * i + 1] - cam.cy) / cam.fy
            val x = s.rot[0] * rx + s.rot[1] * ry + s.rot[2]
            val y = s.rot[3] * rx + s.rot[4] * ry + s.rot[5]
            val z = s.rot[6] * rx + s.rot[7] * ry + s.rot[8]
            out[2 * i] = cam.fx * x / z + cam.cx
            out[2 * i + 1] = cam.fy * y / z + cam.cy
        }
        return out
    }

    fun perPointError(pred: DoubleArray, truth: DoubleArray): DoubleArray {
        val n = pred.size / 2
        return DoubleArray(n) {
            val dx = pred[2 * it] - truth[2 * it]; val dy = pred[2 * it + 1] - truth[2 * it + 1]
            sqrt(dx * dx + dy * dy)
        }
    }

    fun medianError(pred: DoubleArray, truth: DoubleArray): Double = GyroPoseSolver.median(perPointError(pred, truth))

    /**
     * Fit the prior on stored captures by Nelder-Mead on (mean over captures of MEDIAN error) + a pull toward the starting prior.
     * The regulariser keeps a handful of captures from collapsing the model, which is what happened in the first prototype.
     */
    fun fit(samples: List<Sample>, w0: DoubleArray, cam: Cam, lambda: Double = 4.0, start: DoubleArray = w0): DoubleArray {
        if (samples.isEmpty()) return start
        fun cost(w: DoubleArray): Double {
            var tot = 0.0
            for (s in samples) tot += medianError(predictPoints(s, w, cam), s.b)
            var reg = 0.0
            for (i in 0 until NFEAT) reg += (w[i] - w0[i]) * (w[i] - w0[i])
            return tot / samples.size + lambda * reg
        }
        return nelderMead(::cost, start, 0.25, 220)
    }

    fun nelderMead(f: (DoubleArray) -> Double, x0: DoubleArray, step: Double, maxIter: Int): DoubleArray {
        val n = x0.size
        val simplex = Array(n + 1) { i -> x0.copyOf().also { if (i > 0) it[i - 1] += step } }
        val values = DoubleArray(n + 1) { f(simplex[it]) }
        repeat(maxIter) {
            val order = (0..n).sortedBy { values[it] }
            val s2 = Array(n + 1) { simplex[order[it]] }
            val v2 = DoubleArray(n + 1) { values[order[it]] }
            for (i in 0..n) { simplex[i] = s2[i]; values[i] = v2[i] }
            if (abs(values[n] - values[0]) < 1e-5) return simplex[0]
            val cen = DoubleArray(n)
            for (i in 0 until n) for (j in 0 until n) cen[j] += simplex[i][j] / n
            fun pt(t: Double) = DoubleArray(n) { cen[it] + t * (simplex[n][it] - cen[it]) }
            val xr = pt(-1.0); val fr = f(xr)
            if (fr < values[0]) {
                val xe = pt(-2.0); val fe = f(xe)
                if (fe < fr) { simplex[n] = xe; values[n] = fe } else { simplex[n] = xr; values[n] = fr }
            } else if (fr < values[n - 1]) {
                simplex[n] = xr; values[n] = fr
            } else {
                val xc = if (fr < values[n]) pt(-0.5) else pt(0.5)
                val fc = f(xc)
                if (fc < min(fr, values[n])) { simplex[n] = xc; values[n] = fc }
                else for (i in 1..n) {
                    for (j in 0 until n) simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j])
                    values[i] = f(simplex[i])
                }
            }
        }
        var best = 0
        for (i in 1..n) if (values[i] < values[best]) best = i
        return simplex[best]
    }
}
