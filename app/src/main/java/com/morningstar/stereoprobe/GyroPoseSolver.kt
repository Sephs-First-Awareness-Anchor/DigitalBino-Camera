// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Relative camera pose when the ROTATION is known from the gyroscope.
 *
 * Why: decomposing the essential matrix (5 unknowns) is unstable when the baseline is small compared with the scene depth.
 * On real Bino Sweep pairs it flipped between "13% sideways" and "78% sideways" from one image scale to the next, and often
 * pointed mostly forward. With rotation fixed by the gyro, the epipolar constraint  t . (R x1 x x2) = 0  is LINEAR in the
 * translation direction t, so t is the smallest eigenvector of a 3x3 matrix: stable and deterministic. A tiny search then
 * nudges the rotation (gyro error is a fraction of a degree) to minimise the Sampson epipolar error.
 *
 * Convention (same as OpenCV):  X_second = R * X_first + t.  Camera-A-frame baseline direction  c = -R^T t.
 * No Android or OpenCV dependency, so it can be unit-tested on a plain JVM.
 */
object GyroPoseSolver {

    class Result(
        val rotation: DoubleArray,      // 3x3 row-major, camera A -> camera B
        val tUnit: DoubleArray,         // unit translation, sign resolved by cheirality
        val sampsonPx: Double,          // median Sampson epipolar error in pixels (at the intrinsics' scale)
        val nudgeDeg: Double,           // rotation correction applied on top of the gyro
        val used: Int                   // correspondences used
    )

    /** [pa]/[pb] are interleaved x,y pixel coordinates of corresponding points in image A / image B. */
    fun solve(
        pa: DoubleArray, pb: DoubleArray,
        fx: Double, fy: Double, cx: Double, cy: Double,
        rGyro: DoubleArray, refine: Boolean = true
    ): Result? {
        val n = pa.size / 2
        if (n < 20 || pb.size != pa.size) return null
        val x1 = DoubleArray(n * 3)
        val x2 = DoubleArray(n * 3)
        for (i in 0 until n) {
            x1[i * 3] = (pa[2 * i] - cx) / fx; x1[i * 3 + 1] = (pa[2 * i + 1] - cy) / fy; x1[i * 3 + 2] = 1.0
            x2[i * 3] = (pb[2 * i] - cx) / fx; x2[i * 3 + 1] = (pb[2 * i + 1] - cy) / fy; x2[i * 3 + 2] = 1.0
        }

        var bestR = rGyro
        var bestT = fitTranslation(x1, x2, n, rGyro) ?: return null
        var bestCost = sampsonMedian(x1, x2, n, bestR, bestT, fx)
        var nudge = doubleArrayOf(0.0, 0.0, 0.0)

        if (refine) {
            var step = 0.5
            for (round in 0 until 6) {
                var improved = false
                for (axis in 0..2) for (sgn in intArrayOf(1, -1)) {
                    val w = nudge.copyOf()
                    w[axis] += sgn * step
                    val rot = Mat3.mul(rodriguesDeg(w), rGyro)
                    val t = fitTranslation(x1, x2, n, rot) ?: continue
                    val c = sampsonMedian(x1, x2, n, rot, t, fx)
                    if (c < bestCost - 1e-4) {
                        bestCost = c; bestR = rot; bestT = t; nudge = w; improved = true
                    }
                }
                if (!improved) step /= 2.0
            }
        }
        return Result(bestR, bestT, bestCost, Mat3.norm(nudge), n)
    }

    /** Sampson error (median, pixels) of a pose over normalised correspondences. */
    fun sampsonMedian(x1: DoubleArray, x2: DoubleArray, n: Int, r: DoubleArray, t: DoubleArray, fx: Double): Double {
        // E = [t]x R
        val tx = doubleArrayOf(0.0, -t[2], t[1], t[2], 0.0, -t[0], -t[1], t[0], 0.0)
        val e = Mat3.mul(tx, r)
        val errs = DoubleArray(n)
        for (i in 0 until n) {
            val a0 = x1[i * 3]; val a1 = x1[i * 3 + 1]; val a2 = x1[i * 3 + 2]
            val b0 = x2[i * 3]; val b1 = x2[i * 3 + 1]; val b2 = x2[i * 3 + 2]
            val ex0 = e[0] * a0 + e[1] * a1 + e[2] * a2
            val ex1 = e[3] * a0 + e[4] * a1 + e[5] * a2
            val ex2 = e[6] * a0 + e[7] * a1 + e[8] * a2
            val etx0 = e[0] * b0 + e[3] * b1 + e[6] * b2
            val etx1 = e[1] * b0 + e[4] * b1 + e[7] * b2
            val num = (b0 * ex0 + b1 * ex1 + b2 * ex2)
            val den = ex0 * ex0 + ex1 * ex1 + etx0 * etx0 + etx1 * etx1
            errs[i] = abs(num) / sqrt(den + 1e-18) * fx
        }
        return median(errs)
    }

    /** Translation direction minimising sum (t . a_i)^2 with robust (Cauchy) reweighting; sign by cheirality. */
    fun fitTranslation(x1: DoubleArray, x2: DoubleArray, n: Int, r: DoubleArray): DoubleArray? {
        val a = DoubleArray(n * 3)
        val r1 = DoubleArray(n * 3)
        for (i in 0 until n) {
            val p0 = x1[i * 3]; val p1 = x1[i * 3 + 1]; val p2 = x1[i * 3 + 2]
            val q0 = r[0] * p0 + r[1] * p1 + r[2] * p2
            val q1 = r[3] * p0 + r[4] * p1 + r[5] * p2
            val q2 = r[6] * p0 + r[7] * p1 + r[8] * p2
            r1[i * 3] = q0; r1[i * 3 + 1] = q1; r1[i * 3 + 2] = q2
            val b0 = x2[i * 3]; val b1 = x2[i * 3 + 1]; val b2 = x2[i * 3 + 2]
            var c0 = q1 * b2 - q2 * b1
            var c1 = q2 * b0 - q0 * b2
            var c2 = q0 * b1 - q1 * b0
            val nn = sqrt(c0 * c0 + c1 * c1 + c2 * c2) + 1e-12
            c0 /= nn; c1 /= nn; c2 /= nn
            a[i * 3] = c0; a[i * 3 + 1] = c1; a[i * 3 + 2] = c2
        }
        val w = DoubleArray(n) { 1.0 }
        var t = doubleArrayOf(1.0, 0.0, 0.0)
        for (iter in 0 until 4) {
            val m = DoubleArray(9)
            for (i in 0 until n) {
                val wi = w[i]
                val c0 = a[i * 3]; val c1 = a[i * 3 + 1]; val c2 = a[i * 3 + 2]
                m[0] += wi * c0 * c0; m[1] += wi * c0 * c1; m[2] += wi * c0 * c2
                m[4] += wi * c1 * c1; m[5] += wi * c1 * c2; m[8] += wi * c2 * c2
            }
            m[3] = m[1]; m[6] = m[2]; m[7] = m[5]
            t = smallestEigenvector(m)
            val res = DoubleArray(n) { abs(a[it * 3] * t[0] + a[it * 3 + 1] * t[1] + a[it * 3 + 2] * t[2]) }
            val s = 1.4826 * median(res) + 1e-9
            for (i in 0 until n) { val z = res[i] / (2.5 * s); w[i] = 1.0 / (1.0 + z * z) }
        }
        val plus = frontCount(r1, x2, n, t)
        val minus = frontCount(r1, x2, n, doubleArrayOf(-t[0], -t[1], -t[2]))
        return if (minus > plus) doubleArrayOf(-t[0], -t[1], -t[2]) else t
    }

    /** How many correspondences triangulate in front of both cameras for translation [t]. */
    private fun frontCount(r1: DoubleArray, x2: DoubleArray, n: Int, t: DoubleArray): Int {
        var cnt = 0
        var i = 0
        while (i < n) {
            val a0 = r1[i * 3]; val a1 = r1[i * 3 + 1]; val a2 = r1[i * 3 + 2]
            val b0 = x2[i * 3]; val b1 = x2[i * 3 + 1]; val b2 = x2[i * 3 + 2]
            val aa = a0 * a0 + a1 * a1 + a2 * a2
            val bb = b0 * b0 + b1 * b1 + b2 * b2
            val ab = a0 * b0 + a1 * b1 + a2 * b2
            val at = a0 * t[0] + a1 * t[1] + a2 * t[2]
            val bt = b0 * t[0] + b1 * t[1] + b2 * t[2]
            val det = aa * bb - ab * ab
            if (abs(det) > 1e-12) {
                val l1 = (-at * bb + ab * bt) / det
                val l2 = (aa * bt - ab * at) / det
                if (l1 > 0 && l2 > 0) cnt++
            }
            i += 3
        }
        return cnt
    }

    /** Eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix (cyclic Jacobi). */
    fun smallestEigenvector(sym: DoubleArray): DoubleArray {
        val a = sym.copyOf()
        val v = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        for (sweep in 0 until 40) {
            val off = abs(a[1]) + abs(a[2]) + abs(a[5])
            if (off < 1e-14) break
            for ((p, q) in listOf(Pair(0, 1), Pair(0, 2), Pair(1, 2))) {
                val apq = a[p * 3 + q]
                if (abs(apq) < 1e-18) continue
                val theta = (a[q * 3 + q] - a[p * 3 + p]) / (2.0 * apq)
                val tt = (if (theta >= 0) 1.0 else -1.0) / (abs(theta) + sqrt(theta * theta + 1.0))
                val c = 1.0 / sqrt(tt * tt + 1.0)
                val s = tt * c
                for (k in 0..2) {
                    val akp = a[k * 3 + p]; val akq = a[k * 3 + q]
                    a[k * 3 + p] = c * akp - s * akq
                    a[k * 3 + q] = s * akp + c * akq
                }
                for (k in 0..2) {
                    val apk = a[p * 3 + k]; val aqk = a[q * 3 + k]
                    a[p * 3 + k] = c * apk - s * aqk
                    a[q * 3 + k] = s * apk + c * aqk
                }
                for (k in 0..2) {
                    val vkp = v[k * 3 + p]; val vkq = v[k * 3 + q]
                    v[k * 3 + p] = c * vkp - s * vkq
                    v[k * 3 + q] = s * vkp + c * vkq
                }
            }
        }
        var idx = 0
        var mn = a[0]
        for (k in 1..2) if (a[k * 3 + k] < mn) { mn = a[k * 3 + k]; idx = k }
        val out = doubleArrayOf(v[idx], v[3 + idx], v[6 + idx])
        val nn = sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2])
        return doubleArrayOf(out[0] / nn, out[1] / nn, out[2] / nn)
    }

    /** Rodrigues rotation matrix from a rotation vector given in DEGREES. */
    fun rodriguesDeg(wDeg: DoubleArray): DoubleArray {
        val w = doubleArrayOf(Math.toRadians(wDeg[0]), Math.toRadians(wDeg[1]), Math.toRadians(wDeg[2]))
        val th = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
        if (th < 1e-12) return doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        val k = doubleArrayOf(w[0] / th, w[1] / th, w[2] / th)
        val kx = doubleArrayOf(0.0, -k[2], k[1], k[2], 0.0, -k[0], -k[1], k[0], 0.0)
        val kx2 = Mat3.mul(kx, kx)
        val s = sin(th)
        val c = 1.0 - cos(th)
        val out = DoubleArray(9)
        for (i in 0..8) out[i] = (if (i % 4 == 0) 1.0 else 0.0) + s * kx[i] + c * kx2[i]
        return out
    }

    fun median(xs: DoubleArray): Double {
        if (xs.isEmpty()) return 0.0
        val s = xs.sortedArray()
        val n = s.size
        return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2.0
    }

    /** Camera A -> B rotation from a device rotation (pitch,yaw,roll degrees, right-handed about device X,Y,Z in A's axes). */
    fun cameraRotationFromDeviceDeg(pitch: Double, yaw: Double, roll: Double): DoubleArray {
        val delta = rodriguesDeg(doubleArrayOf(pitch, yaw, roll))          // device B -> device A
        return Mat3.devToCam(Mat3.transpose(delta))
    }

    @Suppress("unused")
    private fun unusedMax(a: Double, b: Double) = max(a, b)
}
