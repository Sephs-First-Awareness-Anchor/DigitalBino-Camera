// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max

class FrameFeatures(
    val points: Array<Point>,
    val descriptors: Mat,
    val width: Int,
    val height: Int,
    val sharpness: Double,      // variance of the Laplacian
    val brightness: Double,     // mean gray level
    val hullArea: Double        // convex-hull area of ALL keypoints: the region where matches are even possible
) {
    fun release() = descriptors.release()
}

/**
 * Evidence extracted by comparing two views.
 *
 * - eInliers: essential-matrix RANSAC inliers, counted BEFORE any depth filtering. This is the geometry-quality number.
 * - poseInliers: recoverPose survivors with far-point rejection disabled (cheirality only).
 * - overlapPct: how much of frame A is still matched in frame B (larger of the essential-inlier hull and the
 *   rotation-consistent-match hull), so it does not collapse when the essential matrix is poorly conditioned.
 * - consistentPct: share of ALL matches whose position agrees (loosely) with rotation-only prediction from the IMU.
 *   Low values mean matches are wrong or the scene changed, not merely that parallax is small.
 * - parallax*Px: median residual after explaining motion by ROTATION ONLY. Pure pivoting leaves about 0.
 */
class VisualMetrics(
    val matches: Int,
    val eInliers: Int,
    val poseInliers: Int,
    val overlapPct: Double,
    val consistentPct: Double,
    val parallaxVisRotPx: Double,
    val parallaxImuRotPx: Double,
    val flowMedianPx: Double,
    val rotation: DoubleArray?,
    val tUnit: DoubleArray?,
    val imuVisRotDisagreementDeg: Double
) {
    companion object {
        val EMPTY = VisualMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, null, null, 0.0)
    }
}

class VisualTracker(nFeatures: Int = 1500) {

    private val orb = ORB.create(nFeatures)
    private val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)

    fun describe(gray: Mat): FrameFeatures {
        val kp = MatOfKeyPoint()
        val desc = Mat()
        orb.detectAndCompute(gray, Mat(), kp, desc)
        val pts = kp.toArray().map { it.pt }.toTypedArray()
        kp.release()

        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val mu = MatOfDouble()
        val sd = MatOfDouble()
        Core.meanStdDev(lap, mu, sd)
        val s = sd.toArray()[0]
        lap.release(); mu.release(); sd.release()

        val bright = Core.mean(gray).`val`[0]
        return FrameFeatures(pts, desc, gray.cols(), gray.rows(), s * s, bright, hullAreaOf(pts.toList()))
    }

    private fun hullAreaOf(pts: List<Point>): Double {
        if (pts.size < 3) return 0.0
        val mp = MatOfPoint()
        mp.fromList(pts)
        val hullIdx = MatOfInt()
        Imgproc.convexHull(mp, hullIdx)
        val idx = hullIdx.toArray()
        val hull = MatOfPoint2f()
        hull.fromList(idx.map { pts[it] })
        val area = Imgproc.contourArea(hull)
        mp.release(); hullIdx.release(); hull.release()
        return area
    }

    /** [k] must be the intrinsics of the images the features came from. [imuRotCam] maps camera-A to camera-B coordinates. */
    fun compare(a: FrameFeatures, b: FrameFeatures, k: Intrinsics, imuRotCam: DoubleArray?): VisualMetrics {
        if (a.descriptors.empty() || b.descriptors.empty()) return VisualMetrics.EMPTY

        val knn = ArrayList<MatOfDMatch>()
        matcher.knnMatch(a.descriptors, b.descriptors, knn, 2)

        val pa = ArrayList<Point>()
        val pb = ArrayList<Point>()
        for (m in knn) {
            val arr = m.toArray()
            if (arr.size >= 2 && arr[0].distance < 0.75f * arr[1].distance) {
                pa.add(a.points[arr[0].queryIdx])
                pb.add(b.points[arr[0].trainIdx])
            }
            m.release()
        }
        val n = pa.size
        if (n < 15) return VisualMetrics(n, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, null, null, 0.0)

        // Rotation-only prediction (IMU if available, otherwise identity) for every match.
        val rPred = imuRotCam ?: doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        val resid = DoubleArray(n) { residual(pa[it], pb[it], rPred, k) }
        val consistThresh = 0.06 * a.width
        val keepC = BooleanArray(n) { resid[it] < consistThresh }
        val consistentPct = 100.0 * keepC.count { it } / n
        val flowMedian = FrameSynchronizer.median(resid.toList())
        val overlapC = coveragePct(pa, keepC, a.hullArea)

        val p1 = MatOfPoint2f(); p1.fromList(pa)
        val p2 = MatOfPoint2f(); p2.fromList(pb)
        val kMat = k.kMat()
        val mask = Mat()
        val eAll = Calib3d.findEssentialMat(p1, p2, kMat, Calib3d.RANSAC, 0.999, 2.0, 1000, mask)
        if (eAll.empty() || eAll.rows() < 3) {
            p1.release(); p2.release(); kMat.release(); mask.release(); eAll.release()
            val par = if (imuRotCam != null) FrameSynchronizer.median(resid.toList()) else 0.0
            return VisualMetrics(n, 0, 0, overlapC, consistentPct, 0.0, par, flowMedian, null, null, 0.0)
        }
        val e = eAll.rowRange(0, 3).clone()

        // Essential-matrix inliers BEFORE recoverPose rewrites the mask.
        val eBytes = ByteArray(n)
        mask.get(0, 0, eBytes)
        val keepE = BooleanArray(n) { eBytes[it].toInt() != 0 }
        val eInliers = keepE.count { it }

        // recoverPose with far-point rejection disabled: the default discards points beyond ~50 baselines,
        // which for a 5 cm slide is everything farther than about 2.5 m.
        val poseMask = mask.clone()
        val tri = Mat()
        val rMat = Mat()
        val tMat = Mat()
        val poseInliers = Calib3d.recoverPose(e, p1, p2, kMat, rMat, tMat, 1.0e6, poseMask, tri)

        val rArr = DoubleArray(9)
        val tArr = DoubleArray(3)
        val poseOk = rMat.rows() == 3 && rMat.cols() == 3 && tMat.total() == 3L
        if (poseOk) {
            rMat.get(0, 0, rArr)
            tMat.get(0, 0, tArr)
        }

        val overlapE = coveragePct(pa, keepE, a.hullArea)
        val visRes = if (poseOk) medianResidual(pa, pb, keepE, rArr, k) else 0.0
        val imuRes = if (imuRotCam != null) FrameSynchronizer.median((0 until n).filter { keepE[it] }.map { resid[it] }) else 0.0
        val disagreement = if (poseOk && imuRotCam != null) Mat3.angleDeg(Mat3.mul(Mat3.transpose(rArr), imuRotCam)) else 0.0

        p1.release(); p2.release(); kMat.release(); mask.release(); poseMask.release(); tri.release()
        eAll.release(); e.release(); rMat.release(); tMat.release()
        return VisualMetrics(
            n, eInliers, poseInliers, max(overlapE, overlapC), consistentPct,
            visRes, imuRes, flowMedian, if (poseOk) rArr else null, if (poseOk) tArr else null, disagreement
        )
    }

    /**
     * Hull area of the selected matches as a percentage of the hull of ALL keypoints in frame A.
     * Normalising by textured area (not the whole frame) means low-texture regions such as plain fabric no longer
     * cap the score at ~45%: 100% means every place that could match, did.
     */
    private fun coveragePct(pa: List<Point>, keep: BooleanArray, allHullArea: Double): Double {
        if (allHullArea <= 1.0) return 0.0
        val pts = ArrayList<Point>()
        for (i in pa.indices) if (keep[i]) pts.add(pa[i])
        return (100.0 * hullAreaOf(pts) / allHullArea).coerceIn(0.0, 100.0)
    }

    /** Distance between observed x2 and the rotation-only prediction x2 ≈ K R K^-1 x1. */
    private fun residual(p1: Point, p2: Point, r: DoubleArray, k: Intrinsics): Double {
        val x = (p1.x - k.cx) / k.fx
        val y = (p1.y - k.cy) / k.fy
        val xx = r[0] * x + r[1] * y + r[2]
        val yy = r[3] * x + r[4] * y + r[5]
        val zz = r[6] * x + r[7] * y + r[8]
        if (zz <= 1e-6) return 1.0e6
        val u = k.fx * xx / zz + k.cx
        val v = k.fy * yy / zz + k.cy
        return hypot(p2.x - u, p2.y - v)
    }

    private fun medianResidual(pa: List<Point>, pb: List<Point>, keep: BooleanArray, r: DoubleArray, k: Intrinsics): Double {
        val res = ArrayList<Double>()
        for (i in pa.indices) if (keep[i]) res.add(residual(pa[i], pb[i], r, k))
        return FrameSynchronizer.median(res)
    }
}
