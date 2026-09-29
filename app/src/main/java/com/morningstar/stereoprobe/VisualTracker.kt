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

class FrameFeatures(
    val points: Array<Point>,
    val descriptors: Mat,
    val width: Int,
    val height: Int,
    val sharpness: Double,      // variance of the Laplacian
    val brightness: Double      // mean gray level
) {
    fun release() = descriptors.release()
}

/**
 * Evidence extracted by comparing two views.
 * - parallax*Px: median residual after explaining the motion by ROTATION ONLY. Pure pivoting leaves ≈ 0;
 *   genuine viewpoint translation leaves depth-dependent parallax.
 * - rotation / tUnit: relative pose from the essential matrix (translation is up to scale).
 */
class VisualMetrics(
    val matches: Int,
    val inliers: Int,
    val poseInliers: Int,
    val overlapPct: Double,
    val parallaxVisRotPx: Double,
    val parallaxImuRotPx: Double,
    val rotation: DoubleArray?,
    val tUnit: DoubleArray?,
    val imuVisRotDisagreementDeg: Double
) {
    companion object {
        val EMPTY = VisualMetrics(0, 0, 0, 0.0, 0.0, 0.0, null, null, 0.0)
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
        return FrameFeatures(pts, desc, gray.cols(), gray.rows(), s * s, bright)
    }

    /** [k] must be the intrinsics of the images the features were computed on. [imuRotCam] maps camera-A to camera-B coordinates. */
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
        if (n < 15) return VisualMetrics(n, 0, 0, 0.0, 0.0, 0.0, null, null, 0.0)

        val p1 = MatOfPoint2f(); p1.fromList(pa)
        val p2 = MatOfPoint2f(); p2.fromList(pb)
        val kMat = k.kMat()
        val mask = Mat()
        val eAll = Calib3d.findEssentialMat(p1, p2, kMat, Calib3d.RANSAC, 0.999, 1.5, 1000, mask)
        if (eAll.empty() || eAll.rows() < 3) {
            p1.release(); p2.release(); kMat.release(); mask.release(); eAll.release()
            return VisualMetrics(n, 0, 0, 0.0, 0.0, 0.0, null, null, 0.0)
        }
        val e = eAll.rowRange(0, 3).clone()
        val inliers = Core.countNonZero(mask)

        val rMat = Mat()
        val tMat = Mat()
        val poseInliers = Calib3d.recoverPose(e, p1, p2, kMat, rMat, tMat, mask)

        val mk = ByteArray(n)
        mask.get(0, 0, mk)
        val keep = BooleanArray(n) { mk[it].toInt() != 0 }

        val rArr = DoubleArray(9)
        val tArr = DoubleArray(3)
        val poseOk = rMat.rows() == 3 && rMat.cols() == 3 && tMat.total() == 3L
        if (poseOk) {
            rMat.get(0, 0, rArr)
            tMat.get(0, 0, tArr)
        }

        val overlap = coveragePct(pa, keep, a.width, a.height)
        val visRes = if (poseOk) medianResidual(pa, pb, keep, rArr, k) else 0.0
        val imuRes = if (imuRotCam != null) medianResidual(pa, pb, keep, imuRotCam, k) else 0.0
        val disagreement = if (poseOk && imuRotCam != null) Mat3.angleDeg(Mat3.mul(Mat3.transpose(rArr), imuRotCam)) else 0.0

        p1.release(); p2.release(); kMat.release(); mask.release(); eAll.release(); e.release(); rMat.release(); tMat.release()
        return VisualMetrics(n, inliers, poseInliers, overlap, visRes, imuRes, if (poseOk) rArr else null, if (poseOk) tArr else null, disagreement)
    }

    /** Convex-hull area of the surviving matches in image A, as a percentage of the frame: "how much of A is still matched". */
    private fun coveragePct(pa: List<Point>, keep: BooleanArray, w: Int, h: Int): Double {
        val pts = ArrayList<Point>()
        for (i in pa.indices) if (keep[i]) pts.add(pa[i])
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
        return (100.0 * area / (w.toDouble() * h)).coerceIn(0.0, 100.0)
    }

    /** Median distance between observed x2 and the position predicted by rotation-only motion x2 ≈ K R K^-1 x1. */
    private fun medianResidual(pa: List<Point>, pb: List<Point>, keep: BooleanArray, r: DoubleArray, k: Intrinsics): Double {
        val res = ArrayList<Double>()
        for (i in pa.indices) {
            if (!keep[i]) continue
            val x = (pa[i].x - k.cx) / k.fx
            val y = (pa[i].y - k.cy) / k.fy
            val xx = r[0] * x + r[1] * y + r[2]
            val yy = r[3] * x + r[4] * y + r[5]
            val zz = r[6] * x + r[7] * y + r[8]
            if (zz <= 1e-6) continue
            val u = k.fx * xx / zz + k.cx
            val v = k.fy * yy / zz + k.cy
            res.add(hypot(pb[i].x - u, pb[i].y - v))
        }
        return FrameSynchronizer.median(res)
    }
}
