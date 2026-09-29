// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import org.json.JSONObject
import org.opencv.calib3d.Calib3d
import org.opencv.calib3d.StereoSGBM
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

private const val TAG = "STEREO"

/** Matcher parameters, kept explicit so they can be varied experimentally. numDisparities is at full resolution. */
class StereoParams(
    var scale: Double = 0.5,
    var numDisparities: Int = 128,
    var blockSize: Int = 5,
    var uniquenessRatio: Int = 10,
    var speckleWindow: Int = 100,
    var speckleRange: Int = 2,
    var disp12MaxDiff: Int = 1,
    var preFilterCap: Int = 31
)

class StereoOutput(
    val rectLeft: Mat,              // full-resolution rectified BGR
    val rectRight: Mat,
    val rectPairLines: Mat,         // side-by-side, downscaled, horizontal guide lines drawn
    val disparityColor: Mat,        // at matcher scale
    val depthColor: Mat,
    val disparity16: Mat,           // CV_16S, fixed point x16, at matcher scale
    val focalPx: Double,            // at matcher scale
    val baselineM: Double,
    val swapped: Boolean,           // true if the second capture became the left image
    val diagnostics: JSONObject
) {
    /** Estimated depth in metres at a point in matcher-scale coordinates (median of a 5x5 neighbourhood), or null if invalid. */
    fun depthAt(x: Int, y: Int): Double? {
        val vals = ArrayList<Double>()
        val buf = ShortArray(1)
        for (dy in -2..2) for (dx in -2..2) {
            val xx = x + dx
            val yy = y + dy
            if (xx < 0 || yy < 0 || xx >= disparity16.cols() || yy >= disparity16.rows()) continue
            disparity16.get(yy, xx, buf)
            val d = buf[0] / 16.0
            if (d > 0.5) vals.add(d)
        }
        if (vals.size < 5) return null
        val d = FrameSynchronizer.median(vals)
        return focalPx * baselineM / d
    }

    fun release() {
        rectLeft.release(); rectRight.release(); rectPairLines.release()
        disparityColor.release(); depthColor.release(); disparity16.release()
    }
}

/**
 * StereoPipeline: everything after acquisition. It never asks how the pair was obtained;
 * it only needs two images, intrinsics, a relative pose and a baseline.
 */
class StereoPipeline(private val params: StereoParams = StereoParams()) {

    fun process(res: StereoAcquisitionResult): StereoOutput {
        val t0 = System.nanoTime()
        val w = res.first.cols()
        val h = res.first.rows()
        val kMat = res.intrinsics.kMat()
        val dist = Mat.zeros(5, 1, CvType.CV_64F)

        var left = res.first
        var right = res.second
        var r = res.relativeRotation.copyOf()
        var t = doubleArrayOf(
            res.translationUnit[0] * res.baselineMeters,
            res.translationUnit[1] * res.baselineMeters,
            res.translationUnit[2] * res.baselineMeters
        )

        // Baseline direction in first-camera coordinates: c = -R^T T.
        val rt = Mat3.mulVec(Mat3.transpose(r), t)
        val c = doubleArrayOf(-rt[0], -rt[1], -rt[2])
        val cn = Mat3.norm(c)
        if (cn < 1e-9) throw IllegalStateException("Zero baseline: no usable translation between the two views.")
        if (abs(c[0]) < 0.5 * cn) {
            throw IllegalStateException("Baseline is not mostly horizontal (x share ${"%.0f".format(100 * abs(c[0]) / cn)}%). Capture again by sliding sideways.")
        }
        var swapped = false
        if (c[0] < 0) {
            // Second camera lies to the LEFT of the first: swap roles so the left image is the left camera.
            swapped = true
            left = res.second
            right = res.first
            val rT = Mat3.transpose(r)
            val tNew = Mat3.mulVec(rT, t)
            t = doubleArrayOf(-tNew[0], -tNew[1], -tNew[2])
            r = rT
        }

        val rM = matFrom(r, 3, 3)
        val tM = matFrom(t, 3, 1)
        val r1 = Mat(); val r2 = Mat(); val p1 = Mat(); val p2 = Mat(); val q = Mat()
        Calib3d.stereoRectify(kMat, dist, kMat, dist, Size(w.toDouble(), h.toDouble()), rM, tM, r1, r2, p1, p2, q, Calib3d.CALIB_ZERO_DISPARITY, 0.0)

        val m1 = Mat(); val m2 = Mat()
        val rectL = Mat(); val rectR = Mat()
        Calib3d.initUndistortRectifyMap(kMat, dist, r1, p1, Size(w.toDouble(), h.toDouble()), CvType.CV_32FC1, m1, m2)
        Imgproc.remap(left, rectL, m1, m2, Imgproc.INTER_LINEAR)
        Calib3d.initUndistortRectifyMap(kMat, dist, r2, p2, Size(w.toDouble(), h.toDouble()), CvType.CV_32FC1, m1, m2)
        Imgproc.remap(right, rectR, m1, m2, Imgproc.INTER_LINEAR)
        val tRect = System.nanoTime()

        // Downscaled working copies for matching and visualisation.
        val s = params.scale
        val sw = (w * s).toInt()
        val sh = (h * s).toInt()
        val smallL = Mat(); val smallR = Mat(); val gl = Mat(); val gr = Mat()
        Imgproc.resize(rectL, smallL, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.resize(rectR, smallR, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.cvtColor(smallL, gl, Imgproc.COLOR_BGR2GRAY)
        Imgproc.cvtColor(smallR, gr, Imgproc.COLOR_BGR2GRAY)

        val epiErr = epipolarError(gl, gr)

        // Disparity
        val bs = if (params.blockSize % 2 == 1) params.blockSize else params.blockSize + 1
        val nd = max(16, ((params.numDisparities * s / 16.0).toInt() + 1) * 16)
        val sgbm = StereoSGBM.create(
            0, nd, bs, 8 * bs * bs, 32 * bs * bs, params.disp12MaxDiff, params.preFilterCap,
            params.uniquenessRatio, params.speckleWindow, params.speckleRange, StereoSGBM.MODE_SGBM_3WAY
        )
        val disp16 = Mat()
        sgbm.compute(gl, gr, disp16)
        val tDisp = System.nanoTime()

        val disp32 = Mat()
        disp16.convertTo(disp32, CvType.CV_32F, 1.0 / 16.0)
        val invalid = Mat()
        Core.compare(disp32, Scalar(0.5), invalid, Core.CMP_LE)
        val valid = Mat()
        Core.bitwise_not(invalid, valid)
        val validCount = Core.countNonZero(valid)
        val validFraction = validCount.toDouble() / (sw.toDouble() * sh)

        val focal = p1.get(0, 0)[0] * s
        val baseline = Mat3.norm(t)
        val mm = Core.minMaxLoc(disp32, valid)
        val dMin = mm.minVal
        val dMax = max(mm.maxVal, dMin + 1e-3)

        val disp8 = Mat()
        disp32.convertTo(disp8, CvType.CV_8U, 255.0 / (dMax - dMin), -dMin * 255.0 / (dMax - dMin))
        val dispColor = Mat()
        Imgproc.applyColorMap(disp8, dispColor, Imgproc.COLORMAP_TURBO)
        dispColor.setTo(Scalar(0.0, 0.0, 0.0), invalid)

        // Depth Z = f B / d, visualised near = warm, far = cool, invalid black.
        val depth = Mat()
        Core.divide(focal * baseline, disp32, depth)
        depth.setTo(Scalar(0.0), invalid)
        val zMin = 0.3
        val zMax = 10.0
        val clamped = Mat()
        Core.min(depth, Scalar(zMax), clamped)
        Core.max(clamped, Scalar(zMin), clamped)
        val depth8 = Mat()
        clamped.convertTo(depth8, CvType.CV_8U, -255.0 / (zMax - zMin), 255.0 * zMax / (zMax - zMin))
        val depthColor = Mat()
        Imgproc.applyColorMap(depth8, depthColor, Imgproc.COLORMAP_TURBO)
        depthColor.setTo(Scalar(0.0, 0.0, 0.0), invalid)

        // Rectified pair with horizontal guide lines: corresponding features should sit on the same line.
        val pair = Mat()
        Core.hconcat(listOf(smallL, smallR), pair)
        var y = 0
        while (y < pair.rows()) {
            Imgproc.line(pair, Point(0.0, y.toDouble()), Point(pair.cols().toDouble(), y.toDouble()), Scalar(0.0, 255.0, 0.0), 1)
            y += 32
        }

        val confidence = (validFraction * exp(-epiErr / 2.0)).coerceIn(0.0, 1.0)
        val tEnd = System.nanoTime()
        val diag = JSONObject()
            .put("acquisitionMethod", res.method.name)
            .put("secondCaptureBecameLeft", swapped)
            .put("baselineM", baseline)
            .put("baselineSource", res.baselineSource)
            .put("focalPxAtMatcherScale", focal)
            .put("matcherScale", s)
            .put("numDisparitiesAtMatcherScale", nd)
            .put("blockSize", bs)
            .put("validDisparityFraction", validFraction)
            .put("disparityMinPx", dMin).put("disparityMaxPx", dMax)
            .put("rectifiedEpipolarErrorMedianPx", epiErr)
            .put("reconstructionConfidence", confidence)
            .put("depthMetricScaleNote", "Metric depth is only as good as the baseline estimate (IMU-derived, confidence ${"%.2f".format(res.baselineConfidence)}). Relative structure is independent of it.")
            .put("timingMs", JSONObject()
                .put("rectify", (tRect - t0) / 1e6)
                .put("disparity", (tDisp - tRect) / 1e6)
                .put("total", (tEnd - t0) / 1e6))
        Diagnostics.log(TAG, "valid=${"%.0f".format(validFraction * 100)}% epipolarErr=${"%.2f".format(epiErr)}px baseline=${"%.1f".format(baseline * 100)}cm swapped=$swapped total=${"%.0f".format((tEnd - t0) / 1e6)}ms")

        for (mat in listOf(kMat, dist, rM, tM, r1, r2, p1, p2, q, m1, m2, smallL, smallR, gl, gr, disp32, invalid, valid, disp8, depth, clamped, depth8)) mat.release()
        return StereoOutput(rectL, rectR, pair, dispColor, depthColor, disp16, focal, baseline, swapped, diag)
    }

    private fun matFrom(a: DoubleArray, rows: Int, cols: Int): Mat {
        val m = Mat(rows, cols, CvType.CV_64F)
        m.put(0, 0, *a)
        return m
    }

    /**
     * Rectification quality: after rectification, corresponding features must share a row.
     * Returns the median |dy| in pixels (at matcher scale) over ratio-test matches; large values mean rectification failed.
     */
    private fun epipolarError(gl: Mat, gr: Mat): Double {
        val orb = ORB.create(1000)
        val kpL = MatOfKeyPoint(); val kpR = MatOfKeyPoint()
        val dL = Mat(); val dR = Mat()
        orb.detectAndCompute(gl, Mat(), kpL, dL)
        orb.detectAndCompute(gr, Mat(), kpR, dR)
        if (dL.empty() || dR.empty()) return 99.0
        val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
        val knn = ArrayList<MatOfDMatch>()
        matcher.knnMatch(dL, dR, knn, 2)
        val a = kpL.toArray()
        val b = kpR.toArray()
        val dys = ArrayList<Double>()
        for (m in knn) {
            val arr = m.toArray()
            if (arr.size >= 2 && arr[0].distance < 0.75f * arr[1].distance) {
                val dy = abs(a[arr[0].queryIdx].pt.y - b[arr[0].trainIdx].pt.y)
                if (dy < 15.0) dys.add(dy)
            }
            m.release()
        }
        kpL.release(); kpR.release(); dL.release(); dR.release()
        return if (dys.size < 10) 99.0 else FrameSynchronizer.median(dys)
    }
}
