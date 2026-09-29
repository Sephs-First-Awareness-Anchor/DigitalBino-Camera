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
import kotlin.math.min

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
    var preFilterCap: Int = 31,
    /** 0 = crop to the all-valid region (can zoom in dramatically when rectifying rotations are large); 1 = keep every source pixel. */
    var rectifyAlpha: Double = 1.0,
    /** Local contrast normalisation before matching: makes matching robust to brightness/colour changes between the two captures. */
    var photometricNormalization: Boolean = true
)

class StereoOutput(
    val rectLeft: Mat,              // full-resolution rectified BGR
    val rectRight: Mat,
    val rectPairLines: Mat,         // side-by-side, downscaled, horizontal guide lines drawn
    val disparityColor: Mat,        // at matcher scale
    val depthColor: Mat,
    val disparity16: Mat,           // CV_16S, fixed point x16, at matcher scale
    val focalPx: Double,            // at matcher scale
    var baselineM: Double,          // metres when scaleState != UNSCALED, otherwise 1.0 (depth in "baselines")
    val swapped: Boolean,           // true if the second capture became the left image
    private val dLoPx: Double,      // 2nd percentile of valid disparities
    private val dHiPx: Double,      // 98th percentile
    var scaleState: String,         // UNSCALED / IMU / FOCUS / USER
    val diagnostics: JSONObject
) {
    fun unitLabel(): String = if (scaleState == "UNSCALED") "baselines" else "m"

    /** "in view: nearest ≈ … farthest ≈ …" using the CURRENT scale. */
    fun rangeText(): String {
        val near = focalPx * baselineM / dHiPx
        val far = focalPx * baselineM / dLoPx
        return "in view: nearest ≈ ${"%.2f".format(near)}, farthest ≈ ${"%.2f".format(far)} ${unitLabel()}  (scale: $scaleState)"
    }

    /** Apply a user-supplied true distance at a point: depth scales linearly with the baseline. */
    fun anchor(trueDistanceM: Double, x: Int, y: Int): Boolean {
        val z = depthAt(x, y) ?: return false
        if (z <= 0.0 || trueDistanceM <= 0.0) return false
        baselineM *= trueDistanceM / z
        scaleState = "USER"
        return true
    }

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

    private class Rectified(
        val rectL: Mat, val rectR: Mat,
        val smallL: Mat, val smallR: Mat,
        val gl: Mat, val gr: Mat,
        val valid: Mat,
        val focalFull: Double,
        val r1Deg: Double, val r2Deg: Double, val zoom: Double
    ) {
        fun release() {
            rectL.release(); rectR.release(); smallL.release(); smallR.release(); gl.release(); gr.release(); valid.release()
        }
    }

    private class MatchStats(
        val dyMedian: Double, val dxMedian: Double, val alignedCount: Int,
        val dxP2: Double = 0.0, val dxP98: Double = 0.0,
        val centerDx: Double = 0.0, val centerCount: Int = 0
    )

    fun process(res: StereoAcquisitionResult): StereoOutput {
        val t0 = System.nanoTime()
        val w = res.first.cols()
        val h = res.first.rows()
        val kMat = res.intrinsics.kMat()

        var r = res.relativeRotation.copyOf()
        var t = doubleArrayOf(
            res.translationUnit[0] * res.baselineMeters,
            res.translationUnit[1] * res.baselineMeters,
            res.translationUnit[2] * res.baselineMeters
        )

        // Baseline direction in first-camera coordinates: c = -R^T T. Only its ORIENTATION is trusted, not its sign.
        val rt = Mat3.mulVec(Mat3.transpose(r), t)
        val c = doubleArrayOf(-rt[0], -rt[1], -rt[2])
        val cn = Mat3.norm(c)
        if (cn < 1e-9) throw IllegalStateException("Zero baseline: no usable translation between the two views.")
        val horizontalShare = abs(c[0]) / cn
        if (horizontalShare < 0.5) {
            throw IllegalStateException("Baseline is not mostly horizontal (${"%.0f".format(100 * horizontalShare)}% sideways). Capture again by sliding straight sideways.")
        }

        // stereoRectify does not care about the sign of T, so the sign from recoverPose (which can be wrong on noisy data)
        // must not decide which image is "left". Rectify, MEASURE the disparity sign from matched features, swap if negative.
        var left = res.first
        var right = res.second
        var rect = rectify(left, right, r, t, kMat, w, h)
        if (rect.zoom <= 0.15) {
            rect.release()
            throw IllegalStateException("Rectification is degenerate (the camera moved mostly toward or away from the scene, not sideways). Capture again by sliding straight sideways.")
        }
        var stats = matchStats(rect.gl, rect.gr)
        var swapped = false
        var signNote = "left/right order confirmed by measured disparity sign"
        if (stats.alignedCount >= 10 && stats.dxMedian < -1.0) {
            rect.release()
            left = res.second
            right = res.first
            val rT = Mat3.transpose(r)
            val tNew = Mat3.mulVec(rT, t)
            t = doubleArrayOf(-tNew[0], -tNew[1], -tNew[2])
            r = rT
            rect = rectify(left, right, r, t, kMat, w, h)
            stats = matchStats(rect.gl, rect.gr)
            swapped = true
            signNote = "images swapped: measured disparity was negative with the pose sign from recoverPose"
        } else if (stats.alignedCount < 10) {
            signNote = "disparity sign could not be measured (only ${stats.alignedCount} aligned matches)"
        }
        val tRect = System.nanoTime()

        val s = params.scale
        val sw = rect.gl.cols()
        val sh = rect.gl.rows()

        val ml = if (params.photometricNormalization) normalizeLocal(rect.gl) else rect.gl.clone()
        val mr = if (params.photometricNormalization) normalizeLocal(rect.gr) else rect.gr.clone()

        val bs = if (params.blockSize % 2 == 1) params.blockSize else params.blockSize + 1
        // Search range from the disparities actually measured between matched features (fixed 128 saturated on wide baselines).
        val ndDefault = max(16, ((params.numDisparities * s / 16.0).toInt() + 1) * 16)
        val nd = if (stats.alignedCount >= 30 && stats.dxP98 > 0.0) max(32, min(256, (((stats.dxP98 * 1.3 + 24.0) / 16.0).toInt() + 1) * 16)) else ndDefault
        val sgbm = StereoSGBM.create(
            0, nd, bs, 8 * bs * bs, 32 * bs * bs, params.disp12MaxDiff, params.preFilterCap,
            params.uniquenessRatio, params.speckleWindow, params.speckleRange, StereoSGBM.MODE_SGBM_3WAY
        )
        val disp16 = Mat()
        sgbm.compute(ml, mr, disp16)
        val tDisp = System.nanoTime()

        val disp32 = Mat()
        disp16.convertTo(disp32, CvType.CV_32F, 1.0 / 16.0)
        val invalid = Mat()
        Core.compare(disp32, Scalar(0.5), invalid, Core.CMP_LE)
        val outside = Mat()
        Core.bitwise_not(rect.valid, outside)
        Core.bitwise_or(invalid, outside, invalid)          // pixels with no source data are invalid
        val valid = Mat()
        Core.bitwise_not(invalid, valid)
        val validCount = Core.countNonZero(valid)
        val validFraction = validCount.toDouble() / (sw.toDouble() * sh)
        val validAreaFraction = Core.countNonZero(rect.valid).toDouble() / (sw.toDouble() * sh)

        val focal = rect.focalFull * s
        val baselineUnits = Mat3.norm(t)
        val mm = Core.minMaxLoc(disp32, valid)
        val dMin = if (validCount > 0) mm.minVal else 0.0
        val dMax = if (validCount > 0) max(mm.maxVal, dMin + 1e-3) else 1.0

        // Disparity samples at matcher scale: percentiles for colouring, and the centre of frame for the focus anchor.
        val dispAll = FloatArray(sw * sh)
        disp32.get(0, 0, dispAll)
        val validD = ArrayList<Float>()
        for (v in dispAll) if (v > 0.5f) validD.add(v)
        validD.sort()
        val dLo = if (validD.isNotEmpty()) validD[(validD.size * 0.02).toInt()].toDouble() else 1.0
        val dHi = if (validD.isNotEmpty()) max(validD[min(validD.size - 1, (validD.size * 0.98).toInt())].toDouble(), dLo + 1e-3) else 2.0
        val roiVals = ArrayList<Double>()
        for (yy in (sh * 0.4).toInt() until (sh * 0.6).toInt()) for (xx in (sw * 0.4).toInt() until (sw * 0.6).toInt()) {
            val v = dispAll[yy * sw + xx]
            if (v > 0.5f) roiVals.add(v.toDouble())
        }
        val dCenter = if (roiVals.size >= 50) FrameSynchronizer.median(roiVals) else if (stats.centerCount >= 8) stats.centerDx else 0.0
        val focusM = res.focusDistanceM
        // B = Z * d / f with Z = the distance the camera focused at and d = disparity of the frame centre (where AF looks).
        val focusBaselineHint = if (focusM != null && dCenter > 0.0) focusM * dCenter / focal else 0.0

        // Scale priority: a sweep-short IMU estimate, then a CALIBRATED/APPROXIMATE focus distance, otherwise relative units.
        val baseline: Double
        val scaleState: String
        if (res.scaleKnown) {
            baseline = baselineUnits
            scaleState = "IMU"
        } else if (focusBaselineHint > 0.0 && (res.focusCalibration == "APPROXIMATE" || res.focusCalibration == "CALIBRATED")) {
            baseline = focusBaselineHint
            scaleState = "FOCUS"
        } else {
            baseline = 1.0
            scaleState = "UNSCALED"
        }

        val disp8 = Mat()
        disp32.convertTo(disp8, CvType.CV_8U, 255.0 / (dMax - dMin), -dMin * 255.0 / (dMax - dMin))
        val dispColor = Mat()
        Imgproc.applyColorMap(disp8, dispColor, Imgproc.COLORMAP_TURBO)
        dispColor.setTo(Scalar(0.0, 0.0, 0.0), invalid)

        // Depth Z = f B / d, visualised near = warm, far = cool, invalid black.
        val depth = Mat()
        Core.divide(focal * baseline, disp32, depth)
        depth.setTo(Scalar(0.0), invalid)
        // Colour range from the data itself (2nd..98th percentile), so it does not depend on the metric scale being right.
        val zMin = focal * baseline / dHi
        val zMax = max(focal * baseline / dLo, zMin + 1e-6)
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
        Core.hconcat(listOf(rect.smallL, rect.smallR), pair)
        var y = 0
        while (y < pair.rows()) {
            Imgproc.line(pair, Point(0.0, y.toDouble()), Point(pair.cols().toDouble(), y.toDouble()), Scalar(0.0, 255.0, 0.0), 1)
            y += 32
        }

        val epiErr = stats.dyMedian
        val confidence = (validFraction * exp(-epiErr / 2.0)).coerceIn(0.0, 1.0)
        val tEnd = System.nanoTime()
        val diag = JSONObject()
            .put("acquisitionMethod", res.method.name)
            .put("secondCaptureBecameLeft", swapped)
            .put("leftRightDecision", signNote)
            .put("measuredDisparitySignMedianPx", stats.dxMedian)
            .put("alignedMatchesForSign", stats.alignedCount)
            .put("featureDisparityP2_P98_px", org.json.JSONArray(listOf(stats.dxP2, stats.dxP98)))
            .put("adaptiveNumDisparities", nd)
            .put("scaleState", scaleState)
            .put("baselineM_orUnits", baseline)
            .put("baselineSource", res.baselineSource)
            .put("focusDistanceM", focusM ?: JSONObject.NULL)
            .put("focusCalibration", res.focusCalibration)
            .put("centerDisparityPx", dCenter)
            .put("focusAnchoredBaselineM_hint", focusBaselineHint)
            .put("disparityPercentile2_98_px", org.json.JSONArray(listOf(dLo, dHi)))
            .put("baselineHorizontalShare", horizontalShare)
            .put("rectifyAlpha", params.rectifyAlpha)
            .put("rectifyRotationDeg_view1", rect.r1Deg)
            .put("rectifyRotationDeg_view2", rect.r2Deg)
            .put("rectifyZoomFactor", rect.zoom)
            .put("validSourceAreaFraction", validAreaFraction)
            .put("photometricNormalization", params.photometricNormalization)
            .put("focalPxAtMatcherScale", focal)
            .put("matcherScale", s)
            .put("numDisparitiesAtMatcherScale", nd)
            .put("blockSize", bs)
            .put("validDisparityFraction", validFraction)
            .put("validDisparityFractionOfUsableArea", validFraction / max(1e-6, validAreaFraction))
            .put("disparityMinPx", dMin).put("disparityMaxPx", dMax)
            .put("rectifiedEpipolarErrorMedianPx", epiErr)
            .put("reconstructionConfidence", confidence)
            .put("depthMetricScaleNote", "Metric depth is only as good as the baseline estimate (motion-sensor derived, confidence ${"%.2f".format(res.baselineConfidence)}). Relative structure is independent of it.")
            .put("timingMs", JSONObject()
                .put("rectify", (tRect - t0) / 1e6)
                .put("disparity", (tDisp - tRect) / 1e6)
                .put("total", (tEnd - t0) / 1e6))
        Diagnostics.log(
            TAG,
            "valid=${"%.0f".format(100 * validFraction / max(1e-6, validAreaFraction))}%ofUsable epipolarErr=${"%.2f".format(epiErr)}px dispSign=${"%.1f".format(stats.dxMedian)}px " +
                "swapped=$swapped rectRot=(${"%.1f".format(rect.r1Deg)},${"%.1f".format(rect.r2Deg)})° zoom=${"%.2f".format(rect.zoom)} " +
                "scale=$scaleState ${if (scaleState == "UNSCALED") "(relative)" else "%.1fcm".format(baseline * 100)} focusHint=${"%.1fcm".format(focusBaselineHint * 100)} total=${"%.0f".format((tEnd - t0) / 1e6)}ms"
        )

        val rectL = rect.rectL
        val rectR = rect.rectR
        for (mat in listOf(kMat, ml, mr, disp32, invalid, outside, valid, disp8, depth, clamped, depth8, rect.smallL, rect.smallR, rect.gl, rect.gr, rect.valid)) mat.release()
        return StereoOutput(rectL, rectR, pair, dispColor, depthColor, disp16, focal, baseline, swapped, dLo, dHi, scaleState, diag)
    }

    /** Stereo-rectify [left] and [right] given the pose (X_right = R X_left + T). */
    private fun rectify(left: Mat, right: Mat, r: DoubleArray, t: DoubleArray, kMat: Mat, w: Int, h: Int): Rectified {
        val dist = Mat.zeros(5, 1, CvType.CV_64F)
        val rM = matFrom(r, 3, 3)
        val tM = matFrom(t, 3, 1)
        val r1 = Mat(); val r2 = Mat(); val p1 = Mat(); val p2 = Mat(); val q = Mat()
        Calib3d.stereoRectify(
            kMat, dist, kMat, dist, Size(w.toDouble(), h.toDouble()), rM, tM, r1, r2, p1, p2, q,
            Calib3d.CALIB_ZERO_DISPARITY, params.rectifyAlpha
        )

        val r1Arr = DoubleArray(9); r1.get(0, 0, r1Arr)
        val r2Arr = DoubleArray(9); r2.get(0, 0, r2Arr)
        val focalFull = p1.get(0, 0)[0]
        val zoom = focalFull / kMat.get(0, 0)[0]

        val ones = Mat(h, w, CvType.CV_8UC1, Scalar(255.0))
        val m1 = Mat(); val m2 = Mat()
        val rectL = Mat(); val rectR = Mat()
        val maskL = Mat(); val maskR = Mat()
        val size = Size(w.toDouble(), h.toDouble())
        Calib3d.initUndistortRectifyMap(kMat, dist, r1, p1, size, CvType.CV_32FC1, m1, m2)
        Imgproc.remap(left, rectL, m1, m2, Imgproc.INTER_LINEAR)
        Imgproc.remap(ones, maskL, m1, m2, Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, Scalar(0.0))
        Calib3d.initUndistortRectifyMap(kMat, dist, r2, p2, size, CvType.CV_32FC1, m1, m2)
        Imgproc.remap(right, rectR, m1, m2, Imgproc.INTER_LINEAR)
        Imgproc.remap(ones, maskR, m1, m2, Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, Scalar(0.0))

        val s = params.scale
        val sw = (w * s).toInt()
        val sh = (h * s).toInt()
        val smallL = Mat(); val smallR = Mat(); val gl = Mat(); val gr = Mat()
        Imgproc.resize(rectL, smallL, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.resize(rectR, smallR, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.cvtColor(smallL, gl, Imgproc.COLOR_BGR2GRAY)
        Imgproc.cvtColor(smallR, gr, Imgproc.COLOR_BGR2GRAY)

        val both = Mat()
        Core.bitwise_and(maskL, maskR, both)
        val valid = Mat()
        Imgproc.resize(both, valid, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
        Imgproc.erode(valid, valid, kernel)

        for (mat in listOf(dist, rM, tM, r1, r2, p1, p2, q, ones, m1, m2, maskL, maskR, both, kernel)) mat.release()
        return Rectified(rectL, rectR, smallL, smallR, gl, gr, valid, focalFull, Mat3.angleDeg(r1Arr), Mat3.angleDeg(r2Arr), zoom)
    }

    /** (I - localMean) / (localStd + eps): removes gain and offset differences (including colour-cast changes) before matching. */
    private fun normalizeLocal(gray: Mat): Mat {
        val f = Mat()
        gray.convertTo(f, CvType.CV_32F)
        val mu = Mat()
        Imgproc.GaussianBlur(f, mu, Size(0.0, 0.0), 7.0)
        val diff = Mat()
        Core.subtract(f, mu, diff)
        val sq = Mat()
        Core.multiply(diff, diff, sq)
        val variance = Mat()
        Imgproc.GaussianBlur(sq, variance, Size(0.0, 0.0), 7.0)
        val sd = Mat()
        Core.sqrt(variance, sd)
        Core.add(sd, Scalar(8.0), sd)
        val norm = Mat()
        Core.divide(diff, sd, norm)
        val out = Mat()
        norm.convertTo(out, CvType.CV_8U, 48.0, 128.0)
        f.release(); mu.release(); diff.release(); sq.release(); variance.release(); sd.release(); norm.release()
        return out
    }

    private fun matFrom(a: DoubleArray, rows: Int, cols: Int): Mat {
        val m = Mat(rows, cols, CvType.CV_64F)
        m.put(0, 0, *a)
        return m
    }

    /**
     * Rectification quality and left/right order from matched features in the rectified pair:
     * dy = median |row difference| (should be ~0), dx = median (xLeft - xRight) over row-aligned matches (should be > 0).
     */
    private fun matchStats(gl: Mat, gr: Mat): MatchStats {
        val orb = ORB.create(1200)
        val kpL = MatOfKeyPoint(); val kpR = MatOfKeyPoint()
        val dL = Mat(); val dR = Mat()
        orb.detectAndCompute(gl, Mat(), kpL, dL)
        orb.detectAndCompute(gr, Mat(), kpR, dR)
        if (dL.empty() || dR.empty()) return MatchStats(99.0, 0.0, 0)
        val cw = gl.cols().toDouble()
        val ch = gl.rows().toDouble()
        val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
        val knn = ArrayList<MatOfDMatch>()
        matcher.knnMatch(dL, dR, knn, 2)
        val a = kpL.toArray()
        val b = kpR.toArray()
        val dys = ArrayList<Double>()
        val dxs = ArrayList<Double>()
        val centerDxs = ArrayList<Double>()
        for (m in knn) {
            val arr = m.toArray()
            if (arr.size >= 2 && arr[0].distance < 0.75f * arr[1].distance) {
                val pl = a[arr[0].queryIdx].pt
                val pr = b[arr[0].trainIdx].pt
                val dy = abs(pl.y - pr.y)
                if (dy < 15.0) dys.add(dy)
                if (dy < 4.0) {
                    dxs.add(pl.x - pr.x)
                    if (abs(pl.x - cw / 2) < cw * 0.15 && abs(pl.y - ch / 2) < ch * 0.15) centerDxs.add(pl.x - pr.x)
                }
            }
            m.release()
        }
        kpL.release(); kpR.release(); dL.release(); dR.release()
        val dyMed = if (dys.size < 10) 99.0 else FrameSynchronizer.median(dys)
        val dxMed = if (dxs.size < 10) 0.0 else FrameSynchronizer.median(dxs)
        val sorted = dxs.sorted()
        val p2 = if (sorted.size >= 10) sorted[(sorted.size * 0.02).toInt()] else 0.0
        val p98 = if (sorted.size >= 10) sorted[min(sorted.size - 1, (sorted.size * 0.98).toInt())] else 0.0
        val cMed = if (centerDxs.size >= 8) FrameSynchronizer.median(centerDxs) else 0.0
        return MatchStats(dyMed, dxMed, dxs.size, p2, p98, cMed, centerDxs.size)
    }
}
