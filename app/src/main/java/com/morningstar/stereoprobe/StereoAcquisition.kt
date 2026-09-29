// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.graphics.Bitmap
import org.json.JSONObject
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * How a binocular pair was obtained. The stereo engine never branches on this: it only consumes
 * [StereoAcquisitionResult]. Device-specific workarounds live in the providers.
 */
enum class AcquisitionMethod(val label: String) {
    CONCURRENT_CAMERAS("Simultaneous stereo cameras"),
    MOTION_BASELINE("Motion-Baseline (Bino Sweep)")
}

/** Last probe/pair-test results, shared between screens. */
object AppState {
    @Volatile var probe: ProbeResult? = null
    @Volatile var run: TestRun? = null
}

object AcquisitionSelector {
    /**
     * Use simultaneous acquisition only when the pair test has VALIDATED a true stereo candidate.
     * Anything else (including "pair test not run yet") uses Motion-Baseline. A missing concurrent
     * capability is never treated as terminal incompatibility.
     */
    fun select(run: TestRun?): AcquisitionMethod {
        val validated = run?.pairs?.any { it.classification == PairClass.TRUE_STEREO_CANDIDATE } == true
        return if (validated) AcquisitionMethod.CONCURRENT_CAMERAS else AcquisitionMethod.MOTION_BASELINE
    }
}

/** Pinhole intrinsics, in pixels, for the image as delivered (upright). Distortion is not modelled. */
class Intrinsics(
    val fx: Double, val fy: Double, val cx: Double, val cy: Double,
    val width: Int, val height: Int, val source: String
) {
    fun kMat(): Mat {
        val k = Mat.eye(3, 3, CvType.CV_64F)
        k.put(0, 0, fx)
        k.put(1, 1, fy)
        k.put(0, 2, cx)
        k.put(1, 2, cy)
        return k
    }

    fun scaled(s: Double): Intrinsics =
        Intrinsics(fx * s, fy * s, cx * s, cy * s, Math.round(width * s).toInt(), Math.round(height * s).toInt(), "$source, scaled x$s")

    fun toJson(): JSONObject = JSONObject()
        .put("fx", fx).put("fy", fy).put("cx", cx).put("cy", cy)
        .put("width", width).put("height", height).put("source", source)

    companion object {
        /** Approximate pinhole model from reported focal length and physical sensor width (landscape sensor frame). */
        fun derive(info: CameraInfo, w: Int, h: Int): Intrinsics {
            val widthMm = info.json.optString("sensorPhysicalSizeMm").split("x").getOrNull(0)?.trim()?.toDoubleOrNull()
            val f = info.focalLengths.firstOrNull()?.toDouble()
            return if (widthMm != null && f != null && widthMm > 0) {
                val fx = f * w / widthMm
                Intrinsics(fx, fx, w / 2.0, h / 2.0, w, h,
                    "DERIVED from focal length ${f}mm and sensor width ${widthMm}mm (uncalibrated, no distortion model)")
            } else {
                val fx = w * 0.8
                Intrinsics(fx, fx, w / 2.0, h / 2.0, w, h, "FALLBACK GUESS (focal length or sensor size missing)")
            }
        }

        /** Intrinsics after the sensor image is rotated to upright by [sensorOrientation] degrees clockwise. */
        fun rotatedForUpright(k: Intrinsics, sensorOrientation: Int): Intrinsics = when (sensorOrientation) {
            90 -> Intrinsics(k.fy, k.fx, k.height - k.cy, k.cx, k.height, k.width, k.source + ", rotated 90 cw")
            270 -> Intrinsics(k.fy, k.fx, k.cy, k.width - k.cx, k.height, k.width, k.source + ", rotated 270 cw")
            180 -> Intrinsics(k.fx, k.fy, k.width - k.cx, k.height - k.cy, k.width, k.height, k.source + ", rotated 180")
            else -> k
        }
    }
}

/**
 * Common output of every acquisition method.
 * Pose convention (same as OpenCV stereoCalibrate): X_second = R * X_first + translationUnit * baselineMeters.
 * [first] and [second] are upright BGR images of identical size; this object owns them.
 */
class StereoAcquisitionResult(
    val method: AcquisitionMethod,
    val first: Mat,
    val second: Mat,
    val firstTimestampNs: Long,
    val secondTimestampNs: Long,
    val intrinsics: Intrinsics,
    val relativeRotation: DoubleArray,      // 3x3 row-major
    val translationUnit: DoubleArray,       // unit vector
    val baselineMeters: Double,
    val baselineSource: String,
    val baselineConfidence: Double,         // 0..1
    val captureConfidence: Double,          // 0..1
    val quality: JSONObject
) {
    fun release() {
        first.release()
        second.release()
    }
}

class SweepStatus(
    val phase: String,
    val headline: String,
    val detail: String,
    val baselineCm: Double,
    val targetCm: Double,
    val pitchDeg: Double,
    val yawDeg: Double,
    val rollDeg: Double,
    val overlapPct: Double,
    val matches: Int,
    val inliers: Int,
    val parallaxPx: Double,
    val orientationOk: Boolean
)

interface SweepListener {
    fun onLiveFrame(frame: Bitmap)
    fun onStatus(status: SweepStatus)
    fun onEyeALocked(ghost: Bitmap)
    fun onEyeBLocked()
    fun onResult(result: StereoAcquisitionResult)
    fun onFailure(message: String)
}

interface StereoAcquisitionProvider {
    val method: AcquisitionMethod
    fun start()
    fun captureFirstEye()
    fun setTargetBaselineCm(cm: Double)
    fun stop()
}

object Bmp {
    fun fromBgr(bgr: Mat): Bitmap {
        val rgba = Mat()
        Imgproc.cvtColor(bgr, rgba, Imgproc.COLOR_BGR2RGBA)
        val bmp = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bmp)
        rgba.release()
        return bmp
    }
}
