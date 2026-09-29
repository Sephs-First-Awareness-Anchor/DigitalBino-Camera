// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Size
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

private const val TAG = "SWEEP"

/** All thresholds in one place so they can be tuned from exported diagnostics without touching logic. */
class SweepConfig {
    @Volatile var targetBaselineM = 0.06        // guidance and metric-scale reference, NOT the acceptance test
    var acceptParallaxPx = 10.0                 // translation-induced parallax (tracking-scale px) required to accept Eye B
    var rotationToleranceDeg = 5.0
    var minOverlapPct = 45.0
    var minEInliers = 40
    var minSharpnessRatio = 0.45
    var maxBrightnessDiff = 0.25
    var minLateralFraction = 0.4                // loose: integrated IMU displacement drifts, so this is only a sanity check
    var minSignedLateralFraction = 0.3          // of target baseline, in the sweep direction
    var settleMs = 350L
    var timeoutMs = 30_000L
    var trackScale = 0.5
    var analysisW = 1280
    var analysisH = 960
    var candidateBuffer = 8
    var candidateMaxAgeMs = 2000L
    var sceneBadFrames = 25
    var stillAccRms = 0.25                      // m/s^2: below this the phone is considered held still
    var stillGyro = 0.12                        // rad/s
}

object Yuv {
    /** Copy a YUV_420_888 image into a tightly packed NV21 array. */
    fun toNv21(img: Image): ByteArray {
        val w = img.width
        val h = img.height
        val out = ByteArray(w * h * 3 / 2)
        val yP = img.planes[0]
        val uP = img.planes[1]
        val vP = img.planes[2]
        val yBuf = yP.buffer
        var pos = 0
        for (row in 0 until h) {
            yBuf.position(row * yP.rowStride)
            yBuf.get(out, pos, w)
            pos += w
        }
        val uBuf = uP.buffer
        val vBuf = vP.buffer
        var o = w * h
        for (row in 0 until h / 2) {
            for (col in 0 until w / 2) {
                out[o++] = vBuf.get(row * vP.rowStride + col * vP.pixelStride)
                out[o++] = uBuf.get(row * uP.rowStride + col * uP.pixelStride)
            }
        }
        return out
    }
}

/**
 * MotionBaselineProvider: ONE rear camera stays open and streaming the whole time.
 * Eye A is a frame from the live stream; Eye B is a later frame from the same stream, chosen automatically once
 * the view has shifted by genuine translation parallax (not explained by rotation) with acceptable rotation,
 * overlap, sharpness and exposure. The camera is never closed or reopened between the two eyes.
 *
 * What proves "enough baseline" is measured PARALLAX (vision). Integrated accelerometer displacement drifts by
 * centimetres within seconds, so it is used only for direction sanity and for a coarse metric-scale estimate.
 *
 * Threading: camera callbacks copy frames and hand them to one processing thread. All sweep state lives on that thread.
 */
class MotionBaselineProvider(
    private val context: Context,
    private val probe: ProbeResult,
    private val listener: SweepListener,
    val config: SweepConfig = SweepConfig()
) : StereoAcquisitionProvider {

    override val method = AcquisitionMethod.MOTION_BASELINE

    private enum class Phase { LIVE, WAIT_EYE_A, SWEEPING, SETTLING, PROCESSING, DONE, FAILED }

    private class EyeA(val up: Mat, val feats: FrameFeatures, val tsNs: Long, val rot: DoubleArray)

    private class Candidate(
        val up: Mat, val tsNs: Long, val score: Double, val parallax: Double,
        val rotB: DoubleArray, val pWorld: DoubleArray, val vWorld: DoubleArray, val still: Boolean,
        val metrics: VisualMetrics, val pitch: Double, val yaw: Double, val roll: Double,
        val sharp: Double, val wallMs: Long
    )

    private val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val camThread = HandlerThread("sweep-camera").also { it.start() }
    private val camHandler = Handler(camThread.looper)
    private val procThread = HandlerThread("sweep-proc").also { it.start() }
    private val procHandler = Handler(procThread.looper)
    private val camExecutor = Executor { r -> camHandler.post(r) }

    private val sensors = MotionSensors(context)
    private val tracker = VisualTracker(1500)

    private val busy = AtomicBoolean(false)
    @Volatile private var stopped = false

    // camera
    private lateinit var camId: String
    private lateinit var camInfo: CameraInfo
    private var sensorOrientation = 90
    private var realtimeTimestamps = true
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var builder: CaptureRequest.Builder? = null
    private var reader: ImageReader? = null
    private var afContinuous = false
    private var aeLockAvailable = false

    // geometry
    private lateinit var kUp: Intrinsics
    private lateinit var kTrack: Intrinsics
    private var rotateCode = -1

    // sweep state (processing thread only)
    private var phase = Phase.LIVE
    private var eyeA: EyeA? = null
    private val candidates = ArrayList<Candidate>()
    private var waitStartMs = 0L
    private var sweepStartMs = 0L
    private var settleStartNs = 0L
    private var rejected = 0
    private val rejectReasons = HashMap<String, Int>()
    private var sceneBadStreak = 0
    private var frameCounter = 0
    private var sweepSign = 0                       // +1 right, -1 left, 0 not yet determined
    private val frameLog = ArrayList<JSONObject>()
    private var lastUp: Mat? = null

    // ───────────────────────── lifecycle ─────────────────────────

    private fun failUi(msg: String) {
        if (!stopped) listener.onFailure(msg)
    }

    override fun start() {
        if (!sensors.usable) {
            failUi("Motion sensors (rotation + linear acceleration) are unavailable on this device.")
            return
        }
        val rear = probe.cameras.filter { it.listed && it.facing == CameraCharacteristics.LENS_FACING_BACK }
        if (rear.isEmpty()) {
            failUi("No listed rear camera found.")
            return
        }
        camInfo = rear.maxByOrNull { pixelArea(it) } ?: rear.first()
        camId = camInfo.id
        sensorOrientation = camInfo.json.optInt("sensorOrientationDeg", 90)
        realtimeTimestamps = camInfo.timestampSource == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME

        val size = chooseSize() ?: run {
            failUi("Camera $camId exposes no YUV_420_888 output size.")
            return
        }
        val kLand = Intrinsics.derive(camInfo, size.width, size.height)
        kUp = Intrinsics.rotatedForUpright(kLand, sensorOrientation)
        kTrack = kUp.scaled(config.trackScale)
        rotateCode = when (sensorOrientation) {
            90 -> Core.ROTATE_90_CLOCKWISE
            180 -> Core.ROTATE_180
            270 -> Core.ROTATE_90_COUNTERCLOCKWISE
            else -> -1
        }
        Diagnostics.log(TAG, "camera $camId ${size.width}x${size.height} sensorOrientation=$sensorOrientation realtimeTs=$realtimeTimestamps")
        Diagnostics.log(TAG, "K(upright)=fx ${"%.1f".format(kUp.fx)} fy ${"%.1f".format(kUp.fy)} cx ${"%.1f".format(kUp.cx)} cy ${"%.1f".format(kUp.cy)}; ${kUp.source}")
        Diagnostics.log(TAG, "reported lensIntrinsicCalibration=${camInfo.json.optString("lensIntrinsicCalibration_fx_fy_cx_cy_s")} derivedHFovDeg=${camInfo.json.optJSONArray("derivedHorizontalFovDeg")}")
        Diagnostics.log(TAG, "sensors: ${sensors.describe()}")
        sensors.start()
        openCamera(size)
    }

    override fun stop() {
        stopped = true
        try { sensors.endIntegration(); sensors.stop() } catch (_: Throwable) {}
        camHandler.post {
            try { session?.stopRepeating() } catch (_: Throwable) {}
            try { session?.close() } catch (_: Throwable) {}
            try { device?.close() } catch (_: Throwable) {}
            try { reader?.setOnImageAvailableListener(null, null) } catch (_: Throwable) {}
            try { reader?.close() } catch (_: Throwable) {}
            session = null; device = null; reader = null
            camThread.quitSafely()
        }
        procHandler.post {
            clearCandidates(null)
            // eyeA is not released if it was handed to a result.
            if (phase != Phase.DONE) { eyeA?.up?.release(); eyeA?.feats?.release() }
            lastUp?.release(); lastUp = null
            procThread.quitSafely()
        }
    }

    override fun setTargetBaselineCm(cm: Double) {
        config.targetBaselineM = cm / 100.0
    }

    override fun captureFirstEye() {
        procHandler.post {
            if (phase == Phase.LIVE) {
                phase = Phase.WAIT_EYE_A
                waitStartMs = SystemClock.elapsedRealtime()
                Diagnostics.log(TAG, "capture requested: waiting for a stable Eye A frame")
            }
        }
    }

    private fun pixelArea(c: CameraInfo): Long {
        val parts = c.json.optString("pixelArraySize").split("x").map { it.trim().toLongOrNull() }
        return if (parts.size == 2 && parts[0] != null && parts[1] != null) parts[0]!! * parts[1]!! else 0L
    }

    private fun chooseSize(): Size? {
        val sizes = camInfo.streamSizes[ImageFormat.YUV_420_888] ?: return null
        val target = config.analysisW.toLong() * config.analysisH
        val fourThree = sizes.filter { abs(it.width.toDouble() / it.height - 4.0 / 3.0) < 0.01 }
        val pool = if (fourThree.isNotEmpty()) fourThree else sizes
        return pool.minByOrNull { abs(it.width.toLong() * it.height - target) }
    }

    // ───────────────────────── camera ─────────────────────────

    @SuppressLint("MissingPermission")
    private fun openCamera(size: Size) {
        try {
            val chars = manager.getCameraCharacteristics(camId)
            afContinuous = chars.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
                ?.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE) == true
            aeLockAvailable = chars.get(CameraCharacteristics.CONTROL_AE_LOCK_AVAILABLE) == true

            val rd = ImageReader.newInstance(size.width, size.height, ImageFormat.YUV_420_888, 3)
            reader = rd
            rd.setOnImageAvailableListener({ r ->
                var img: Image? = null
                try {
                    img = r.acquireLatestImage()
                    if (img != null && !stopped && !busy.get()) {
                        val nv21 = Yuv.toNv21(img)
                        val ts = if (realtimeTimestamps) img.timestamp else SystemClock.elapsedRealtimeNanos()
                        val w = img.width
                        val h = img.height
                        busy.set(true)
                        procHandler.post {
                            try {
                                process(nv21, w, h, ts)
                            } catch (t: Throwable) {
                                Diagnostics.error(TAG, "frame processing failed", t)
                            } finally {
                                busy.set(false)
                            }
                        }
                    }
                } catch (t: Throwable) {
                    if (!stopped) Diagnostics.error(TAG, "image handling failed", t)
                } finally {
                    try { img?.close() } catch (_: Throwable) {}
                }
            }, camHandler)

            manager.openCamera(camId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    if (stopped) { try { camera.close() } catch (_: Throwable) {}; return }
                    device = camera
                    createSession(camera, rd)
                }

                override fun onDisconnected(camera: CameraDevice) {
                    try { camera.close() } catch (_: Throwable) {}
                    failUi("Camera disconnected (another app may have taken it).")
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    try { camera.close() } catch (_: Throwable) {}
                    failUi("Camera error: ${Diagnostics.deviceErrorName(error)}")
                }
            }, camHandler)
        } catch (t: Throwable) {
            failUi("Could not open camera: ${Diagnostics.describe(t)}")
        }
    }

    private fun createSession(camera: CameraDevice, rd: ImageReader) {
        try {
            val cfg = SessionConfiguration(
                SessionConfiguration.SESSION_REGULAR,
                listOf(OutputConfiguration(rd.surface)),
                camExecutor,
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(s: CameraCaptureSession) {
                        if (stopped) { try { s.close() } catch (_: Throwable) {}; return }
                        session = s
                        try {
                            val b = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                            b.addTarget(rd.surface)
                            if (afContinuous) b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                            builder = b
                            s.setRepeatingRequest(b.build(), null, camHandler)
                            Diagnostics.log(TAG, "camera streaming (single continuous session)")
                        } catch (t: Throwable) {
                            failUi("Could not start streaming: ${Diagnostics.describe(t)}")
                        }
                    }

                    override fun onConfigureFailed(s: CameraCaptureSession) {
                        failUi("Camera session configuration failed.")
                    }
                }
            )
            camera.createCaptureSession(cfg)
        } catch (t: Throwable) {
            failUi("Could not create camera session: ${Diagnostics.describe(t)}")
        }
    }

    /** Lock exposure, white balance and focus so both eyes share the same photometric and optical state. */
    private fun setLocks(locked: Boolean) {
        camHandler.post {
            val b = builder ?: return@post
            val s = session ?: return@post
            try {
                if (aeLockAvailable) {
                    b.set(CaptureRequest.CONTROL_AE_LOCK, locked)
                    b.set(CaptureRequest.CONTROL_AWB_LOCK, locked)
                }
                if (afContinuous) {
                    b.set(
                        CaptureRequest.CONTROL_AF_MODE,
                        if (locked) CaptureRequest.CONTROL_AF_MODE_AUTO else CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                    )
                }
                s.setRepeatingRequest(b.build(), null, camHandler)
                Diagnostics.log(TAG, "AE/AWB/AF ${if (locked) "LOCKED" else "released"} (aeLockAvailable=$aeLockAvailable)")
            } catch (t: Throwable) {
                if (!stopped) Diagnostics.error(TAG, "lock update failed", t)
            }
        }
    }

    // ───────────────────────── frame processing (processing thread) ─────────────────────────

    private fun process(nv21: ByteArray, w: Int, h: Int, tsNs: Long) {
        if (stopped) return
        val yuv = Mat(h + h / 2, w, CvType.CV_8UC1)
        val bgr = Mat()
        val up = Mat()
        val small = Mat()
        val gray = Mat()
        try {
            yuv.put(0, 0, nv21)
            Imgproc.cvtColor(yuv, bgr, Imgproc.COLOR_YUV2BGR_NV21)
            if (rotateCode >= 0) Core.rotate(bgr, up, rotateCode) else bgr.copyTo(up)
            val tw = (up.cols() * config.trackScale).toInt()
            val th = (up.rows() * config.trackScale).toInt()
            Imgproc.resize(up, small, org.opencv.core.Size(tw.toDouble(), th.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.cvtColor(small, gray, Imgproc.COLOR_BGR2GRAY)
            listener.onLiveFrame(Bmp.fromBgr(small))
            frameCounter++

            if (phase == Phase.SWEEPING || phase == Phase.SETTLING) {
                lastUp?.release()
                lastUp = up.clone()
            }

            when (phase) {
                Phase.LIVE -> liveStep()
                Phase.WAIT_EYE_A -> waitEyeAStep(up, small, gray, tsNs)
                Phase.SWEEPING, Phase.SETTLING -> sweepStep(up, gray, tsNs)
                else -> {}
            }
        } finally {
            yuv.release(); bgr.release(); up.release(); small.release(); gray.release()
        }
    }

    private fun status(
        headline: String, detail: String = "", baseline: Double = 0.0, progress: Double = 0.0,
        pitch: Double = 0.0, yaw: Double = 0.0, roll: Double = 0.0,
        m: VisualMetrics = VisualMetrics.EMPTY, parallax: Double = 0.0, orientationOk: Boolean = true
    ) {
        listener.onStatus(
            SweepStatus(
                phase.name, headline, detail, baseline * 100.0, config.targetBaselineM * 100.0,
                progress.coerceIn(0.0, 1.0), config.acceptParallaxPx,
                pitch, yaw, roll, m.overlapPct, m.matches, m.eInliers, parallax, orientationOk
            )
        )
    }

    private fun liveStep() {
        val moving = sensors.gyroSpeed() > 0.3
        status(
            if (moving) "Hold steady, then press CAPTURE" else "Frame your subject, then press CAPTURE",
            "Aim at something with texture and some depth, about 1 to 3 metres away"
        )
    }

    private fun waitEyeAStep(up: Mat, small: Mat, gray: Mat, tsNs: Long) {
        val waited = SystemClock.elapsedRealtime() - waitStartMs
        val stable = sensors.gyroSpeed() < 0.3 || waited > 1500
        if (!stable) { status("Hold still…"); return }

        val feats = tracker.describe(gray)
        if (feats.points.size < 100) {
            feats.release()
            status("Not enough scene detail here", "Aim at something with texture and depth, then press CAPTURE again")
            phase = Phase.LIVE
            return
        }
        val rot = sensors.rotationAt(tsNs)
        if (rot == null) {
            feats.release()
            phase = Phase.FAILED
            failUi("Motion sensors are not reporting orientation.")
            return
        }
        eyeA = EyeA(up.clone(), feats, tsNs, rot)
        sensors.beginIntegration(tsNs)
        setLocks(true)
        rejected = 0; rejectReasons.clear(); sceneBadStreak = 0; sweepSign = 0; frameLog.clear()
        sweepStartMs = SystemClock.elapsedRealtime()
        phase = Phase.SWEEPING
        Diagnostics.log(TAG, "Eye A locked: features=${feats.points.size} sharp=${"%.1f".format(feats.sharpness)} bias=${sensors.biasSnapshot().joinToString { "%.4f".format(it) }}")
        listener.onEyeALocked(Bmp.fromBgr(small))
    }

    private fun sweepStep(up: Mat, gray: Mat, tsNs: Long) {
        val a = eyeA ?: return
        val cfg = config
        val nowMs = SystemClock.elapsedRealtime()

        if (nowMs - sweepStartMs > cfg.timeoutMs) {
            fail("Sweep timed out. Press RESET and try again.")
            return
        }
        val rotB = sensors.rotationAt(tsNs)
        if (rotB == null) { status("Motion sensors not reporting"); return }

        // Device rotation from A to B expressed in A's axes (right-handed about device X, Y, Z).
        val delta = Mat3.mul(Mat3.transpose(a.rot), rotB)
        val w = Mat3.smallAngleVec(delta)
        val pitch = Math.toDegrees(w[0])
        val yaw = Math.toDegrees(w[1])
        val roll = Math.toDegrees(w[2])
        val relCam = Mat3.devToCam(Mat3.mul(Mat3.transpose(rotB), a.rot))
        val rotWorst = max(abs(pitch), max(abs(yaw), abs(roll)))
        val tol = cfg.rotationToleranceDeg
        val rotOk = rotWorst <= tol

        // Coarse IMU displacement in device-A axes (X right, Y up, Z toward the user). Drifts; used for direction sanity only.
        val pW = sensors.displacementWorld()
        val vW = sensors.velocityWorld()
        val dDev = Mat3.mulVec(Mat3.transpose(a.rot), pW)
        val total = Mat3.norm(dDev)
        val lateralRaw = dDev[0]
        if (sweepSign == 0 && abs(lateralRaw) >= 0.012) sweepSign = if (lateralRaw > 0) 1 else -1
        val dirSign = if (sweepSign == 0) 1 else sweepSign
        val lateral = lateralRaw * dirSign
        val dirWord = if (dirSign > 0) "right" else "left"
        val target = cfg.targetBaselineM

        val fb = tracker.describe(gray)
        val m = tracker.compare(a.feats, fb, kTrack, relCam)

        val parallax = m.parallaxImuRotPx
        val progress = parallax / cfg.acceptParallaxPx
        val overlapOk = m.overlapPct >= cfg.minOverlapPct
        val geomOk = m.eInliers >= cfg.minEInliers && m.rotation != null
        val parallaxOk = parallax >= cfg.acceptParallaxPx
        val sharpOk = fb.sharpness >= cfg.minSharpnessRatio * a.feats.sharpness
        val brightOk = abs(fb.brightness - a.feats.brightness) / max(1.0, a.feats.brightness) <= cfg.maxBrightnessDiff
        val latFrac = if (total > 1e-6) lateral / total else 0.0
        val latDirOk = latFrac >= cfg.minLateralFraction && lateral >= cfg.minSignedLateralFraction * target
        val consistentOk = m.matches < 60 || m.consistentPct >= 25.0
        val eRatio = if (m.matches > 0) m.eInliers.toDouble() / m.matches else 0.0
        // "Scene changed" only when matches are wrong everywhere (rotation-inconsistent AND no epipolar structure).
        if (m.matches >= 80 && m.consistentPct < 15.0 && eRatio < 0.15) sceneBadStreak++ else sceneBadStreak = 0

        val reasons = ArrayList<String>()
        if (!rotOk) reasons.add("rotation")
        if (!overlapOk) reasons.add("overlap")
        if (!geomOk) reasons.add("geometry")
        if (!sharpOk) reasons.add("blur")
        if (!brightOk) reasons.add("exposure")
        if (!latDirOk) reasons.add("not-lateral")
        if (!consistentOk) reasons.add("scene-changed")
        val gatesExceptParallax = reasons.isEmpty()
        if (!parallaxOk) reasons.add("parallax")
        val eligible = reasons.isEmpty()
        if (!eligible) {
            rejected++
            for (r in reasons) rejectReasons[r] = (rejectReasons[r] ?: 0) + 1
        }

        val stillNow = sensors.gyroSpeed() < cfg.stillGyro && sensors.recentAccRms() < cfg.stillAccRms
        // Prefer frames a little past the required parallax, sharp, well-oriented, with lots of geometry.
        val parScore = 1.0 - min(1.0, abs(progress - 1.25) / 1.25)
        val score = 0.35 * parScore +
            0.20 * min(1.0, m.eInliers / 250.0) +
            0.15 * min(1.0, fb.sharpness / max(1e-6, a.feats.sharpness)) +
            0.15 * max(0.0, 1.0 - rotWorst / tol) +
            0.10 * min(1.0, m.overlapPct / 100.0) +
            0.05 * (if (stillNow) 1.0 else 0.0)

        if (gatesExceptParallax && progress >= 0.5) {
            candidates.add(Candidate(up.clone(), tsNs, score, parallax, rotB, pW, vW, stillNow, m, pitch, yaw, roll, fb.sharpness, nowMs))
            pruneCandidates(nowMs)
        }
        fb.release()

        val row = JSONObject()
            .put("tMs", (tsNs - a.tsNs) / 1e6)
            .put("imuLateralCm", lateralRaw * 100).put("imuTotalCm", total * 100)
            .put("pitch", pitch).put("yaw", yaw).put("roll", roll)
            .put("matches", m.matches).put("eInliers", m.eInliers).put("poseInliers", m.poseInliers)
            .put("overlapPct", m.overlapPct).put("consistentPct", m.consistentPct)
            .put("parallaxImuPx", m.parallaxImuRotPx).put("parallaxVisPx", m.parallaxVisRotPx)
            .put("flowMedianPx", m.flowMedianPx).put("imuVisRotDisagreeDeg", m.imuVisRotDisagreementDeg)
            .put("sharpRatio", fb.sharpness / max(1e-6, a.feats.sharpness))
            .put("score", score).put("still", stillNow).put("eligible", eligible)
            .put("reasons", reasons.joinToString(","))
        if (frameLog.size < 600) frameLog.add(row)

        if (frameCounter % 3 == 0) {
            Diagnostics.log(
                TAG,
                "par=${"%.1f".format(parallax)}/${"%.0f".format(cfg.acceptParallaxPx)}px imu=${"%.1f".format(lateralRaw * 100)}/${"%.1f".format(total * 100)}cm " +
                    "rot=(${"%.1f".format(pitch)},${"%.1f".format(yaw)},${"%.1f".format(roll)})° match=${m.matches} E=${m.eInliers} pose=${m.poseInliers} " +
                    "ovl=${"%.0f".format(m.overlapPct)}% cons=${"%.0f".format(m.consistentPct)}% flow=${"%.1f".format(m.flowMedianPx)}px " +
                    "${if (eligible) "OK" else reasons.joinToString(",")}"
            )
        }

        if (sceneBadStreak >= cfg.sceneBadFrames) {
            fail("Scene changed too much. Try again.")
            return
        }

        if (phase == Phase.SWEEPING && eligible) {
            phase = Phase.SETTLING
            settleStartNs = tsNs
        }
        if (phase == Phase.SETTLING) {
            if (tsNs - settleStartNs >= cfg.settleMs * 1_000_000L) {
                val best = candidates.filter { it.parallax >= cfg.acceptParallaxPx }.maxByOrNull { it.score }
                if (best != null) {
                    finalizeCapture(a, best)
                    return
                }
                phase = Phase.SWEEPING
            } else {
                status("Baseline acquired", "Hold still: choosing the best frame", lateral, progress, pitch, yaw, roll, m, parallax, rotOk)
                return
            }
        }

        val (headline, detail) = when {
            !rotOk -> rotationHint(pitch, yaw, roll, tol)
            !parallaxOk && total < 0.008 && progress < 0.15 -> Pair("Slide phone $dirWord", "Keep the phone's orientation, move it sideways")
            !parallaxOk -> Pair(
                "Keep sliding $dirWord",
                "Parallax ${"%.1f".format(parallax)} of ${"%.0f".format(cfg.acceptParallaxPx)} px  ·  about ${"%.1f".format(lateral * 100)} cm by motion sensors (coarse)"
            )
            !latDirOk -> Pair("Move sideways, not forward or up", "Sensors say the motion is mostly not sideways")
            !overlapOk -> Pair("Bring the scene back into view", "")
            !sharpOk -> Pair("Slow down", "Motion blur")
            !geomOk -> Pair("Not enough matching detail", "Point at something with more texture and depth")
            !consistentOk -> Pair("Scene is changing", "Keep the subject still")
            else -> Pair("Baseline acquired", "")
        }
        status(headline, detail, lateral, progress, pitch, yaw, roll, m, parallax, rotOk)
    }

    /** Derived from right-hand rotation of the device about its own axes; direction wording still to be confirmed on hardware. */
    private fun rotationHint(pitch: Double, yaw: Double, roll: Double, tol: Double): Pair<String, String> {
        val worst = max(abs(pitch), max(abs(yaw), abs(roll)))
        val text = when (worst) {
            abs(pitch) -> if (pitch > 0) "Tilt phone down slightly" else "Tilt phone up slightly"
            abs(yaw) -> if (yaw > 0) "Rotate phone slightly right" else "Rotate phone slightly left"
            else -> "Level phone"
        }
        return Pair(text, "pitch ${"%.1f".format(pitch)}°  yaw ${"%.1f".format(yaw)}°  roll ${"%.1f".format(roll)}°  (limit ±${"%.0f".format(tol)}°)")
    }

    private fun pruneCandidates(nowMs: Long) {
        val it = candidates.iterator()
        while (it.hasNext()) {
            val c = it.next()
            if (nowMs - c.wallMs > config.candidateMaxAgeMs) { c.up.release(); it.remove() }
        }
        while (candidates.size > config.candidateBuffer) {
            candidates.removeAt(0).up.release()
        }
    }

    private fun clearCandidates(keep: Candidate?) {
        for (c in candidates) if (c !== keep) c.up.release()
        candidates.clear()
    }

    private fun fail(message: String) {
        phase = Phase.FAILED
        sensors.endIntegration()
        Diagnostics.log(TAG, "FAILED: $message rejected=$rejected reasons=$rejectReasons")
        dumpFailure(message)
        failUi(message)
    }

    /** Failure is data: save the per-frame telemetry and the two frames involved so the run can be analysed offline. */
    private fun dumpFailure(reason: String) {
        try {
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val base = "sweepfail_$stamp"
            val j = JSONObject()
                .put("reportFormat", "binosweep-failure/1")
                .put("reason", reason)
                .put("rejectedFrames", rejected)
                .put("rejectReasons", JSONObject(rejectReasons as Map<*, *>))
                .put("intrinsics", kUp.toJson())
                .put("rotationSource", sensors.rotationSourceName)
                .put("imuBias", JSONArray(sensors.biasSnapshot().toList()))
                .put("sweepSign", sweepSign)
                .put("config", configJson())
                .put("frames", JSONArray(frameLog))
            ReportExporter.save(context, "${base}.json", "application/json", j.toString(2).toByteArray())
            savePng("${base}_eyeA.png", eyeA?.up)
            savePng("${base}_lastFrame.png", lastUp)
        } catch (t: Throwable) {
            Diagnostics.error(TAG, "failure dump failed", t)
        }
    }

    private fun savePng(name: String, m: Mat?) {
        if (m == null || m.empty()) return
        val buf = MatOfByte()
        Imgcodecs.imencode(".png", m, buf)
        ReportExporter.save(context, name, "image/png", buf.toArray())
        buf.release()
    }

    private fun configJson(): JSONObject = JSONObject()
        .put("targetBaselineM", config.targetBaselineM)
        .put("acceptParallaxPx", config.acceptParallaxPx)
        .put("rotationToleranceDeg", config.rotationToleranceDeg)
        .put("minOverlapPct", config.minOverlapPct)
        .put("minEInliers", config.minEInliers)
        .put("minSharpnessRatio", config.minSharpnessRatio)
        .put("minLateralFraction", config.minLateralFraction)
        .put("sceneBadFrames", config.sceneBadFrames)

    // ───────────────────────── finalize: refine pose, fuse baseline, emit result ─────────────────────────

    private fun finalizeCapture(a: EyeA, best: Candidate) {
        phase = Phase.PROCESSING
        sensors.endIntegration()
        listener.onEyeBLocked()
        status("Eye B locked", "Estimating relative pose")

        val relCam = Mat3.devToCam(Mat3.mul(Mat3.transpose(best.rotB), a.rot))
        val gA = Mat()
        val gB = Mat()
        val refine = VisualTracker(3500)
        var vm = best.metrics
        var poseSource = "tracking-scale (${kTrack.width}x${kTrack.height})"
        try {
            Imgproc.cvtColor(a.up, gA, Imgproc.COLOR_BGR2GRAY)
            Imgproc.cvtColor(best.up, gB, Imgproc.COLOR_BGR2GRAY)
            val fa = refine.describe(gA)
            val fb = refine.describe(gB)
            val full = refine.compare(fa, fb, kUp, relCam)
            fa.release(); fb.release()
            if (full.rotation != null && full.tUnit != null && full.eInliers >= 60) {
                vm = full
                poseSource = "full-resolution (${kUp.width}x${kUp.height})"
            }
        } catch (t: Throwable) {
            Diagnostics.error(TAG, "full-resolution pose refinement failed; using tracking-scale pose", t)
        } finally {
            gA.release(); gB.release()
        }

        val rot = vm.rotation
        val tu = vm.tUnit
        if (rot == null || tu == null || vm.eInliers < config.minEInliers) {
            clearCandidates(null)
            a.up.release(); a.feats.release()
            fail("Geometry could not be recovered confidently. Try again with more texture and depth in view.")
            return
        }

        // Visual baseline direction in camera-A coordinates: c = -R^T t.
        val rt = Mat3.mulVec(Mat3.transpose(rot), tu)
        val cVis = doubleArrayOf(-rt[0], -rt[1], -rt[2])

        // Drift correction: if the phone was held still at Eye B, true velocity is ~0, so the residual velocity is
        // accumulated bias. Constant bias b gives v_err = bT and p_err = bT^2/2 = v_end*T/2.
        val tSec = (best.tsNs - a.tsNs) / 1e9
        val pCorrW = if (best.still) doubleArrayOf(
            best.pWorld[0] - best.vWorld[0] * tSec / 2.0,
            best.pWorld[1] - best.vWorld[1] * tSec / 2.0,
            best.pWorld[2] - best.vWorld[2] * tSec / 2.0
        ) else best.pWorld
        val rAT = Mat3.transpose(a.rot)
        val dDevRaw = Mat3.mulVec(rAT, best.pWorld)
        val dDevUsed = Mat3.mulVec(rAT, pCorrW)
        val dCam = doubleArrayOf(dDevUsed[0], -dDevUsed[1], -dDevUsed[2])   // upright camera-A axes
        val dNorm = Mat3.norm(dCam)
        val cos = if (dNorm > 1e-9) Mat3.dot(dCam, cVis) / (dNorm * Mat3.norm(cVis)) else 0.0
        val baseline: Double
        val baselineSource: String
        if (cos > 0.5) {
            baseline = Mat3.dot(dCam, cVis) / Mat3.norm(cVis)
            baselineSource = "motion-sensor displacement${if (best.still) " (drift-corrected, held still)" else " (NOT drift-corrected: phone was moving)"} " +
                "projected onto the visual translation direction (cos=${"%.2f".format(cos)})"
        } else {
            baseline = dNorm
            baselineSource = "motion-sensor displacement magnitude; its direction DISAGREES with vision (cos=${"%.2f".format(cos)}), scale unreliable"
        }
        val stillFactor = if (best.still) 1.0 else 0.6
        val baselineConf = cos.coerceIn(0.0, 1.0) * min(1.0, vm.eInliers / 150.0) * stillFactor
        val captureConf = (0.5 * best.score + 0.5 * baselineConf).coerceIn(0.0, 1.0)

        val q = JSONObject()
            .put("cameraId", camId)
            .put("sensorOrientationDeg", sensorOrientation)
            .put("rotationSource", sensors.rotationSourceName)
            .put("intrinsics", kUp.toJson())
            .put("poseSource", poseSource)
            .put("requestedBaselineM", config.targetBaselineM)
            .put("estimatedBaselineM", baseline)
            .put("baselineSource", baselineSource)
            .put("heldStillAtEyeB", best.still)
            .put("imuDisplacementRawDeviceAxesM_xRight_yUp_zTowardUser", JSONArray(dDevRaw.toList()))
            .put("imuDisplacementUsedDeviceAxesM", JSONArray(dDevUsed.toList()))
            .put("imuBiasDeviceAxes", JSONArray(sensors.biasSnapshot().toList()))
            .put("visualTranslationDirCamA", JSONArray(cVis.toList()))
            .put("imuVsVisualDirectionCos", cos)
            .put("rotationDeviceDeg_pitchYawRoll", JSONArray(listOf(best.pitch, best.yaw, best.roll)))
            .put("imuVsVisualRotationDisagreementDeg", vm.imuVisRotDisagreementDeg)
            .put("matches", vm.matches).put("essentialInliers", vm.eInliers).put("poseInliers", vm.poseInliers)
            .put("overlapPct", vm.overlapPct).put("consistentPct", vm.consistentPct)
            .put("parallaxImuRotationOnlyPx_trackingScale", best.metrics.parallaxImuRotPx)
            .put("parallaxVisualRotationOnlyPx_finalRes", vm.parallaxVisRotPx)
            .put("poseConfidence", baselineConf)
            .put("eyeATimestampNs", a.tsNs).put("eyeBTimestampNs", best.tsNs)
            .put("timeSeparationMs", (best.tsNs - a.tsNs) / 1e6)
            .put("rejectedFrames", rejected)
            .put("rejectReasons", JSONObject(rejectReasons as Map<*, *>))
            .put("selectedCandidateScore", best.score)
            .put("sharpnessEyeA", a.feats.sharpness).put("sharpnessEyeB", best.sharp)
            .put("aeAwbAfLocked", aeLockAvailable)
            .put("config", configJson())
            .put("frames", JSONArray(frameLog))

        clearCandidates(best)
        a.feats.release()
        val result = StereoAcquisitionResult(
            AcquisitionMethod.MOTION_BASELINE, a.up, best.up, a.tsNs, best.tsNs, kUp,
            rot, tu, baseline, baselineSource, baselineConf, captureConf, q
        )
        phase = Phase.DONE
        Diagnostics.log(TAG, "RESULT baseline=${"%.1f".format(baseline * 100)}cm conf=${"%.2f".format(captureConf)} E=${vm.eInliers} dt=${"%.0f".format(tSec * 1000)}ms still=${best.still}")
        if (!stopped) listener.onResult(result)
    }
}
