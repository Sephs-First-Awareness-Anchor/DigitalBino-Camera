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
import org.opencv.imgproc.Imgproc
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

private const val TAG = "SWEEP"

/** All thresholds in one place so they can be tuned from diagnostics without touching logic. */
class SweepConfig {
    @Volatile var targetBaselineM = 0.06
    var rotationToleranceDeg = 5.0
    var minOverlapPct = 45.0
    var minPoseInliers = 40
    var minParallaxPx = 4.0            // at tracking scale
    var minSharpnessRatio = 0.45
    var maxBrightnessDiff = 0.25
    var minLateralFraction = 0.7
    var settleMs = 350L
    var timeoutMs = 25_000L
    var trackScale = 0.5
    var analysisW = 1280
    var analysisH = 960
    var candidateBuffer = 8
    var candidateMaxAgeMs = 2000L
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
 * Eye A is a frame from the live stream; Eye B is a later frame from the same stream, chosen automatically
 * once the phone has been translated sideways with acceptable rotation, overlap, sharpness and parallax.
 * The camera is never closed or reopened between the two eyes.
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
        val up: Mat, val tsNs: Long, val score: Double, val lateral: Double,
        val rotB: DoubleArray, val dDev: DoubleArray, val metrics: VisualMetrics,
        val pitch: Double, val yaw: Double, val roll: Double, val sharp: Double, val wallMs: Long
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

    // ───────────────────────── lifecycle ─────────────────────────

    override fun start() {
        if (!sensors.usable) {
            listener.onFailure("Motion sensors (rotation vector + linear acceleration) are unavailable on this device.")
            return
        }
        val rear = probe.cameras.filter { it.listed && it.facing == CameraCharacteristics.LENS_FACING_BACK }
        if (rear.isEmpty()) {
            listener.onFailure("No listed rear camera found.")
            return
        }
        // Prefer the rear camera with the largest pixel array (the main camera on this handset).
        camInfo = rear.maxByOrNull { pixelArea(it) } ?: rear.first()
        camId = camInfo.id
        sensorOrientation = camInfo.json.optInt("sensorOrientationDeg", 90)
        realtimeTimestamps = camInfo.timestampSource == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME

        val size = chooseSize() ?: run {
            listener.onFailure("Camera $camId exposes no YUV_420_888 output size.")
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
        Diagnostics.log(TAG, "camera $camId size ${size.width}x${size.height} sensorOrientation=$sensorOrientation realtimeTs=$realtimeTimestamps K=${kUp.source}")
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
            // eyeA is intentionally not released here if it was handed to a result.
            if (phase != Phase.DONE) { eyeA?.up?.release(); eyeA?.feats?.release() }
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
                    Diagnostics.error(TAG, "image handling failed", t)
                } finally {
                    try { img?.close() } catch (_: Throwable) {}
                }
            }, camHandler)

            manager.openCamera(camId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    device = camera
                    createSession(camera, rd)
                }

                override fun onDisconnected(camera: CameraDevice) {
                    try { camera.close() } catch (_: Throwable) {}
                    if (!stopped) listener.onFailure("Camera disconnected (another app may have taken it).")
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    try { camera.close() } catch (_: Throwable) {}
                    if (!stopped) listener.onFailure("Camera error: ${Diagnostics.deviceErrorName(error)}")
                }
            }, camHandler)
        } catch (t: Throwable) {
            listener.onFailure("Could not open camera: ${Diagnostics.describe(t)}")
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
                        session = s
                        try {
                            val b = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                            b.addTarget(rd.surface)
                            if (afContinuous) b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                            builder = b
                            s.setRepeatingRequest(b.build(), null, camHandler)
                            Diagnostics.log(TAG, "camera streaming (single continuous session)")
                        } catch (t: Throwable) {
                            listener.onFailure("Could not start streaming: ${Diagnostics.describe(t)}")
                        }
                    }

                    override fun onConfigureFailed(s: CameraCaptureSession) {
                        listener.onFailure("Camera session configuration failed.")
                    }
                }
            )
            camera.createCaptureSession(cfg)
        } catch (t: Throwable) {
            listener.onFailure("Could not create camera session: ${Diagnostics.describe(t)}")
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
                Diagnostics.error(TAG, "lock update failed", t)
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
        headline: String, detail: String = "", baseline: Double = 0.0,
        pitch: Double = 0.0, yaw: Double = 0.0, roll: Double = 0.0,
        m: VisualMetrics = VisualMetrics.EMPTY, orientationOk: Boolean = true
    ) {
        listener.onStatus(
            SweepStatus(
                phase.name, headline, detail, baseline * 100.0, config.targetBaselineM * 100.0,
                pitch, yaw, roll, m.overlapPct, m.matches, m.poseInliers, m.parallaxImuRotPx, orientationOk
            )
        )
    }

    private fun liveStep() {
        val moving = sensors.gyroSpeed() > 0.3
        status(if (moving) "Hold steady, then press CAPTURE" else "Frame your subject, then press CAPTURE",
            "Camera is running; motion sensors ${if (sensors.usable) "OK" else "missing"}")
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
            listener.onFailure("Motion sensors are not reporting orientation.")
            return
        }
        eyeA = EyeA(up.clone(), feats, tsNs, rot)
        sensors.beginIntegration(tsNs)
        setLocks(true)
        rejected = 0; rejectReasons.clear(); sceneBadStreak = 0
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

        // Device rotation from A to B, expressed in A's axes (right-handed about device X, Y, Z).
        val delta = Mat3.mul(Mat3.transpose(a.rot), rotB)
        val w = Mat3.smallAngleVec(delta)
        val pitch = Math.toDegrees(w[0])
        val yaw = Math.toDegrees(w[1])
        val roll = Math.toDegrees(w[2])
        val relCam = Mat3.devToCam(Mat3.mul(Mat3.transpose(rotB), a.rot))

        // IMU displacement expressed in device-A axes (X right, Y up, Z toward the user).
        val dDev = Mat3.mulVec(Mat3.transpose(a.rot), sensors.displacementWorld())
        val total = Mat3.norm(dDev)
        val lateral = dDev[0]
        val target = cfg.targetBaselineM
        val tol = cfg.rotationToleranceDeg
        val rotWorst = max(abs(pitch), max(abs(yaw), abs(roll)))
        val rotOk = rotWorst <= tol

        if (total < 0.01) {
            status("Slide phone right", "Keep the phone's orientation, move it sideways", lateral, pitch, yaw, roll, VisualMetrics.EMPTY, rotOk)
            return
        }

        val fb = tracker.describe(gray)
        val m = tracker.compare(a.feats, fb, kTrack, relCam)

        val overlapOk = m.overlapPct >= cfg.minOverlapPct
        val poseOk = m.poseInliers >= cfg.minPoseInliers
        val parallaxOk = m.parallaxImuRotPx >= cfg.minParallaxPx
        val sharpOk = fb.sharpness >= cfg.minSharpnessRatio * a.feats.sharpness
        val brightOk = abs(fb.brightness - a.feats.brightness) / max(1.0, a.feats.brightness) <= cfg.maxBrightnessDiff
        val latFrac = if (total > 1e-6) lateral / total else 0.0
        val latDirOk = latFrac >= cfg.minLateralFraction
        val inlierRatio = if (m.matches > 0) m.poseInliers.toDouble() / m.matches else 0.0
        val sceneOk = m.matches < 80 || inlierRatio >= 0.3
        if (m.matches >= 80 && inlierRatio < 0.25) sceneBadStreak++ else sceneBadStreak = 0

        val reasons = ArrayList<String>()
        if (!rotOk) reasons.add("rotation")
        if (!overlapOk) reasons.add("overlap")
        if (!poseOk) reasons.add("pose")
        if (!parallaxOk) reasons.add("parallax")
        if (!sharpOk) reasons.add("blur")
        if (!brightOk) reasons.add("exposure")
        if (!latDirOk) reasons.add("not-lateral")
        if (!sceneOk) reasons.add("scene-changed")
        val eligible = reasons.isEmpty()
        if (!eligible) {
            rejected++
            for (r in reasons) rejectReasons[r] = (rejectReasons[r] ?: 0) + 1
        }

        // Candidate score (documented in the exported diagnostics).
        val latScore = 1.0 - min(1.0, abs(lateral - target) / target)
        val score = 0.35 * latScore +
            0.20 * min(1.0, m.poseInliers / 200.0) +
            0.15 * min(1.0, fb.sharpness / max(1e-6, a.feats.sharpness)) +
            0.15 * max(0.0, 1.0 - rotWorst / tol) +
            0.15 * min(1.0, m.overlapPct / 100.0)

        if (eligible && lateral >= 0.5 * target) {
            candidates.add(Candidate(up.clone(), tsNs, score, lateral, rotB, dDev, m, pitch, yaw, roll, fb.sharpness, nowMs))
            pruneCandidates(nowMs)
        }
        fb.release()

        if (frameCounter % 4 == 0) {
            Diagnostics.log(
                TAG,
                "lat=${"%.1f".format(lateral * 100)}cm tot=${"%.1f".format(total * 100)}cm rot=(${"%.1f".format(pitch)},${"%.1f".format(yaw)},${"%.1f".format(roll)})° " +
                    "match=${m.matches} pose=${m.poseInliers} overlap=${"%.0f".format(m.overlapPct)}% par=${"%.1f".format(m.parallaxImuRotPx)}px " +
                    "score=${"%.2f".format(score)} ${if (eligible) "OK" else reasons.joinToString(",")}"
            )
        }

        if (sceneBadStreak >= 12) {
            fail("Scene changed too much. Try again.")
            return
        }

        if (phase == Phase.SWEEPING && eligible && lateral >= 0.9 * target) {
            phase = Phase.SETTLING
            settleStartNs = tsNs
        }
        if (phase == Phase.SETTLING) {
            if (tsNs - settleStartNs >= cfg.settleMs * 1_000_000L) {
                val best = candidates.filter { it.lateral >= 0.9 * target }.maxByOrNull { it.score }
                if (best != null) {
                    finalizeCapture(a, best)
                    return
                }
                phase = Phase.SWEEPING
            } else {
                status("Baseline acquired", "Hold still: choosing the best frame", lateral, pitch, yaw, roll, m, rotOk)
                return
            }
        }

        // Guidance: first unmet condition wins.
        val (headline, detail) = when {
            !rotOk -> rotationHint(pitch, yaw, roll, tol)
            lateral > 1.8 * target -> Pair("Too far: slide back a little", "")
            lateral < 0.9 * target -> Pair("Keep sliding right", "Progress ${"%.1f".format(lateral * 100)} of ${"%.1f".format(target * 100)} cm")
            !latDirOk -> Pair("Move sideways, not forward or up", "Sideways share of motion ${"%.0f".format(latFrac * 100)}%")
            !overlapOk -> Pair("Bring the scene back into view", "")
            !sharpOk -> Pair("Slow down", "Motion blur")
            !parallaxOk -> Pair("No parallax detected", "Scene may be too distant, or you are pivoting instead of sliding")
            !poseOk -> Pair("Not enough matching detail", "Point at something with more texture")
            else -> Pair("Baseline acquired", "")
        }
        status(headline, detail, lateral, pitch, yaw, roll, m, rotOk)
    }

    /** Derived from right-hand rotation of the device about its own axes; direction wording is still to be confirmed on hardware. */
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
        listener.onFailure(message)
    }

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
            if (full.rotation != null && full.tUnit != null && full.poseInliers >= 60) {
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
        if (rot == null || tu == null || vm.poseInliers < config.minPoseInliers) {
            clearCandidates(null)
            a.up.release(); a.feats.release()
            fail("Geometry could not be recovered confidently. Try again with more texture and depth in view.")
            return
        }

        // Visual baseline direction in camera-A coordinates: c = -R^T t.
        val rT = Mat3.transpose(rot)
        val rt = Mat3.mulVec(rT, tu)
        val cVis = doubleArrayOf(-rt[0], -rt[1], -rt[2])
        // IMU displacement in upright camera-A coordinates (x right, y down, z forward).
        val dCam = doubleArrayOf(best.dDev[0], -best.dDev[1], -best.dDev[2])
        val dNorm = Mat3.norm(dCam)
        val cos = if (dNorm > 1e-9) Mat3.dot(dCam, cVis) / (dNorm * Mat3.norm(cVis)) else 0.0
        val baseline: Double
        val baselineSource: String
        if (cos > 0.5) {
            baseline = Mat3.dot(dCam, cVis) / Mat3.norm(cVis)
            baselineSource = "IMU displacement projected onto the visual translation direction (cos=${"%.2f".format(cos)})"
        } else {
            baseline = dNorm
            baselineSource = "IMU displacement magnitude; direction DISAGREES with vision (cos=${"%.2f".format(cos)}), scale unreliable"
        }
        val baselineConf = (cos.coerceIn(0.0, 1.0)) * min(1.0, vm.poseInliers / 150.0)
        val captureConf = (0.5 * best.score + 0.5 * baselineConf).coerceIn(0.0, 1.0)

        val q = JSONObject()
            .put("cameraId", camId)
            .put("sensorOrientationDeg", sensorOrientation)
            .put("intrinsics", kUp.toJson())
            .put("poseSource", poseSource)
            .put("requestedBaselineM", config.targetBaselineM)
            .put("estimatedBaselineM", baseline)
            .put("baselineSource", baselineSource)
            .put("imuDisplacementDeviceAxesM_xRight_yUp_zTowardUser", JSONArray(best.dDev.toList()))
            .put("imuBiasDeviceAxes", JSONArray(sensors.biasSnapshot().toList()))
            .put("visualTranslationDirCamA", JSONArray(cVis.toList()))
            .put("imuVsVisualDirectionCos", cos)
            .put("rotationDeviceDeg_pitchYawRoll", JSONArray(listOf(best.pitch, best.yaw, best.roll)))
            .put("imuVsVisualRotationDisagreementDeg", vm.imuVisRotDisagreementDeg)
            .put("matches", vm.matches).put("essentialInliers", vm.inliers).put("poseInliers", vm.poseInliers)
            .put("overlapPct", vm.overlapPct)
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
            .put("config", JSONObject()
                .put("rotationToleranceDeg", config.rotationToleranceDeg)
                .put("minOverlapPct", config.minOverlapPct)
                .put("minPoseInliers", config.minPoseInliers)
                .put("minParallaxPx", config.minParallaxPx)
                .put("minSharpnessRatio", config.minSharpnessRatio)
                .put("minLateralFraction", config.minLateralFraction))

        clearCandidates(best)
        a.feats.release()
        val result = StereoAcquisitionResult(
            AcquisitionMethod.MOTION_BASELINE, a.up, best.up, a.tsNs, best.tsNs, kUp,
            rot, tu, baseline, baselineSource, baselineConf, captureConf, q
        )
        phase = Phase.DONE
        Diagnostics.log(TAG, "RESULT baseline=${"%.1f".format(baseline * 100)}cm conf=${"%.2f".format(captureConf)} poseInliers=${vm.poseInliers} dt=${"%.0f".format((best.tsNs - a.tsNs) / 1e6)}ms")
        listener.onResult(result)
    }
}
