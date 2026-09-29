// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Size
import java.util.Collections
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.max

private const val TAG = "COORD"

/** One completed capture as reported by the camera device. */
data class FrameRec(val timestampNs: Long, val exposureNs: Long, val frameNumber: Long)

/** Everything one stream produced. All lists are safe to touch from the camera thread. */
class StreamRecorder(val label: String) {
    val imageTs: MutableList<Long> = Collections.synchronizedList(ArrayList<Long>())
    val captures: MutableList<FrameRec> = Collections.synchronizedList(ArrayList<FrameRec>())
    val failures = AtomicInteger(0)
    val failureReasons = ConcurrentHashMap<Int, Int>()
    val imageErrors = AtomicInteger(0)
    val firstRxNs = AtomicLong(0)

    fun imageTimestamps(): List<Long> = synchronized(imageTs) { ArrayList(imageTs) }
    fun captureRecords(): List<FrameRec> = synchronized(captures) { ArrayList(captures) }
}

/** Resources owned by one running stream; close() is safe to call at any point. */
class Rig {
    val readers = ArrayList<ImageReader>()
    var device: CameraDevice? = null
    var session: CameraCaptureSession? = null

    fun close() {
        try { session?.stopRepeating() } catch (_: Throwable) {}
        try { session?.abortCaptures() } catch (_: Throwable) {}
        try { session?.close() } catch (_: Throwable) {}
        try { device?.close() } catch (_: Throwable) {}
        for (r in readers) {
            try { r.setOnImageAvailableListener(null, null) } catch (_: Throwable) {}
            try { r.close() } catch (_: Throwable) {}
        }
        readers.clear()
        session = null
        device = null
    }
}

class OpenOutcome(val device: CameraDevice?, val error: String?, val throwable: Throwable?, val errorCode: Int?)
class SessionOutcome(val session: CameraCaptureSession?, val error: String?, val throwable: Throwable?)
class StepFail(val stage: String, val msg: String, val throwable: Throwable?, val code: Int?)

/**
 * CameraCoordinator: camera opening, sessions, concurrency and lifecycle.
 * Every method that blocks MUST be called from a worker thread (never the main thread and never
 * the camera callback thread owned by this class).
 */
class CameraCoordinator(context: Context, private val probe: ProbeResult) {

    companion object {
        const val STREAM_MS = 3500L
        const val MIN_FRAMES = 10
        const val SETTLE_MS = 700L
        private const val DROP_LEADING = 5
    }

    private val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val thread = HandlerThread("stereoprobe-camera").also { it.start() }
    val handler = Handler(thread.looper)
    val executor = Executor { r -> handler.post(r) }

    fun shutdown() {
        thread.quitSafely()
    }

    // ───────────────────────── primitives ─────────────────────────

    fun pickSize(info: CameraInfo, format: Int, targetW: Int = 640, targetH: Int = 480): Size? =
        info.streamSizes[format]?.minByOrNull {
            abs(it.width.toLong() * it.height - targetW.toLong() * targetH)
        }

    fun openBlocking(id: String, late: MutableMap<String, String>, timeoutMs: Long = 5000): OpenOutcome {
        val latch = CountDownLatch(1)
        val ref = AtomicReference<CameraDevice?>(null)
        val err = AtomicReference<String?>(null)
        val code = AtomicReference<Int?>(null)
        val cb = object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                ref.set(camera)
                latch.countDown()
            }

            override fun onDisconnected(camera: CameraDevice) {
                late[id] = "onDisconnected"
                if (ref.get() == null) {
                    err.set("onDisconnected during open")
                    latch.countDown()
                }
                try { camera.close() } catch (_: Throwable) {}
            }

            override fun onError(camera: CameraDevice, error: Int) {
                val n = Diagnostics.deviceErrorName(error)
                late[id] = "onError($n)"
                if (ref.get() == null) {
                    err.set("onError($n)")
                    code.set(error)
                    latch.countDown()
                }
                try { camera.close() } catch (_: Throwable) {}
            }
        }
        return try {
            manager.openCamera(id, cb, handler)
            if (!latch.await(timeoutMs, TimeUnit.MILLISECONDS)) {
                OpenOutcome(null, "open timed out after ${timeoutMs}ms", null, null)
            } else if (ref.get() != null) {
                OpenOutcome(ref.get(), null, null, null)
            } else {
                OpenOutcome(null, err.get(), null, code.get())
            }
        } catch (t: Throwable) {
            OpenOutcome(null, Diagnostics.describe(t), t, null)
        }
    }

    fun createSessionBlocking(device: CameraDevice, outputs: List<OutputConfiguration>, timeoutMs: Long = 5000): SessionOutcome {
        val latch = CountDownLatch(1)
        val ref = AtomicReference<CameraCaptureSession?>(null)
        val err = AtomicReference<String?>(null)
        val cb = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                ref.set(session)
                latch.countDown()
            }

            override fun onConfigureFailed(session: CameraCaptureSession) {
                err.set("onConfigureFailed")
                latch.countDown()
            }
        }
        return try {
            device.createCaptureSession(SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outputs, executor, cb))
            if (!latch.await(timeoutMs, TimeUnit.MILLISECONDS)) {
                SessionOutcome(null, "session configure timed out after ${timeoutMs}ms", null)
            } else if (ref.get() != null) {
                SessionOutcome(ref.get(), null, null)
            } else {
                SessionOutcome(null, err.get(), null)
            }
        } catch (t: Throwable) {
            SessionOutcome(null, Diagnostics.describe(t), t)
        }
    }

    private fun attachReader(reader: ImageReader, rec: StreamRecorder) {
        reader.setOnImageAvailableListener({ r ->
            var img: android.media.Image? = null
            try {
                img = r.acquireNextImage()
                if (img != null) {
                    if (rec.firstRxNs.get() == 0L) rec.firstRxNs.set(SystemClock.elapsedRealtimeNanos())
                    rec.imageTs.add(img.timestamp)
                }
            } catch (t: Throwable) {
                rec.imageErrors.incrementAndGet()
            } finally {
                try { img?.close() } catch (_: Throwable) {}
            }
        }, handler)
    }

    private fun captureCb(rec: StreamRecorder) = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
            val exp = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L
            rec.captures.add(FrameRec(ts, exp, result.frameNumber))
        }

        override fun onCaptureFailed(session: CameraCaptureSession, request: CaptureRequest, failure: CaptureFailure) {
            rec.failures.incrementAndGet()
            rec.failureReasons.merge(failure.reason, 1) { a, b -> a + b }
        }
    }

    /** Open one camera, configure one YUV stream into an ImageReader, start repeating capture. Null = success. */
    fun bringUp(id: String, size: Size, rig: Rig, rec: StreamRecorder, late: MutableMap<String, String>): StepFail? {
        val rd = ImageReader.newInstance(size.width, size.height, ImageFormat.YUV_420_888, 4)
        rig.readers.add(rd)
        attachReader(rd, rec)

        val o = openBlocking(id, late)
        val dev = o.device ?: return StepFail("open", o.error ?: "unknown open failure", o.throwable, o.errorCode)
        rig.device = dev

        val s = createSessionBlocking(dev, listOf(OutputConfiguration(rd.surface)))
        val sess = s.session ?: return StepFail("session", s.error ?: "unknown session failure", s.throwable, null)
        rig.session = sess

        return try {
            val req = dev.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply { addTarget(rd.surface) }.build()
            sess.setRepeatingRequest(req, captureCb(rec), handler)
            null
        } catch (t: Throwable) {
            StepFail("repeating", Diagnostics.describe(t), t, null)
        }
    }

    // ───────────────────────── the full sweep ─────────────────────────

    fun testAll(progress: (String) -> Unit): TestRun {
        val singles = ArrayList<SingleResult>()
        val pairs = ArrayList<PairResult>()
        val hidden = LinkedHashMap<String, String>()

        // 0. Baseline: does each listed camera stream on its own? Separates "our bug" from "device limit".
        val listed = probe.cameras.filter { it.listed }.map { it.id }.sortedWith(idComparator)
        for (id in listed) {
            progress("single-camera baseline $id")
            singles.add(testSingle(id))
            Thread.sleep(SETTLE_MS)
        }
        val singleOk = singles.associate { it.id to it.streamed }

        // 1. Direct-open pairs: everything Android claims, plus every rear+rear pair.
        val claimed = probe.claimedPairs()
        val wanted = LinkedHashSet<Pair<String, String>>()
        for (p in claimed) if (p.first in listed && p.second in listed) wanted.add(p)
        for (p in probe.rearRearListedPairs()) wanted.add(p)
        Diagnostics.log(TAG, "pairs to test: $wanted (claimed by Android: $claimed)")

        for ((a, b) in wanted) {
            progress("pair $a + $b (direct open)")
            val r = try {
                testDirectPair(a, b, claimed.contains(Pair(a, b)))
            } catch (t: Throwable) {
                Diagnostics.error(TAG, "pair $a+$b crashed inside test harness", t)
                PairResult("DIRECT_OPEN", a, b).also {
                    it.failStage = "harness"
                    it.failure = Diagnostics.describe(t)
                    it.notes.add("Test harness exception: treat as implementation problem, not a device verdict.")
                }
            }
            if (singleOk[a] == false || singleOk[b] == false) {
                r.notes.add("A single-camera baseline failed for ${listOf(a, b).filter { singleOk[it] == false }}: this pair result is not cleanly attributable to the device.")
            }
            pairs.add(r)
            Thread.sleep(SETTLE_MS)
        }

        // 2. Logical multi-camera: two physical streams inside one logical device.
        for (li in probe.cameras.filter { it.listed && it.logical && it.physicalIds.size >= 2 }) {
            val phys = li.physicalIds.sortedWith(idComparator)
            for (i in phys.indices) for (j in i + 1 until phys.size) {
                progress("logical ${li.id}: physical ${phys[i]} + ${phys[j]}")
                val r = try {
                    testLogicalPhysical(li.id, phys[i], phys[j])
                } catch (t: Throwable) {
                    Diagnostics.error(TAG, "logical ${li.id} ${phys[i]}+${phys[j]} crashed inside harness", t)
                    PairResult("LOGICAL_PHYSICAL_STREAMS", phys[i], phys[j], li.id).also {
                        it.failStage = "harness"
                        it.failure = Diagnostics.describe(t)
                    }
                }
                pairs.add(r)
                Thread.sleep(SETTLE_MS)
            }
        }

        // 3. Hidden physical IDs: can a third-party app open them directly?
        for (h in probe.cameras.filter { !it.listed }) {
            progress("direct open of hidden physical id ${h.id}")
            val late = ConcurrentHashMap<String, String>()
            val o = openBlocking(h.id, late)
            if (o.device != null) {
                hidden[h.id] = "OPENED directly (unexpected for a hidden physical id)"
                try { o.device.close() } catch (_: Throwable) {}
            } else {
                hidden[h.id] = "open refused: ${o.error}"
            }
            Diagnostics.log(TAG, "hidden ${h.id}: ${hidden[h.id]}")
            Thread.sleep(SETTLE_MS)
        }

        return TestRun(singles, pairs, hidden)
    }

    // ───────────────────────── single-camera baseline ─────────────────────────

    private fun testSingle(id: String): SingleResult {
        val r = SingleResult(id)
        val info = probe.byId[id]
        val size = info?.let { pickSize(it, ImageFormat.YUV_420_888) }
        if (size == null) {
            r.failure = "no YUV_420_888 output size exposed"
            return r
        }
        val rec = StreamRecorder(id)
        val rig = Rig()
        val late = ConcurrentHashMap<String, String>()
        val t0 = SystemClock.elapsedRealtime()
        try {
            val fail = bringUp(id, size, rig, rec, late)
            if (fail != null) {
                r.opened = rig.device != null
                r.failure = "[${fail.stage}] ${fail.msg}"
                Diagnostics.log(TAG, "single $id FAIL ${r.failure}")
            } else {
                r.opened = true
                Thread.sleep(2500)
                val ts = rec.imageTimestamps()
                r.images = ts.size
                r.captures = rec.captureRecords().size
                r.streamed = ts.size >= MIN_FRAMES
                val secs = (SystemClock.elapsedRealtime() - t0) / 1000.0
                r.approxFps = if (secs > 0) ts.size / secs else 0.0
                val period = FrameSynchronizer.medianPeriodMs(ts.sorted())
                if (period > 0) r.approxFps = 1000.0 / period
                if (!r.streamed) r.failure = "only ${ts.size} images received"
                Diagnostics.log(TAG, "single $id ${r.summary()}")
            }
        } catch (t: Throwable) {
            r.failure = Diagnostics.describe(t)
            Diagnostics.error(TAG, "single $id exception", t)
        } finally {
            rig.close()
        }
        return r
    }

    // ───────────────────────── direct pair test ─────────────────────────

    private fun sharesSensor(a: CameraInfo, b: CameraInfo): Boolean =
        a.id in b.physicalIds || b.id in a.physicalIds || a.physicalIds.intersect(b.physicalIds).isNotEmpty()

    private fun checkConcurrentConfig(r: PairResult, a: String, b: String, sa: Size, sb: Size) {
        val ra = ImageReader.newInstance(sa.width, sa.height, ImageFormat.YUV_420_888, 2)
        val rb = ImageReader.newInstance(sb.width, sb.height, ImageFormat.YUV_420_888, 2)
        try {
            val noop = object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {}
                override fun onConfigureFailed(session: CameraCaptureSession) {}
            }
            val m = mapOf(
                a to SessionConfiguration(SessionConfiguration.SESSION_REGULAR, listOf(OutputConfiguration(ra.surface)), executor, noop),
                b to SessionConfiguration(SessionConfiguration.SESSION_REGULAR, listOf(OutputConfiguration(rb.surface)), executor, noop)
            )
            val ok = manager.isConcurrentSessionConfigurationSupported(m)
            r.concurrentConfigSupported = ok
            r.concurrentConfigNote = "isConcurrentSessionConfigurationSupported(YUV ${sa.width}x${sa.height} + YUV ${sb.width}x${sb.height}) = $ok"
        } catch (t: Throwable) {
            r.concurrentConfigNote = "isConcurrentSessionConfigurationSupported threw ${Diagnostics.describe(t)}"
        } finally {
            try { ra.close() } catch (_: Throwable) {}
            try { rb.close() } catch (_: Throwable) {}
        }
        Diagnostics.log(TAG, "${r.idA}+${r.idB} ${r.concurrentConfigNote}")
    }

    private fun recordFailure(r: PairResult, f: StepFail) {
        r.failStage = f.stage
        r.failure = f.msg
        val t = f.throwable
        val restricted = t is SecurityException ||
            (t is CameraAccessException && t.reason == CameraAccessException.CAMERA_DISABLED) ||
            f.code == CameraDevice.StateCallback.ERROR_CAMERA_DISABLED
        if (restricted) r.restricted = true

        val lower = f.msg
        r.locusHint = when {
            restricted ->
                "Policy/permission layer: the camera service refused access (security or disabled-camera policy)."
            lower.contains("MAX_CAMERAS_IN_USE") ->
                "Camera service / HAL resource limit: too many cameras open. Typical of hardware or HAL not supporting this combination; not an app bug."
            lower.contains("CAMERA_IN_USE") ->
                "Another client (or an unreleased earlier test) holds a camera. Close other camera apps and retry before drawing conclusions."
            f.stage == "session" ->
                "Camera HAL rejected the stream configuration for this camera while the other was open."
            else -> null
        }
        Diagnostics.log(TAG, "FAIL[${f.stage}] ${r.idA}+${r.idB}: ${f.msg}")
    }

    private fun testDirectPair(a: String, b: String, claimed: Boolean): PairResult {
        val r = PairResult("DIRECT_OPEN", a, b)
        val ia = probe.byId.getValue(a)
        val ib = probe.byId.getValue(b)
        r.claimedConcurrent = claimed
        r.rearRear = ia.facing == CameraCharacteristics.LENS_FACING_BACK && ib.facing == CameraCharacteristics.LENS_FACING_BACK
        r.sharedSensor = sharesSensor(ia, ib)
        r.timestampsComparable = ia.timestampSource == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME &&
            ib.timestampSource == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
        if (ia.focalLengths.isNotEmpty() && ia.focalLengths == ib.focalLengths) {
            r.notes.add("Both ids report identical focal lengths ${ia.focalLengths}: check they are not aliases of one sensor.")
        }
        if (r.sharedSensor) r.notes.add("One id is the logical parent of the other (shared physical sensor): NOT two independent viewpoints.")
        if (!r.timestampsComparable) r.notes.add("SENSOR_INFO_TIMESTAMP_SOURCE is not REALTIME on both: cross-camera Δt is not meaningful.")

        val sa = pickSize(ia, ImageFormat.YUV_420_888)
        val sb = pickSize(ib, ImageFormat.YUV_420_888)
        if (sa == null || sb == null) {
            r.failStage = "setup"
            r.failure = "no YUV_420_888 output size for ${if (sa == null) a else b}"
            conclude(r, StreamRecorder(a), StreamRecorder(b))
            return r
        }

        checkConcurrentConfig(r, a, b, sa, sb)

        val recA = StreamRecorder(a)
        val recB = StreamRecorder(b)
        val rigA = Rig()
        val rigB = Rig()
        val late = ConcurrentHashMap<String, String>()
        try {
            run stage@{
                val fa = bringUp(a, sa, rigA, recA, late)
                if (fa != null) { recordFailure(r, fa); r.notes.add("failed while bringing up FIRST camera $a"); return@stage }
                val fb = bringUp(b, sb, rigB, recB, late)
                if (fb != null) { recordFailure(r, fb); r.notes.add("failed while bringing up SECOND camera $b (first camera $a was already streaming)"); return@stage }
                Thread.sleep(STREAM_MS)
            }
        } catch (t: Throwable) {
            recordFailure(r, StepFail("exception", Diagnostics.describe(t), t, null))
        } finally {
            r.lateEvents = HashMap(late)
            rigA.close()
            rigB.close()
        }
        conclude(r, recA, recB)
        return r
    }

    // ───────────────────────── logical camera, two physical streams ─────────────────────────

    private fun testLogicalPhysical(logicalId: String, p1: String, p2: String): PairResult {
        val r = PairResult("LOGICAL_PHYSICAL_STREAMS", p1, p2, logicalId)
        val li = probe.byId.getValue(logicalId)
        val i1 = probe.byId[p1]
        val i2 = probe.byId[p2]
        r.rearRear = li.facing == CameraCharacteristics.LENS_FACING_BACK
        r.sharedSensor = false
        r.timestampsComparable = li.timestampSource == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
        r.notes.add("Logical multi-camera sensor sync type reported by Android: ${li.json.optString("logicalSyncType")} (APPROXIMATE = not hardware locked, CALIBRATED = timestamps calibrated).")
        if (i1 == null || i2 == null) r.notes.add("Physical camera characteristics unavailable; using the logical camera's stream map for sizes.")

        val s1 = pickSize(i1 ?: li, ImageFormat.YUV_420_888)
        val s2 = pickSize(i2 ?: li, ImageFormat.YUV_420_888)
        if (s1 == null || s2 == null) {
            r.failStage = "setup"
            r.failure = "no YUV_420_888 size"
            conclude(r, StreamRecorder(p1), StreamRecorder(p2))
            return r
        }

        val rec1 = StreamRecorder(p1)
        val rec2 = StreamRecorder(p2)
        val rig = Rig()
        val late = ConcurrentHashMap<String, String>()
        val physicalResultsSeen = AtomicInteger(0)
        try {
            run stage@{
                val rd1 = ImageReader.newInstance(s1.width, s1.height, ImageFormat.YUV_420_888, 4)
                val rd2 = ImageReader.newInstance(s2.width, s2.height, ImageFormat.YUV_420_888, 4)
                rig.readers.add(rd1); rig.readers.add(rd2)
                attachReader(rd1, rec1)
                attachReader(rd2, rec2)

                val o = openBlocking(logicalId, late)
                val dev = o.device
                if (dev == null) { recordFailure(r, StepFail("open", o.error ?: "unknown", o.throwable, o.errorCode)); return@stage }
                rig.device = dev

                val oc1 = OutputConfiguration(rd1.surface)
                val oc2 = OutputConfiguration(rd2.surface)
                oc1.setPhysicalCameraId(p1)
                oc2.setPhysicalCameraId(p2)

                val so = createSessionBlocking(dev, listOf(oc1, oc2))
                val sess = so.session
                if (sess == null) { recordFailure(r, StepFail("session", so.error ?: "unknown", so.throwable, null)); return@stage }
                rig.session = sess

                val req = dev.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(rd1.surface)
                    addTarget(rd2.surface)
                }.build()
                sess.setRepeatingRequest(req, object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                        val pr = result.physicalCameraResults
                        if (pr.isNotEmpty()) physicalResultsSeen.incrementAndGet()
                        for ((pid, rec) in listOf(Pair(p1, rec1), Pair(p2, rec2))) {
                            val cr = pr[pid] ?: continue
                            val ts = cr.get(CaptureResult.SENSOR_TIMESTAMP) ?: continue
                            rec.captures.add(FrameRec(ts, cr.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L, result.frameNumber))
                        }
                    }

                    override fun onCaptureFailed(session: CameraCaptureSession, request: CaptureRequest, failure: CaptureFailure) {
                        rec1.failures.incrementAndGet()
                        rec2.failures.incrementAndGet()
                    }
                }, handler)
                Thread.sleep(STREAM_MS)
            }
        } catch (t: Throwable) {
            recordFailure(r, StepFail("exception", Diagnostics.describe(t), t, null))
        } finally {
            r.lateEvents = HashMap(late)
            rig.close()
        }
        r.notes.add("Per-physical capture results delivered on ${physicalResultsSeen.get()} capture callbacks.")
        conclude(r, rec1, rec2)
        return r
    }

    // ───────────────────────── classification ─────────────────────────

    private fun conclude(r: PairResult, recA: StreamRecorder, recB: StreamRecorder) {
        val tsA = recA.imageTimestamps().sorted()
        val tsB = recB.imageTimestamps().sorted()
        val capA = recA.captureRecords()
        val capB = recB.captureRecords()
        r.imagesA = tsA.size
        r.imagesB = tsB.size
        r.capturesA = capA.size
        r.capturesB = capB.size
        r.failedCapturesA = recA.failures.get()
        r.failedCapturesB = recB.failures.get()

        if (tsA.size >= MIN_FRAMES && tsB.size >= MIN_FRAMES) {
            r.statsStart = FrameSynchronizer.compute(tsA.drop(DROP_LEADING), tsB.drop(DROP_LEADING))
        }
        if (capA.size >= MIN_FRAMES && capB.size >= MIN_FRAMES) {
            val midA = capA.drop(DROP_LEADING).map { it.timestampNs + it.exposureNs / 2 }
            val midB = capB.drop(DROP_LEADING).map { it.timestampNs + it.exposureNs / 2 }
            r.statsMid = FrameSynchronizer.compute(midA, midB)
        }

        val streamed = tsA.size >= MIN_FRAMES && tsB.size >= MIN_FRAMES
        r.classification = when {
            streamed && r.lateEvents.isEmpty() -> when {
                !r.rearRear -> PairClass.CONCURRENT_NOT_REAR_PAIR
                r.sharedSensor || !r.timestampsComparable -> PairClass.CONCURRENT_UNCERTAIN
                else -> PairClass.TRUE_STEREO_CANDIDATE
            }
            streamed -> {
                r.notes.add("Both streams delivered frames but a camera reported a late event: ${r.lateEvents}")
                PairClass.CONCURRENT_UNCERTAIN
            }
            r.restricted -> PairClass.RESTRICTED
            r.mode == "LOGICAL_PHYSICAL_STREAMS" -> PairClass.LOGICAL_MULTI_CAMERA_ONLY
            else -> PairClass.UNSUPPORTED
        }

        if (!streamed && r.failure == null) {
            r.failStage = "stream"
            r.failure = "streams opened but frames were insufficient (A=${tsA.size}, B=${tsB.size}; need $MIN_FRAMES each)"
        }
        if (r.classification == PairClass.TRUE_STEREO_CANDIDATE) {
            r.notes.add("Label means: two independent rear viewpoints streamed concurrently with comparable timestamps. Synchronisation quality is the measured Δt verdict, not assumed.")
        }
        Diagnostics.log("CLASS", "${r.title()} → ${r.classification.label}")
    }
}
