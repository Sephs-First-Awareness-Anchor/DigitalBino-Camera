// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.params.OutputConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.SystemClock
import android.util.Size
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

private const val TAG = "HOLD"

/**
 * "Suspend one camera and bounce between the two" experiment.
 *
 * Camera2 has no suspend call. The nearest equivalents are:
 *   1. IDLE-OPEN     : device A open with NO session, then try to open B.
 *   2. SESSION-CLOSED: device A open, session created then closed, then try to open B.
 * If both devices can stay open, the SESSION-SWAP loop then measures bouncing by creating/closing
 * a session on each already-open device (no camera open/close in the loop).
 */
class HoldResult(val first: String, val second: String) {
    var idleOpenOutcome = "not run"
    var sessionClosedOutcome = "not run"
    var bothOpenPossible = false
    var swapNote = "not run (both devices could not be held open)"
    val swapSessionMs = ArrayList<Double>()      // createCaptureSession() on an already-open device
    val swapFirstFrameMs = ArrayList<Double>()   // session create → first frame
    val swapGapMs = ArrayList<Double>()          // A frame → B frame (receipt clock)
    val errors = ArrayList<String>()

    fun toJson(): JSONObject = JSONObject()
        .put("firstOpened", first).put("thenTriedToOpen", second)
        .put("idleOpenOutcome", idleOpenOutcome)
        .put("sessionClosedOutcome", sessionClosedOutcome)
        .put("bothDevicesCanStayOpen", bothOpenPossible)
        .put("swapNote", swapNote)
        .put("swapSessionMs", JSONArray(swapSessionMs))
        .put("swapFirstFrameMs", JSONArray(swapFirstFrameMs))
        .put("swapGapReceiptClockMs", JSONArray(swapGapMs))
        .put("errors", JSONArray(errors))

    fun summary(): String {
        val sb = StringBuilder("▶ hold $first → open $second\n")
        sb.append("   idle-open:       $idleOpenOutcome\n")
        sb.append("   session-closed:  $sessionClosedOutcome\n")
        sb.append("   both devices can stay open: $bothOpenPossible\n")
        sb.append("   session-swap: $swapNote\n")
        if (swapGapMs.isNotEmpty()) {
            sb.append("     createCaptureSession: ${FrameSynchronizer.summarize(swapSessionMs)} ms\n")
            sb.append("     session→first frame:  ${FrameSynchronizer.summarize(swapFirstFrameMs)} ms\n")
            sb.append("     gap A→B (receipt):    ${FrameSynchronizer.summarize(swapGapMs)} ms\n")
        }
        if (errors.isNotEmpty()) sb.append("   errors: ${errors.take(5)}\n")
        return sb.toString()
    }
}

class PhaseLists {
    val openMs = ArrayList<Double>()      // includes retry waiting
    val sessionMs = ArrayList<Double>()
    val frameMs = ArrayList<Double>()     // session ready → first frame
    val closeMs = ArrayList<Double>()     // close() → onClosed
    val retries = ArrayList<Double>()

    fun toJson(): JSONObject = JSONObject()
        .put("openMs", JSONArray(openMs)).put("sessionMs", JSONArray(sessionMs))
        .put("firstFrameMs", JSONArray(frameMs)).put("closeMs", JSONArray(closeMs))
        .put("openRetries", JSONArray(retries))

    fun summary(label: String): String =
        "   $label open:${FrameSynchronizer.summarize(openMs)} | session:${FrameSynchronizer.summarize(sessionMs)} | " +
            "frame:${FrameSynchronizer.summarize(frameMs)} | close:${FrameSynchronizer.summarize(closeMs)} | retries:${FrameSynchronizer.summarize(retries)}\n"
}

/** Close-then-immediately-open flip with per-phase timing and no artificial pauses. */
class FastFlipResult(val a: String, val b: String, val cycles: Int) {
    val phasesA = PhaseLists()
    val phasesB = PhaseLists()
    val gapMs = ArrayList<Double>()
    val cycleMs = ArrayList<Double>()
    val errors = ArrayList<String>()

    fun toJson(): JSONObject = JSONObject()
        .put("idA", a).put("idB", b).put("cycles", cycles)
        .put("phasesA", phasesA.toJson()).put("phasesB", phasesB.toJson())
        .put("gapReceiptClockMs", JSONArray(gapMs)).put("cycleMs", JSONArray(cycleMs))
        .put("errors", JSONArray(errors))

    fun summary(): String {
        val sb = StringBuilder("▶ fast flip $a ↔ $b ($cycles cycles, no artificial pauses, retry-on-busy)\n")
        sb.append(phasesA.summary("A"))
        sb.append(phasesB.summary("B"))
        sb.append("   gap A→B (receipt): ${FrameSynchronizer.summarize(gapMs)} ms\n")
        sb.append("   full cycle:        ${FrameSynchronizer.summarize(cycleMs)} ms\n")
        if (errors.isNotEmpty()) sb.append("   errors: ${errors.take(5)}\n")
        return sb.toString()
    }
}

private class OpenAttempt(
    val device: CameraDevice?,
    val closed: CountDownLatch,
    val error: String?,
    val code: Int?,
    val throwable: Throwable?
)

/** One ImageReader plus a re-armable "first frame arrived" latch, so it can serve many sessions. */
private class Tap(handler: Handler, val size: Size) {
    val reader: ImageReader = ImageReader.newInstance(size.width, size.height, ImageFormat.YUV_420_888, 3)
    @Volatile var latch = CountDownLatch(1)
    val rx = AtomicLong(0)

    init {
        reader.setOnImageAvailableListener({ r ->
            var img: Image? = null
            try {
                img = r.acquireNextImage()
                if (img != null && rx.compareAndSet(0, SystemClock.elapsedRealtimeNanos())) latch.countDown()
            } catch (_: Throwable) {
            } finally {
                try { img?.close() } catch (_: Throwable) {}
            }
        }, handler)
    }

    fun arm() {
        rx.set(0)
        latch = CountDownLatch(1)
    }

    fun close() {
        try { reader.setOnImageAvailableListener(null, null) } catch (_: Throwable) {}
        try { reader.close() } catch (_: Throwable) {}
    }
}

class HoldTester(context: Context, private val coord: CameraCoordinator, private val probe: ProbeResult) {

    private val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val handler get() = coord.handler

    private fun sizeFor(id: String): Size? =
        probe.byId[id]?.let { coord.pickSize(it, ImageFormat.YUV_420_888) }

    private fun openOnce(id: String, timeoutMs: Long = 5000): OpenAttempt {
        val opened = CountDownLatch(1)
        val closed = CountDownLatch(1)
        val ref = AtomicReference<CameraDevice?>(null)
        val err = AtomicReference<String?>(null)
        val code = AtomicReference<Int?>(null)
        val cb = object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) { ref.set(camera); opened.countDown() }
            override fun onClosed(camera: CameraDevice) { closed.countDown() }
            override fun onDisconnected(camera: CameraDevice) {
                if (ref.get() == null) { err.set("onDisconnected during open"); opened.countDown() }
                try { camera.close() } catch (_: Throwable) {}
            }
            override fun onError(camera: CameraDevice, error: Int) {
                if (ref.get() == null) {
                    err.set("onError(${Diagnostics.deviceErrorName(error)})")
                    code.set(error)
                    opened.countDown()
                }
                try { camera.close() } catch (_: Throwable) {}
            }
        }
        return try {
            manager.openCamera(id, cb, handler)
            if (!opened.await(timeoutMs, TimeUnit.MILLISECONDS)) OpenAttempt(null, closed, "open timed out", null, null)
            else if (ref.get() != null) OpenAttempt(ref.get(), closed, null, null, null)
            else OpenAttempt(null, closed, err.get(), code.get(), null)
        } catch (t: Throwable) {
            OpenAttempt(null, closed, Diagnostics.describe(t), null, t)
        }
    }

    private fun isBusy(o: OpenAttempt): Boolean {
        val c = o.code
        if (c == CameraDevice.StateCallback.ERROR_MAX_CAMERAS_IN_USE || c == CameraDevice.StateCallback.ERROR_CAMERA_IN_USE) return true
        val t = o.throwable
        return t is CameraAccessException &&
            (t.reason == CameraAccessException.MAX_CAMERAS_IN_USE || t.reason == CameraAccessException.CAMERA_IN_USE)
    }

    private fun closeDevice(o: OpenAttempt?) {
        val d = o?.device ?: return
        try { d.close() } catch (_: Throwable) {}
        try { o.closed.await(2, TimeUnit.SECONDS) } catch (_: Throwable) {}
    }

    // ───────────────────────── suspend / hold-open experiment ─────────────────────────

    fun runHold(first: String, second: String, cycles: Int = 6): HoldResult {
        val r = HoldResult(first, second)
        val s1 = sizeFor(first)
        val s2 = sizeFor(second)
        if (s1 == null || s2 == null) { r.errors.add("no YUV size"); return r }

        // 1. IDLE-OPEN: first device open, no session at all.
        run {
            val o1 = openOnce(first)
            if (o1.device == null) {
                r.idleOpenOutcome = "first open failed: ${o1.error}"
            } else {
                Thread.sleep(300)
                val o2 = openOnce(second)
                r.idleOpenOutcome =
                    if (o2.device != null) "BOTH DEVICES OPEN while $first idle (no session)"
                    else "second open refused while $first idle: ${o2.error}"
                if (o2.device != null) r.bothOpenPossible = true
                closeDevice(o2)
            }
            closeDevice(o1)
            Diagnostics.log(TAG, "hold $first→$second idle-open: ${r.idleOpenOutcome}")
            Thread.sleep(coordSettle())
        }

        // 2. SESSION-CLOSED: first device streamed, then its session closed; device left open.
        run {
            val tap = Tap(handler, s1)
            val o1 = openOnce(first)
            try {
                if (o1.device == null) {
                    r.sessionClosedOutcome = "first open failed: ${o1.error}"
                } else {
                    val so = coord.createSessionBlocking(o1.device, listOf(OutputConfiguration(tap.reader.surface)))
                    val sess = so.session
                    if (sess == null) {
                        r.sessionClosedOutcome = "first session failed: ${so.error}"
                    } else {
                        tap.arm()
                        val req = o1.device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply { addTarget(tap.reader.surface) }.build()
                        sess.setRepeatingRequest(req, null, handler)
                        tap.latch.await(3, TimeUnit.SECONDS)
                        try { sess.stopRepeating() } catch (_: Throwable) {}
                        try { sess.close() } catch (_: Throwable) {}
                        Thread.sleep(400)
                        val o2 = openOnce(second)
                        r.sessionClosedOutcome =
                            if (o2.device != null) "BOTH DEVICES OPEN after $first's session was closed"
                            else "second open refused after $first's session closed (device still open): ${o2.error}"
                        if (o2.device != null) r.bothOpenPossible = true
                        closeDevice(o2)
                    }
                }
            } catch (t: Throwable) {
                r.errors.add("session-closed test: ${Diagnostics.describe(t)}")
            } finally {
                closeDevice(o1)
                tap.close()
            }
            Diagnostics.log(TAG, "hold $first→$second session-closed: ${r.sessionClosedOutcome}")
            Thread.sleep(coordSettle())
        }

        // 3. SESSION-SWAP loop, only if both devices can be held open.
        if (r.bothOpenPossible) swapLoop(first, second, s1, s2, cycles, r)
        return r
    }

    private fun coordSettle(): Long = CameraCoordinator.SETTLE_MS

    private fun swapLoop(a: String, b: String, sa: Size, sb: Size, cycles: Int, r: HoldResult) {
        val oa = openOnce(a)
        val ob = if (oa.device != null) openOnce(b) else null
        val tapA = Tap(handler, sa)
        val tapB = Tap(handler, sb)
        try {
            val devA = oa.device
            val devB = ob?.device
            if (devA == null || devB == null) {
                r.swapNote = "could not hold both open in the swap loop (A=${oa.error ?: "ok"}, B=${ob?.error ?: "ok"})"
                return
            }
            r.swapNote = "both devices held open; each cycle creates/closes one session per device"
            for (i in 1..cycles) {
                val ga = sessionGrab(devA, tapA, r.errors)
                val gb = if (ga != null) sessionGrab(devB, tapB, r.errors) else null
                if (ga == null || gb == null) { Diagnostics.log(TAG, "swap cycle $i incomplete"); continue }
                r.swapSessionMs.add(ga.first); r.swapSessionMs.add(gb.first)
                r.swapFirstFrameMs.add(ga.second); r.swapFirstFrameMs.add(gb.second)
                r.swapGapMs.add((tapB.rx.get() - tapA.rx.get()) / 1e6)
                Diagnostics.log(TAG, "swap cycle $i: A frame ${"%.0f".format(ga.second)} ms, B frame ${"%.0f".format(gb.second)} ms, gap ${"%.0f".format(r.swapGapMs.last())} ms")
            }
        } finally {
            closeDevice(oa)
            closeDevice(ob)
            tapA.close()
            tapB.close()
        }
    }

    /** Create a session on an ALREADY-OPEN device, wait for a frame, close the session. Returns (sessionMs, sessionStart→frameMs). */
    private fun sessionGrab(dev: CameraDevice, tap: Tap, errors: MutableList<String>): Pair<Double, Double>? {
        tap.arm()
        val t0 = System.nanoTime()
        val so = coord.createSessionBlocking(dev, listOf(OutputConfiguration(tap.reader.surface)))
        val sess = so.session
        if (sess == null) { errors.add("session: ${so.error}"); return null }
        val t1 = System.nanoTime()
        try {
            val req = dev.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply { addTarget(tap.reader.surface) }.build()
            sess.setRepeatingRequest(req, null, handler)
            if (!tap.latch.await(4, TimeUnit.SECONDS)) { errors.add("no frame within 4 s after session"); return null }
            val t2 = System.nanoTime()
            return Pair((t1 - t0) / 1e6, (t2 - t0) / 1e6)
        } catch (t: Throwable) {
            errors.add(Diagnostics.describe(t))
            return null
        } finally {
            try { sess.stopRepeating() } catch (_: Throwable) {}
            try { sess.close() } catch (_: Throwable) {}
        }
    }

    // ───────────────────────── fast flip with phase timing ─────────────────────────

    private class Timed(val openMs: Double, val sessionMs: Double, val frameMs: Double, val closeMs: Double, val retries: Int, val rxNs: Long, val startNs: Long, val endNs: Long)

    fun runFastFlip(a: String, b: String, cycles: Int = 6): FastFlipResult {
        val res = FastFlipResult(a, b, cycles)
        val sa = sizeFor(a)
        val sb = sizeFor(b)
        if (sa == null || sb == null) { res.errors.add("no YUV size"); return res }
        for (i in 1..cycles) {
            val ta = timedGrab(a, sa, res.errors)
            val tb = if (ta != null) timedGrab(b, sb, res.errors) else null
            if (ta == null || tb == null) { Diagnostics.log(TAG, "fast flip cycle $i incomplete"); continue }
            record(res.phasesA, ta)
            record(res.phasesB, tb)
            res.gapMs.add((tb.rxNs - ta.rxNs) / 1e6)
            res.cycleMs.add((tb.endNs - ta.startNs) / 1e6)
            Diagnostics.log(TAG, "fast flip cycle $i: A open ${"%.0f".format(ta.openMs)} sess ${"%.0f".format(ta.sessionMs)} frame ${"%.0f".format(ta.frameMs)} close ${"%.0f".format(ta.closeMs)} | B open ${"%.0f".format(tb.openMs)} (retries ${tb.retries}) | gap ${"%.0f".format(res.gapMs.last())} ms")
        }
        return res
    }

    private fun record(p: PhaseLists, t: Timed) {
        p.openMs.add(t.openMs); p.sessionMs.add(t.sessionMs); p.frameMs.add(t.frameMs)
        p.closeMs.add(t.closeMs); p.retries.add(t.retries.toDouble())
    }

    private fun timedGrab(id: String, size: Size, errors: MutableList<String>): Timed? {
        val tap = Tap(handler, size)
        var op: OpenAttempt? = null
        var sess: CameraCaptureSession? = null
        var cleaned = false

        fun cleanup(): Double {
            cleaned = true
            val c0 = System.nanoTime()
            try { sess?.stopRepeating() } catch (_: Throwable) {}
            try { sess?.close() } catch (_: Throwable) {}
            try { op?.device?.close() } catch (_: Throwable) {}
            try { op?.closed?.await(2, TimeUnit.SECONDS) } catch (_: Throwable) {}
            return (System.nanoTime() - c0) / 1e6
        }

        val tStart = System.nanoTime()
        try {
            var retries = 0
            while (true) {
                val o = openOnce(id)
                if (o.device != null) { op = o; break }
                if (!isBusy(o) || (System.nanoTime() - tStart) / 1e6 > 3000) {
                    errors.add("$id open: ${o.error}")
                    return null
                }
                retries++
                Thread.sleep(15)
            }
            val dev = op!!.device!!
            val tOpened = System.nanoTime()

            tap.arm()
            val so = coord.createSessionBlocking(dev, listOf(OutputConfiguration(tap.reader.surface)))
            val s = so.session
            if (s == null) { errors.add("$id session: ${so.error}"); return null }
            sess = s
            val tSess = System.nanoTime()

            val req = dev.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply { addTarget(tap.reader.surface) }.build()
            s.setRepeatingRequest(req, null, handler)
            if (!tap.latch.await(4, TimeUnit.SECONDS)) { errors.add("$id: no frame within 4 s"); return null }
            val tFrame = System.nanoTime()
            val rx = tap.rx.get()

            val closeMs = cleanup()
            val tEnd = System.nanoTime()
            return Timed(
                (tOpened - tStart) / 1e6, (tSess - tOpened) / 1e6, (tFrame - tSess) / 1e6,
                closeMs, retries, rx, tStart, tEnd
            )
        } catch (t: Throwable) {
            errors.add("$id: ${Diagnostics.describe(t)}")
            return null
        } finally {
            if (!cleaned) cleanup()
            tap.close()
        }
    }
}
