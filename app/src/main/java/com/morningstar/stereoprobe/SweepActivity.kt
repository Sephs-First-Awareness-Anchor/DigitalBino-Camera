// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.Manifest
import android.app.Activity
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Bino Sweep screen. UI only: capture logic lives in MotionBaselineProvider and all stereo
 * interpretation in StereoPipeline. The UI avoids camera/HAL vocabulary; technical detail is in the log and export.
 */
class SweepActivity : Activity(), SweepListener {

    companion object {
        private const val TAG = "SWEEPUI"
        private const val REQ_CAMERA = 200
    }

    private enum class ViewMode { LIVE, EYE_A, EYE_B, RECT, DISP, DEPTH }

    private val ui = Handler(Looper.getMainLooper())
    private lateinit var live: ImageView
    private lateinit var ghost: ImageView
    private lateinit var hud: TextView
    private lateinit var progress: TextView
    private lateinit var baselineLabel: TextView
    private lateinit var captureBtn: Button
    private lateinit var resultRow: LinearLayout

    private var provider: MotionBaselineProvider? = null
    private var probe: ProbeResult? = null
    private var targetCm = 6.0
    private var eyeALocked = false

    private var result: StereoAcquisitionResult? = null
    private var output: StereoOutput? = null
    private var mode = ViewMode.LIVE
    private val bitmaps = HashMap<ViewMode, Bitmap>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        buildUi()

        if (!OpenCVLoader.initLocal()) {
            hud.text = "OpenCV failed to load; Bino Sweep cannot run."
            captureBtn.isEnabled = false
            return
        }
        Thread {
            val p = AppState.probe ?: DeviceProbe(applicationContext).probeCameras().also { AppState.probe = it }
            probe = p
            val method = AcquisitionSelector.select(AppState.run)
            Diagnostics.log(TAG, "acquisition method selected: ${method.label}")
            if (method == AcquisitionMethod.CONCURRENT_CAMERAS) {
                Diagnostics.log(TAG, "a validated simultaneous pair exists, but its provider is not implemented yet; using Motion-Baseline")
            }
            ui.post { ensureCameraThenStart() }
        }.start()
    }

    override fun onDestroy() {
        provider?.stop()
        output?.release()
        result?.release()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAMERA && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startProvider()
        } else if (requestCode == REQ_CAMERA) {
            hud.text = "Camera permission is required."
        }
    }

    private fun ensureCameraThenStart() {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startProvider()
        else requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
    }

    private fun startProvider() {
        val p = probe ?: return
        provider?.stop()
        eyeALocked = false
        ghost.setImageDrawable(null)
        val prov = MotionBaselineProvider(applicationContext, p, this)
        prov.setTargetBaselineCm(targetCm)
        provider = prov
        prov.start()
    }

    // ───────────────────────── UI construction ─────────────────────────

    private fun dp(x: Int): Int = (x * resources.displayMetrics.density).toInt()

    private fun btn(label: String, onClick: () -> Unit): Button =
        Button(this).apply {
            text = label
            isAllCaps = false
            textSize = 12f
            setOnClickListener { onClick() }
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }

    private fun buildUi() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }

        live = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER }
        ghost = ImageView(this).apply {
            scaleType = ImageView.ScaleType.FIT_CENTER
            alpha = 0.35f
        }
        root.addView(live, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        root.addView(ghost, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))

        val top = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.argb(150, 0, 0, 0))
            setPadding(dp(10), dp(30), dp(10), dp(8))
        }
        hud = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 17f
            setTypeface(typeface, Typeface.BOLD)
            text = "Starting camera…"
        }
        progress = TextView(this).apply {
            setTextColor(Color.rgb(160, 230, 255))
            textSize = 12f
            typeface = Typeface.MONOSPACE
        }
        top.addView(hud)
        top.addView(progress)
        root.addView(top, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP))

        val bottom = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.argb(150, 0, 0, 0))
            setPadding(dp(6), dp(4), dp(6), dp(10))
        }
        val row1 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        baselineLabel = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 12f
            gravity = Gravity.CENTER
            text = "6.0 cm"
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        captureBtn = btn("CAPTURE") { provider?.captureFirstEye() }
        row1.addView(btn("−") { changeBaseline(-0.5) })
        row1.addView(baselineLabel)
        row1.addView(btn("+") { changeBaseline(0.5) })
        row1.addView(captureBtn)
        row1.addView(btn("RESET") { resetAll() })

        resultRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            visibility = View.GONE
        }
        resultRow.addView(btn("Eye A") { show(ViewMode.EYE_A) })
        resultRow.addView(btn("Eye B") { show(ViewMode.EYE_B) })
        resultRow.addView(btn("Rect") { show(ViewMode.RECT) })
        resultRow.addView(btn("Disp") { show(ViewMode.DISP) })
        resultRow.addView(btn("Depth") { show(ViewMode.DEPTH) })
        resultRow.addView(btn("Export") { export() })

        bottom.addView(row1)
        bottom.addView(resultRow)
        root.addView(bottom, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM))

        live.setOnTouchListener { _, ev ->
            if (ev.action == MotionEvent.ACTION_UP) queryDepth(ev.x, ev.y)
            true
        }
        setContentView(root)
    }

    private fun changeBaseline(deltaCm: Double) {
        targetCm = (targetCm + deltaCm).coerceIn(3.0, 12.0)
        baselineLabel.text = "${"%.1f".format(targetCm)} cm"
        provider?.setTargetBaselineCm(targetCm)
    }

    private fun resetAll() {
        output?.release(); output = null
        result?.release(); result = null
        bitmaps.clear()
        mode = ViewMode.LIVE
        resultRow.visibility = View.GONE
        captureBtn.isEnabled = true
        startProvider()
    }

    // ───────────────────────── SweepListener (called on worker threads) ─────────────────────────

    override fun onLiveFrame(frame: Bitmap) {
        ui.post { if (mode == ViewMode.LIVE) live.setImageBitmap(frame) }
    }

    override fun onStatus(status: SweepStatus) {
        ui.post {
            if (mode != ViewMode.LIVE) return@post
            hud.text = status.headline + if (status.detail.isNotEmpty()) "\n" + status.detail else ""
            hud.setTextColor(if (status.orientationOk) Color.WHITE else Color.rgb(255, 200, 80))
            progress.text = if (eyeALocked) {
                val n = 12
                val frac = (status.baselineCm / status.targetCm).coerceIn(0.0, 1.0)
                val pos = (frac * n).toInt()
                val bar = StringBuilder("Eye A ●")
                for (i in 0 until n) bar.append(if (i == pos) '◦' else '─')
                bar.append("○ Eye B")
                bar.toString() + "\n${"%.1f".format(status.baselineCm)} cm / ${"%.1f".format(status.targetCm)} cm" +
                    "   overlap ${"%.0f".format(status.overlapPct)}%" +
                    "\npitch ${"%.1f".format(status.pitchDeg)}°  yaw ${"%.1f".format(status.yawDeg)}°  roll ${"%.1f".format(status.rollDeg)}°" +
                    "   inliers ${status.inliers}  parallax ${"%.1f".format(status.parallaxPx)}px"
            } else ""
        }
    }

    override fun onEyeALocked(ghost: Bitmap) {
        ui.post {
            eyeALocked = true
            this.ghost.setImageBitmap(ghost)
            captureBtn.isEnabled = false
            live.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
        }
    }

    override fun onEyeBLocked() {
        ui.post {
            live.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
            hud.text = "Second view locked. Processing…"
        }
    }

    override fun onResult(result: StereoAcquisitionResult) {
        this.result = result
        provider?.stop()
        provider = null
        ui.post { hud.text = "Building depth from the two views…" }
        Thread {
            try {
                val out = StereoPipeline().process(result)
                output = out
                bitmaps[ViewMode.EYE_A] = Bmp.fromBgr(result.first)
                bitmaps[ViewMode.EYE_B] = Bmp.fromBgr(result.second)
                bitmaps[ViewMode.RECT] = Bmp.fromBgr(out.rectPairLines)
                bitmaps[ViewMode.DISP] = Bmp.fromBgr(out.disparityColor)
                bitmaps[ViewMode.DEPTH] = Bmp.fromBgr(out.depthColor)
                ui.post {
                    resultRow.visibility = View.VISIBLE
                    ghost.setImageDrawable(null)
                    show(ViewMode.DISP)
                    val d = out.diagnostics
                    progress.text = "baseline ${"%.1f".format(out.baselineM * 100)} cm (${"%.0f".format(result.baselineConfidence * 100)}% conf)  " +
                        "valid ${"%.0f".format(d.optDouble("validDisparityFraction") * 100)}%  " +
                        "epipolar err ${"%.1f".format(d.optDouble("rectifiedEpipolarErrorMedianPx"))}px"
                }
            } catch (t: Throwable) {
                Diagnostics.error(TAG, "stereo processing failed", t)
                ui.post {
                    resultRow.visibility = View.VISIBLE
                    hud.text = "Could not build depth: ${t.message}"
                    bitmaps[ViewMode.EYE_A] = Bmp.fromBgr(result.first)
                    bitmaps[ViewMode.EYE_B] = Bmp.fromBgr(result.second)
                }
            }
        }.start()
    }

    override fun onFailure(message: String) {
        Diagnostics.log(TAG, "failure: $message")
        ui.post {
            live.performHapticFeedback(HapticFeedbackConstants.REJECT)
            hud.text = message
            hud.setTextColor(Color.rgb(255, 120, 120))
            captureBtn.isEnabled = true
        }
    }

    // ───────────────────────── result viewing ─────────────────────────

    private fun show(m: ViewMode) {
        val bmp = bitmaps[m] ?: return
        mode = m
        ghost.visibility = View.GONE
        live.setImageBitmap(bmp)
        val d = output?.diagnostics
        hud.setTextColor(Color.WHITE)
        hud.text = when (m) {
            ViewMode.EYE_A -> "Eye A (first view)"
            ViewMode.EYE_B -> "Eye B (second view)"
            ViewMode.RECT -> "Rectified pair: features should share a green line"
            ViewMode.DISP -> "Disparity (warm = closer). Tap for distance."
            ViewMode.DEPTH -> "Depth (warm = closer). Tap for distance."
            else -> ""
        }
        if (d != null && (m == ViewMode.DISP || m == ViewMode.DEPTH)) {
            hud.text = hud.text.toString() + "\nMetric scale depends on the movement estimate"
        }
    }

    private fun queryDepth(x: Float, y: Float) {
        if (mode != ViewMode.DISP && mode != ViewMode.DEPTH) return
        val out = output ?: return
        val inv = Matrix()
        if (!live.imageMatrix.invert(inv)) return
        val pts = floatArrayOf(x, y)
        inv.mapPoints(pts)
        val bx = pts[0].toInt()
        val by = pts[1].toInt()
        val z = out.depthAt(bx, by)
        hud.text = if (z != null) "≈ ${"%.2f".format(z)} m at ($bx, $by)\n(scale from the movement estimate; relative structure is unaffected)"
        else "No reliable depth at ($bx, $by)"
    }

    private fun export() {
        val res = result ?: return
        Thread {
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val base = "sweep_$stamp"
            fun png(name: String, m: Mat?) {
                if (m == null) return
                val buf = MatOfByte()
                Imgcodecs.imencode(".png", m, buf)
                ReportExporter.save(applicationContext, "${base}_$name.png", "image/png", buf.toArray())
                buf.release()
            }
            png("eyeA", res.first)
            png("eyeB", res.second)
            val out = output
            png("rectLeft", out?.rectLeft)
            png("rectRight", out?.rectRight)
            png("rectPairLines", out?.rectPairLines)
            png("disparity", out?.disparityColor)
            png("depth", out?.depthColor)
            val json = JSONObject()
                .put("reportFormat", "binosweep/1")
                .put("authors", "Sunni (Sir) Morningstar and Cael Devo")
                .put("acquisition", res.quality)
                .put("stereoPipeline", out?.diagnostics ?: JSONObject.NULL)
                .put("baselineConfidence", res.baselineConfidence)
                .put("captureConfidence", res.captureConfidence)
            val lines = org.json.JSONArray()
            Diagnostics.allLines().forEach { lines.put(it) }
            json.put("log", lines)
            ReportExporter.save(applicationContext, "${base}_diagnostics.json", "application/json", json.toString(2).toByteArray())
            ui.post { Toast.makeText(this, "Saved to Downloads/StereoProbe/${base}_*", Toast.LENGTH_LONG).show() }
        }.start()
    }
}
