// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/**
 * UI only. All camera and analysis logic lives in DeviceProbe / CameraCoordinator / FlipTester;
 * this screen just triggers them and renders their results.
 */
class MainActivity : Activity() {

    companion object {
        private const val TAG = "UI"
        private const val REQ_CAMERA = 100
    }

    private val ui = Handler(Looper.getMainLooper())
    private lateinit var stateView: TextView
    private lateinit var reportView: TextView
    private lateinit var scroll: ScrollView

    private var device: JSONObject? = null
    private var deviceSummary: String? = null
    private var probe: ProbeResult? = null
    private var run: TestRun? = null
    private val flips = ArrayList<FlipResult>()
    private val holds = ArrayList<HoldResult>()
    private val fastFlips = ArrayList<FastFlipResult>()

    private val busy = AtomicBoolean(false)
    private val refreshPending = AtomicBoolean(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        buildUi()

        Diagnostics.onLine = { scheduleRefresh() }
        Diagnostics.onState = { s -> ui.post { stateView.text = "STATE: $s" } }
        Diagnostics.log(TAG, "app start")

        if (!hasCameraPermission()) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
        }
        launch("PROBE") { ensureProbe() }
    }

    override fun onDestroy() {
        Diagnostics.onLine = null
        Diagnostics.onState = null
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAMERA) {
            val ok = grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED
            Diagnostics.log(TAG, "camera permission ${if (ok) "GRANTED" else "DENIED"}")
        }
    }

    // ───────────────────────── UI ─────────────────────────

    private fun dp(x: Int): Int = (x * resources.displayMetrics.density).toInt()

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(10), dp(36), dp(10), dp(10))
        }

        root.addView(TextView(this).apply {
            text = "STEREO PROBE · Camera Probe"
            textSize = 18f
            setTypeface(typeface, Typeface.BOLD)
        })
        root.addView(TextView(this).apply {
            text = "Can this phone see through two rear cameras at once?"
            textSize = 12f
        })

        stateView = TextView(this).apply {
            text = "STATE: IDLE"
            textSize = 13f
            setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
            setPadding(0, dp(6), 0, dp(6))
        }
        root.addView(stateView)

        root.addView(button("1 · PROBE DEVICE + CAMERAS") { launch("PROBE") { ensureProbe(force = true) } })
        root.addView(button("2 · TEST ALL CAMERA PAIRS") { doTestPairs() })
        root.addView(button("3 · FLIP TEST (sequential, NOT stereo)") { doFlipTest() })
        root.addView(button("3b · HOLD / SUSPEND + FAST FLIP TEST") { doHoldTest() })
        root.addView(button("5 · BINO SWEEP (single-camera motion stereo)") {
            startActivity(Intent(this, SweepActivity::class.java))
        })
        root.addView(button("4 · EXPORT REPORT (JSON + TXT) & SHARE") { doExport() })

        scroll = ScrollView(this)
        reportView = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            textSize = 10.5f
            setTextIsSelectable(true)
        }
        scroll.addView(reportView)
        root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        setContentView(root)
    }

    private fun button(label: String, onClick: () -> Unit): Button =
        Button(this).apply {
            text = label
            isAllCaps = false
            setOnClickListener { onClick() }
        }

    private fun scheduleRefresh() {
        if (refreshPending.compareAndSet(false, true)) {
            ui.postDelayed({
                refreshPending.set(false)
                render()
            }, 250)
        }
    }

    private fun render() {
        val sb = StringBuilder()
        deviceSummary?.let { sb.append(it).append('\n') }
        probe?.let { sb.append(it.summary()).append('\n') }
        run?.let { sb.append(it.summary()).append('\n') }
        if (flips.isNotEmpty()) {
            sb.append("═══ FLIP TESTS (sequential, not stereo) ═══\n")
            flips.forEach { sb.append(it.summary()).append('\n') }
        }
        if (holds.isNotEmpty() || fastFlips.isNotEmpty()) {
            sb.append("═══ HOLD / SUSPEND TESTS ═══\n")
            holds.forEach { sb.append(it.summary()).append('\n') }
            fastFlips.forEach { sb.append(it.summary()).append('\n') }
        }
        sb.append("═══ LOG (last 60) ═══\n")
        Diagnostics.allLines().takeLast(60).forEach { sb.append(it).append('\n') }
        reportView.text = sb.toString()
    }

    private fun toast(msg: String) = ui.post { Toast.makeText(this, msg, Toast.LENGTH_LONG).show() }

    // ───────────────────────── actions ─────────────────────────

    private fun hasCameraPermission(): Boolean =
        checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    /** Run [block] on a worker thread, one job at a time. A failure here never takes the app down. */
    private fun launch(label: String, block: () -> Unit) {
        if (!busy.compareAndSet(false, true)) {
            toast("Busy — wait for the current job to finish")
            return
        }
        Thread {
            try {
                Diagnostics.setState("$label: RUNNING")
                block()
                Diagnostics.setState("$label: DONE")
            } catch (t: Throwable) {
                Diagnostics.error(TAG, "$label crashed", t)
                Diagnostics.setState("$label: FAILED (see log)")
            } finally {
                busy.set(false)
                scheduleRefresh()
            }
        }.start()
    }

    private fun ensureProbe(force: Boolean = false): ProbeResult {
        val existing = probe
        if (existing != null && !force) return existing
        val dp = DeviceProbe(applicationContext)
        val d = dp.probeDevice()
        device = d
        deviceSummary = dp.deviceSummary(d)
        val p = dp.probeCameras()
        probe = p
        AppState.probe = p
        run = null
        flips.clear()
        holds.clear()
        fastFlips.clear()
        return p
    }

    private fun doTestPairs() {
        if (!hasCameraPermission()) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
            toast("Grant camera permission, then press again")
            return
        }
        launch("PAIR TEST") {
            val p = ensureProbe()
            val coord = CameraCoordinator(applicationContext, p)
            try {
                run = coord.testAll { step -> Diagnostics.setState("PAIR TEST: $step") }
                AppState.run = run
            } finally {
                coord.shutdown()
            }
        }
    }

    private fun doFlipTest() {
        if (!hasCameraPermission()) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
            toast("Grant camera permission, then press again")
            return
        }
        launch("FLIP TEST") {
            val p = ensureProbe()
            val pairs = p.rearRearListedPairs()
            if (pairs.isEmpty()) {
                Diagnostics.log(TAG, "flip test: fewer than two listed rear cameras, nothing to flip")
                return@launch
            }
            val coord = CameraCoordinator(applicationContext, p)
            try {
                val tester = FlipTester(coord, p)
                for ((a, b) in pairs) {
                    Diagnostics.setState("FLIP TEST: $a ↔ $b")
                    flips.add(tester.run(a, b))
                }
            } finally {
                coord.shutdown()
            }
        }
    }

    private fun doHoldTest() {
        if (!hasCameraPermission()) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
            toast("Grant camera permission, then press again")
            return
        }
        launch("HOLD TEST") {
            val p = ensureProbe()
            val pairs = p.rearRearListedPairs()
            if (pairs.isEmpty()) {
                Diagnostics.log(TAG, "hold test: fewer than two listed rear cameras")
                return@launch
            }
            val coord = CameraCoordinator(applicationContext, p)
            try {
                val tester = HoldTester(applicationContext, coord, p)
                for ((a, b) in pairs) {
                    // Both orders: the earlier pair test only tried A first, then B.
                    Diagnostics.setState("HOLD TEST: hold $a, open $b")
                    holds.add(tester.runHold(a, b))
                    Diagnostics.setState("HOLD TEST: hold $b, open $a")
                    holds.add(tester.runHold(b, a))
                    Diagnostics.setState("HOLD TEST: fast flip $a ↔ $b")
                    fastFlips.add(tester.runFastFlip(a, b))
                }
            } finally {
                coord.shutdown()
            }
        }
    }

    private fun doExport() {
        launch("EXPORT") {
            val json = ReportExporter.buildJson(device, probe, run, flips, holds, fastFlips)
            val text = ReportExporter.buildText(deviceSummary, probe, run, flips, holds, fastFlips)
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val model = Build.MODEL.replace(Regex("[^A-Za-z0-9]+"), "_")
            val base = "stereoprobe_${model}_$stamp"
            val ju = ReportExporter.save(applicationContext, "$base.json", "application/json", json.toString(2).toByteArray())
            ReportExporter.save(applicationContext, "$base.txt", "text/plain", text.toByteArray())
            if (ju != null) {
                toast("Saved to Downloads/StereoProbe/$base.(json|txt)")
                ui.post { share(ju) }
            } else {
                toast("Export failed — see log")
            }
        }
    }

    private fun share(uri: Uri) {
        val i = Intent(Intent.ACTION_SEND).apply {
            type = "application/json"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(i, "Share Stereo Probe report"))
    }
}
