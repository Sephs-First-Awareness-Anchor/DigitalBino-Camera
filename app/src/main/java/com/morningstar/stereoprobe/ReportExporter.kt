// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Builds the exportable diagnostic report (JSON for machines/diffing across firmware, text for humans). */
object ReportExporter {

    fun buildJson(
        device: JSONObject?,
        probe: ProbeResult?,
        run: TestRun?,
        flips: List<FlipResult>,
        holds: List<HoldResult> = emptyList(),
        fast: List<FastFlipResult> = emptyList()
    ): JSONObject {
        val j = JSONObject()
        j.put("reportFormat", "stereoprobe/1")
        j.put("generatedAt", SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(Date()))
        j.put("authors", "Sunni (Sir) Morningstar and Cael Devo")
        j.put("device", device ?: JSONObject.NULL)
        j.put("cameraProbe", probe?.json ?: JSONObject.NULL)
        j.put("pairTestRun", run?.toJson() ?: JSONObject.NULL)
        val fl = JSONArray()
        flips.forEach { fl.put(it.toJson()) }
        j.put("flipTests", fl)
        val hl = JSONArray()
        holds.forEach { hl.put(it.toJson()) }
        j.put("holdTests", hl)
        val ff = JSONArray()
        fast.forEach { ff.put(it.toJson()) }
        j.put("fastFlipTests", ff)
        val log = JSONArray()
        Diagnostics.allLines().forEach { log.put(it) }
        j.put("log", log)
        return j
    }

    fun buildText(
        deviceSummary: String?,
        probe: ProbeResult?,
        run: TestRun?,
        flips: List<FlipResult>,
        holds: List<HoldResult> = emptyList(),
        fast: List<FastFlipResult> = emptyList()
    ): String {
        val sb = StringBuilder()
        sb.append("STEREO PROBE REPORT — authored by Sunni (Sir) Morningstar and Cael Devo\n")
        sb.append("generated ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z", Locale.US).format(Date())}\n\n")
        sb.append(deviceSummary ?: "(device not probed)\n").append('\n')
        sb.append(probe?.summary() ?: "(cameras not probed)\n").append('\n')
        sb.append(run?.summary() ?: "(pair test not run)\n").append('\n')
        if (flips.isNotEmpty()) {
            sb.append("═══ FLIP TESTS (sequential, not stereo) ═══\n")
            flips.forEach { sb.append(it.summary()).append('\n') }
        }
        if (holds.isNotEmpty() || fast.isNotEmpty()) {
            sb.append("═══ HOLD / SUSPEND TESTS ═══\n")
            holds.forEach { sb.append(it.summary()).append('\n') }
            fast.forEach { sb.append(it.summary()).append('\n') }
        }
        sb.append("═══ LOG ═══\n")
        Diagnostics.allLines().forEach { sb.append(it).append('\n') }
        return sb.toString()
    }

    /** Saves into Downloads/StereoProbe via MediaStore (no storage permission needed on API 29+). */
    fun save(context: Context, fileName: String, mime: String, bytes: ByteArray): Uri? {
        return try {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                put(MediaStore.MediaColumns.MIME_TYPE, mime)
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/StereoProbe")
            }
            val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            if (uri == null) {
                Diagnostics.log("EXPORT", "MediaStore insert returned null for $fileName")
                return null
            }
            context.contentResolver.openOutputStream(uri)?.use { it.write(bytes) }
            Diagnostics.log("EXPORT", "saved Downloads/StereoProbe/$fileName (${bytes.size} bytes)")
            uri
        } catch (t: Throwable) {
            Diagnostics.error("EXPORT", "save failed for $fileName", t)
            null
        }
    }
}
