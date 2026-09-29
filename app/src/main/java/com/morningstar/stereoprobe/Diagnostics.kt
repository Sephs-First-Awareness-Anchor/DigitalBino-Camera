// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraDevice
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Diagnostics: timestamped log + current test state. Failure is experimental data,
 * so everything that goes wrong is recorded here and included in the exported report.
 */
object Diagnostics {
    private val fmt = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)
    private val lines = ArrayList<String>()

    @Volatile var onLine: ((String) -> Unit)? = null
    @Volatile var onState: ((String) -> Unit)? = null
    @Volatile var state: String = "IDLE"
        private set

    @Synchronized
    fun log(tag: String, msg: String) {
        val line = "${fmt.format(Date())} [$tag] $msg"
        lines.add(line)
        onLine?.invoke(line)
    }

    fun error(tag: String, msg: String, t: Throwable? = null) {
        log(tag, "ERROR $msg" + (if (t != null) " :: " + describe(t) else ""))
    }

    fun setState(s: String) {
        state = s
        log("STATE", s)
        onState?.invoke(s)
    }

    @Synchronized
    fun allLines(): List<String> = ArrayList(lines)

    fun describe(t: Throwable): String {
        val base = "${t.javaClass.simpleName}: ${t.message}"
        return if (t is CameraAccessException) "$base (reason=${reasonName(t.reason)})" else base
    }

    fun reasonName(r: Int): String = when (r) {
        CameraAccessException.CAMERA_DISABLED -> "CAMERA_DISABLED"
        CameraAccessException.CAMERA_DISCONNECTED -> "CAMERA_DISCONNECTED"
        CameraAccessException.CAMERA_ERROR -> "CAMERA_ERROR"
        CameraAccessException.CAMERA_IN_USE -> "CAMERA_IN_USE"
        CameraAccessException.MAX_CAMERAS_IN_USE -> "MAX_CAMERAS_IN_USE"
        else -> "UNKNOWN($r)"
    }

    fun deviceErrorName(code: Int): String = when (code) {
        CameraDevice.StateCallback.ERROR_CAMERA_IN_USE -> "ERROR_CAMERA_IN_USE"
        CameraDevice.StateCallback.ERROR_MAX_CAMERAS_IN_USE -> "ERROR_MAX_CAMERAS_IN_USE"
        CameraDevice.StateCallback.ERROR_CAMERA_DISABLED -> "ERROR_CAMERA_DISABLED"
        CameraDevice.StateCallback.ERROR_CAMERA_DEVICE -> "ERROR_CAMERA_DEVICE"
        CameraDevice.StateCallback.ERROR_CAMERA_SERVICE -> "ERROR_CAMERA_SERVICE"
        else -> "UNKNOWN($code)"
    }
}
