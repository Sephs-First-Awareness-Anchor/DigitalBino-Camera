// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.sqrt

/** Tiny 3x3 row-major helpers (DoubleArray of 9). */
object Mat3 {
    fun mul(a: DoubleArray, b: DoubleArray): DoubleArray {
        val o = DoubleArray(9)
        for (i in 0..2) for (j in 0..2) {
            var s = 0.0
            for (k in 0..2) s += a[i * 3 + k] * b[k * 3 + j]
            o[i * 3 + j] = s
        }
        return o
    }

    fun transpose(a: DoubleArray): DoubleArray =
        doubleArrayOf(a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8])

    fun mulVec(a: DoubleArray, v: DoubleArray): DoubleArray =
        doubleArrayOf(
            a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
            a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
            a[6] * v[0] + a[7] * v[1] + a[8] * v[2]
        )

    /** Small-angle rotation vector (right-handed, about the axes of the frame the matrix is expressed in). */
    fun smallAngleVec(r: DoubleArray): DoubleArray =
        doubleArrayOf(0.5 * (r[7] - r[5]), 0.5 * (r[2] - r[6]), 0.5 * (r[3] - r[1]))

    /** Total rotation angle in degrees. */
    fun angleDeg(r: DoubleArray): Double {
        val c = ((r[0] + r[4] + r[8]) - 1.0) / 2.0
        return Math.toDegrees(acos(c.coerceIn(-1.0, 1.0)))
    }

    /**
     * Device axes (X right, Y up, Z out of screen) → upright camera axes (x right, y down, z forward).
     * The back camera looks along -Z, so C = diag(1, -1, -1). Rotation between frames: R_cam = C R_dev C.
     */
    fun devToCam(rDev: DoubleArray): DoubleArray {
        val c = doubleArrayOf(1.0, -1.0, -1.0)
        val o = DoubleArray(9)
        for (i in 0..2) for (j in 0..2) o[i * 3 + j] = c[i] * c[j] * rDev[i * 3 + j]
        return o
    }

    fun norm(v: DoubleArray): Double = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    fun dot(a: DoubleArray, b: DoubleArray): Double = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/**
 * Rotation vector + linear acceleration + gyro, with time-indexed buffers.
 * Sensor timestamps share the elapsedRealtime clock with camera frames when the camera timestamp source is REALTIME.
 *
 * Translation here is a double integration of linear acceleration from a bias-corrected rest state.
 * It is a coarse estimate that drifts quickly; it is always reported with that caveat.
 */
class MotionSensors(context: Context) : SensorEventListener {

    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    // Prefer the game rotation vector (gyro + accelerometer, no magnetometer): far less jitter near metal and electronics.
    private val rotSensor: Sensor? =
        sm.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR) ?: sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    val rotationSourceName: String
        get() = when (rotSensor?.type) {
            Sensor.TYPE_GAME_ROTATION_VECTOR -> "GAME_ROTATION_VECTOR"
            Sensor.TYPE_ROTATION_VECTOR -> "ROTATION_VECTOR"
            else -> "none"
        }
    private val linSensor: Sensor? = sm.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    private val gyroSensor: Sensor? = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    private val thread = HandlerThread("sweep-sensors").also { it.start() }
    private val handler = Handler(thread.looper)

    val usable: Boolean get() = rotSensor != null && linSensor != null
    fun describe(): String = "rotation=$rotationSourceName linearAcceleration=${linSensor != null} gyroscope=${gyroSensor != null}"

    private class RotSample(val tNs: Long, val r: DoubleArray)
    private class AccSample(val tNs: Long, val a: DoubleArray)

    private val lock = Any()
    private val rotBuf = ArrayList<RotSample>()
    private val accBuf = ArrayList<AccSample>()
    private val rmat = FloatArray(9)

    private var integrating = false
    private var lastT = 0L
    private val bias = DoubleArray(3)
    private val vel = DoubleArray(3)
    private val pos = DoubleArray(3)

    @Volatile private var gyroSpeedRadS = 0.0
    fun gyroSpeed(): Double = gyroSpeedRadS

    fun start() {
        rotSensor?.let { sm.registerListener(this, it, 10_000, handler) }
        linSensor?.let { sm.registerListener(this, it, 10_000, handler) }
        gyroSensor?.let { sm.registerListener(this, it, 20_000, handler) }
    }

    fun stop() {
        try { sm.unregisterListener(this) } catch (_: Throwable) {}
        thread.quitSafely()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR, Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                SensorManager.getRotationMatrixFromVector(rmat, event.values)
                val r = DoubleArray(9) { rmat[it].toDouble() }
                synchronized(lock) {
                    rotBuf.add(RotSample(event.timestamp, r))
                    trim(rotBuf.size) { rotBuf.removeAt(0) }
                    while (rotBuf.size > 1 && event.timestamp - rotBuf[0].tNs > 4_000_000_000L) rotBuf.removeAt(0)
                }
            }
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                val a = doubleArrayOf(event.values[0].toDouble(), event.values[1].toDouble(), event.values[2].toDouble())
                synchronized(lock) {
                    accBuf.add(AccSample(event.timestamp, a))
                    while (accBuf.size > 1 && event.timestamp - accBuf[0].tNs > 4_000_000_000L) accBuf.removeAt(0)
                    if (integrating) integrate(event.timestamp, a)
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                val x = event.values[0].toDouble(); val y = event.values[1].toDouble(); val z = event.values[2].toDouble()
                gyroSpeedRadS = sqrt(x * x + y * y + z * z)
            }
        }
    }

    private inline fun trim(size: Int, drop: () -> Unit) {
        if (size > 2000) drop()
    }

    private fun integrate(tNs: Long, aDev: DoubleArray) {
        val last = rotBuf.lastOrNull() ?: return
        val dt = if (lastT == 0L) 0.0 else (tNs - lastT) / 1e9
        lastT = tNs
        if (dt <= 0.0 || dt > 0.1) return
        val corrected = doubleArrayOf(aDev[0] - bias[0], aDev[1] - bias[1], aDev[2] - bias[2])
        val aW = Mat3.mulVec(last.r, corrected)
        for (i in 0..2) {
            vel[i] += aW[i] * dt
            pos[i] += vel[i] * dt
        }
    }

    /** Start integrating displacement at [atNs]. Bias = mean device-frame linear acceleration over the preceding 0.5 s. */
    fun beginIntegration(atNs: Long) {
        synchronized(lock) {
            // Prefer a rest window ending 0.3 s before Eye A so the finger tap on the screen is not in the bias estimate.
            var recent = accBuf.filter { it.tNs in (atNs - 900_000_000L)..(atNs - 300_000_000L) }
            if (recent.size < 20) recent = accBuf.filter { it.tNs in (atNs - 500_000_000L)..atNs }
            if (recent.size >= 5) {
                for (i in 0..2) bias[i] = recent.sumOf { it.a[i] } / recent.size
            } else {
                for (i in 0..2) bias[i] = 0.0
            }
            for (i in 0..2) { vel[i] = 0.0; pos[i] = 0.0 }
            lastT = 0L
            integrating = true
        }
    }

    fun endIntegration() {
        synchronized(lock) { integrating = false }
    }

    /** Integrated displacement in the world frame, metres, since [beginIntegration]. */
    fun displacementWorld(): DoubleArray = synchronized(lock) { pos.copyOf() }

    fun biasSnapshot(): DoubleArray = synchronized(lock) { bias.copyOf() }

    /** Integrated world-frame velocity since [beginIntegration]. */
    fun velocityWorld(): DoubleArray = synchronized(lock) { vel.copyOf() }

    /** RMS of bias-corrected linear acceleration over the last ~8 samples (m/s^2): small means the phone is being held still. */
    fun recentAccRms(): Double = synchronized(lock) {
        val n = minOf(8, accBuf.size)
        if (n == 0) return@synchronized 0.0
        var s = 0.0
        for (i in accBuf.size - n until accBuf.size) {
            val a = accBuf[i].a
            val dx = a[0] - bias[0]; val dy = a[1] - bias[1]; val dz = a[2] - bias[2]
            s += dx * dx + dy * dy + dz * dz
        }
        sqrt(s / n)
    }

    /** Device→world rotation nearest to [tNs], or null if nothing within 200 ms. */
    fun rotationAt(tNs: Long): DoubleArray? = synchronized(lock) {
        var best: RotSample? = null
        var bestDt = Long.MAX_VALUE
        for (s in rotBuf) {
            val d = abs(s.tNs - tNs)
            if (d < bestDt) { bestDt = d; best = s }
        }
        if (best != null && bestDt <= 200_000_000L) best.r else null
    }
}
