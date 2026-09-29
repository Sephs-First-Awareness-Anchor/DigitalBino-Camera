// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import java.io.File
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

private const val TAG = "PREDICT"

/** A guess for eye B. Built WITHOUT access to eye B: predict() is never given it. */
class Prediction(
    val guess: Mat,                     // BGR, half resolution, holes inpainted
    val validMask: Mat,                 // 8U, 255 where a source pixel actually landed (the rest is inpainted guesswork)
    val coverage: Double,
    val weightsUsed: DoubleArray,
    val metrics: SensorMetrics,
    val cam: ViewSynth.Cam,
    val scale: Double,
    val policy: String                  // INSTRUCTED (slide right by the requested baseline) or IMU (signed accelerometer displacement)
) {
    fun release() { guess.release(); validMask.release() }
}

class Reveal(
    val nPoints: Int,
    val guessPx: Double, val gyroOnlyPx: Double, val noChangePx: Double, val flippedSignGuessPx: Double,
    val photoGuess: Double, val photoGyroOnly: Double, val photoNoChange: Double,
    val coverage: Double,
    val signLikelyWrong: Boolean,
    val learnedFromCaptures: Int,
    val weightsBefore: DoubleArray, val weightsAfter: DoubleArray,
    val trend: String,
    val errorImage: Mat,
    val policyUsed: String = "",
    val instructedPx: Double = Double.NaN,
    val imuPx: Double = Double.NaN,
    val userSlidLeft: Boolean = false
) {
    fun toJson(): JSONObject = JSONObject()
        .put("correspondencesUsed", nPoints)
        .put("featureErrorPx_guess", guessPx).put("featureErrorPx_gyroOnly", gyroOnlyPx)
        .put("featureErrorPx_noChange", noChangePx).put("featureErrorPx_guessWithFlippedSign", flippedSignGuessPx)
        .put("photometricError_guess", photoGuess).put("photometricError_gyroOnly", photoGyroOnly)
        .put("photometricError_noChange", photoNoChange)
        .put("guessCoverage", coverage)
        .put("accelerometerSignLikelyWrong", signLikelyWrong)
        .put("policyUsed", policyUsed).put("featureErrorPx_instructedPolicy", instructedPx)
        .put("featureErrorPx_imuPolicy", imuPx).put("userSlidLeft", userSlidLeft)
        .put("learnedFromCaptures", learnedFromCaptures)
        .put("weightsBefore", JSONArray(weightsBefore.toList())).put("weightsAfter", JSONArray(weightsAfter.toList()))

    fun summary(): String {
        val sb = StringBuilder()
        sb.append("Feature error (px, lower is better):\n  guess ${"%.1f".format(guessPx)}   gyro-only ${"%.1f".format(gyroOnlyPx)}   no change ${"%.1f".format(noChangePx)}\n")
        sb.append("Image error (local-contrast units):\n  guess ${"%.2f".format(photoGuess)}   gyro-only ${"%.2f".format(photoGyroOnly)}   no change ${"%.2f".format(photoNoChange)}\n")
        sb.append("Guess filled ${"%.0f".format(coverage * 100)}% from real pixels; the rest is inpainted\n")
        sb.append("Direction policy used: $policyUsed  (instructed ${"%.1f".format(instructedPx)} px, accelerometer ${"%.1f".format(imuPx)} px)\n")
        if (userSlidLeft) sb.append("You slid LEFT this time; learned with the left direction\n")
        if (signLikelyWrong) sb.append("Accelerometer direction looked WRONG this time (flipped-sign guess: ${"%.1f".format(flippedSignGuessPx)} px); that capture was learned with the corrected sign\n")
        sb.append("Learned from $learnedFromCaptures capture${if (learnedFromCaptures == 1) "" else "s"}\n")
        if (trend.isNotEmpty()) sb.append(trend)
        return sb.toString()
    }
}

/** Persistent learner state: the depth-prior weights, a replay buffer of past captures, and the score history. */
class PredictorStore(context: Context) {
    private val file = File(context.filesDir, "view_predictor.json")
    var w: DoubleArray = ViewSynth.FLAT_PRIOR.copyOf()
    val samples = ArrayList<ViewSynth.Sample>()
    val history = ArrayList<JSONObject>()
    val policyErrors = hashMapOf("INSTRUCTED" to ArrayList<Double>(), "IMU" to ArrayList<Double>())

    init { load() }

    /** Which way of getting the camera's sideways movement predicts better? Decided by the last few captures; starts with the instruction. */
    fun choosePolicy(): String {
        fun med(k: String): Double { val l = policyErrors[k]!!.takeLast(8); return if (l.size < 2) Double.NaN else GyroPoseSolver.median(l.toDoubleArray()) }
        val i = med("INSTRUCTED"); val m = med("IMU")
        return if (!i.isNaN() && !m.isNaN() && m < 0.8 * i) "IMU" else "INSTRUCTED"
    }

    private fun load() {
        try {
            if (!file.exists()) return
            val j = JSONObject(file.readText())
            val wa = j.getJSONArray("w")
            if (wa.length() == ViewSynth.NFEAT) w = DoubleArray(ViewSynth.NFEAT) { wa.getDouble(it) }
            val sa = j.getJSONArray("samples")
            for (i in 0 until sa.length()) {
                val s = sa.getJSONObject(i)
                fun arr(name: String): DoubleArray { val a = s.getJSONArray(name); return DoubleArray(a.length()) { a.getDouble(it) } }
                samples.add(ViewSynth.Sample(arr("a"), arr("b"), s.getDouble("focus"), arr("rot"), arr("c")))
            }
            for (k in listOf("INSTRUCTED", "IMU")) {
                val pa = j.optJSONArray("policy_$k")
                if (pa != null) for (i in 0 until pa.length()) policyErrors[k]!!.add(pa.getDouble(i))
            }
            val ha = j.getJSONArray("history")
            for (i in 0 until ha.length()) history.add(ha.getJSONObject(i))
        } catch (t: Throwable) {
            Diagnostics.error(TAG, "could not load predictor state; starting fresh", t)
        }
    }

    fun save() {
        try {
            val j = JSONObject()
            j.put("w", JSONArray(w.toList()))
            val sa = JSONArray()
            for (s in samples) {
                sa.put(JSONObject()
                    .put("a", JSONArray(s.a.map { Math.round(it * 10) / 10.0 }))
                    .put("b", JSONArray(s.b.map { Math.round(it * 10) / 10.0 }))
                    .put("focus", s.focusM).put("rot", JSONArray(s.rot.toList())).put("c", JSONArray(s.c.toList())))
            }
            j.put("samples", sa)
            for (k in listOf("INSTRUCTED", "IMU")) j.put("policy_$k", JSONArray(policyErrors[k]!!.takeLast(60)))
            val ha = JSONArray(); history.takeLast(100).forEach { ha.put(it) }
            j.put("history", ha)
            file.writeText(j.toString())
        } catch (t: Throwable) {
            Diagnostics.error(TAG, "could not save predictor state", t)
        }
    }

    fun reset() {
        w = ViewSynth.FLAT_PRIOR.copyOf(); samples.clear(); history.clear(); policyErrors.values.forEach { it.clear() }; save()
    }

    fun trendText(): String {
        val h = history.takeLast(6)
        if (h.size < 2) return ""
        fun f(key: String) = h.joinToString(" → ") { "%.0f".format(it.optDouble(key)) }
        return "Trend over last ${h.size} captures (px):\n  guess ${f("guessPx")}\n  gyro-only ${f("gyroOnlyPx")}\n"
    }
}

/**
 * The app IS the model: exact projective physics + a small learned depth prior that the app fits to its own captures.
 * Flow per capture:  predict(eye A, sensor metrics) -> [eye B is revealed] -> reveal(): score against honest baselines, then learn.
 */
class ViewPredictor(context: Context) {

    val store = PredictorStore(context)

    fun predict(eyeA: Mat, k: Intrinsics, m: SensorMetrics): Prediction {
        val cam = ViewSynth.Cam(k.fx, k.fy, k.cx, k.cy, k.width.toDouble(), k.height.toDouble())
        val scale = 0.5
        val focus = m.focusDistanceM ?: 0.8
        val policy = store.choosePolicy()
        // The app asks for a rightward slide, so the camera-B centre is +x. Magnitude: the requested baseline, or the accelerometer's value.
        val c = if (policy == "IMU") doubleArrayOf(m.imuLateralM, 0.0, 0.0) else doubleArrayOf(m.targetBaselineM, 0.0, 0.0)
        val w = store.w.copyOf()

        val small = Mat()
        Imgproc.resize(eyeA, small, Size(eyeA.cols() * scale, eyeA.rows() * scale), 0.0, 0.0, Imgproc.INTER_AREA)
        val sw = small.cols()
        val sh = small.rows()
        val src = ByteArray(sw * sh * 3)
        small.get(0, 0, src)
        val dst = ByteArray(sw * sh * 3)
        val zbuf = FloatArray(sw * sh) { Float.MAX_VALUE }
        val scratch = DoubleArray(ViewSynth.NFEAT)
        val p = DoubleArray(2)

        for (y in 0 until sh) {
            for (x in 0 until sw) {
                val u = (x + 0.5) / scale
                val v = (y + 0.5) / scale
                val nx = (u - cam.width / 2) / (cam.width / 2)
                val ny = (v - cam.height / 2) / (cam.height / 2)
                val invZ = ViewSynth.invZ(w, focus, nx, ny, scratch)
                if (!ViewSynth.project(u, v, invZ, m.gyroRotationCam, c, cam, p)) continue
                val tx = p[0] * scale - 0.5
                val ty = p[1] * scale - 0.5
                val ix = floor(tx).toInt()
                val iy = floor(ty).toInt()
                val depth = (1.0 / invZ).toFloat()
                val si = (y * sw + x) * 3
                for (dy in 0..1) for (dx in 0..1) {
                    val px = ix + dx
                    val py = iy + dy
                    if (px < 0 || py < 0 || px >= sw || py >= sh) continue
                    val idx = py * sw + px
                    if (depth < zbuf[idx]) {
                        zbuf[idx] = depth
                        dst[idx * 3] = src[si]; dst[idx * 3 + 1] = src[si + 1]; dst[idx * 3 + 2] = src[si + 2]
                    }
                }
            }
        }

        val maskBytes = ByteArray(sw * sh) { if (zbuf[it] < Float.MAX_VALUE) 255.toByte() else 0 }
        val guess = Mat(sh, sw, CvType.CV_8UC3)
        guess.put(0, 0, dst)
        val valid = Mat(sh, sw, CvType.CV_8UC1)
        valid.put(0, 0, maskBytes)
        val coverage = Core.countNonZero(valid).toDouble() / (sw.toDouble() * sh)

        // Where nothing landed (disocclusions, image edges) the guess is filled in: honest about being a guess, and reported as coverage.
        val holes = Mat()
        Core.bitwise_not(valid, holes)
        val filled = Mat()
        Photo.inpaint(guess, holes, filled, 3.0, Photo.INPAINT_TELEA)
        small.release(); guess.release(); holes.release()
        Diagnostics.log(TAG, "guess for B built from eye A + sensors only: policy=$policy focus=${"%.2f".format(focus)}m lateral=${"%.1f".format(m.imuLateralM * 100)}cm coverage=${"%.0f".format(coverage * 100)}% w=${w.joinToString(",") { "%.2f".format(it) }}")
        return Prediction(filled, valid, coverage, w, m, cam, scale, policy)
    }

    /** Eye B is revealed HERE, after the guess exists. Scores the guess, then teaches the model. */
    fun reveal(pred: Prediction, eyeA: Mat, eyeB: Mat): Reveal {
        val cam = pred.cam
        val m = pred.metrics
        val focus = m.focusDistanceM ?: 0.8
        val (a, b) = correspondences(eyeA, eyeB)
        val c = if (pred.policy == "IMU") doubleArrayOf(m.imuLateralM, 0.0, 0.0) else doubleArrayOf(m.targetBaselineM, 0.0, 0.0)
        val sample = ViewSynth.Sample(a, b, focus, m.gyroRotationCam, c)
        val instrS = ViewSynth.Sample(a, b, focus, m.gyroRotationCam, doubleArrayOf(m.targetBaselineM, 0.0, 0.0))
        val instrLeftS = ViewSynth.Sample(a, b, focus, m.gyroRotationCam, doubleArrayOf(-m.targetBaselineM, 0.0, 0.0))
        val imuS = ViewSynth.Sample(a, b, focus, m.gyroRotationCam, doubleArrayOf(m.imuLateralM, 0.0, 0.0))
        val instrPx = if (a.size >= 40) ViewSynth.medianError(ViewSynth.predictPoints(instrS, pred.weightsUsed, cam), b) else Double.NaN
        val instrLeftPx = if (a.size >= 40) ViewSynth.medianError(ViewSynth.predictPoints(instrLeftS, pred.weightsUsed, cam), b) else Double.NaN
        val imuPolicyPx = if (a.size >= 40) ViewSynth.medianError(ViewSynth.predictPoints(imuS, pred.weightsUsed, cam), b) else Double.NaN
        val userSlidLeft = a.size >= 40 && instrLeftPx < 0.6 * instrPx && instrPx > 8.0
        val flipped = ViewSynth.Sample(a, b, focus, m.gyroRotationCam, doubleArrayOf(-c[0], 0.0, 0.0))

        val guessPx = if (a.size >= 40) ViewSynth.medianError(ViewSynth.predictPoints(sample, pred.weightsUsed, cam), b) else Double.NaN
        val gyroPx = if (a.size >= 40) ViewSynth.medianError(ViewSynth.rotationOnlyPoints(sample, cam), b) else Double.NaN
        val idPx = if (a.size >= 40) ViewSynth.medianError(ViewSynth.identityPoints(sample), b) else Double.NaN
        val flipPx = if (a.size >= 40) ViewSynth.medianError(ViewSynth.predictPoints(flipped, pred.weightsUsed, cam), b) else Double.NaN
        val signWrong = a.size >= 40 && flipPx < 0.6 * guessPx && guessPx > 8.0

        // Photometric comparison in local-contrast space (robust to the colour-cycling lights).
        val s = pred.scale
        val bSmall = Mat(); val aSmall = Mat()
        Imgproc.resize(eyeB, bSmall, pred.guess.size(), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.resize(eyeA, aSmall, pred.guess.size(), 0.0, 0.0, Imgproc.INTER_AREA)
        val gB = normalizeLocal(toGray(bSmall))
        val gGuess = normalizeLocal(toGray(pred.guess))
        val gA = normalizeLocal(toGray(aSmall))
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        val validErode = Mat()
        Imgproc.erode(pred.validMask, validErode, kernel)
        val photoGuess = maskedMeanAbs(gGuess, gB, validErode)

        val all = Mat(gA.size(), CvType.CV_8UC1, Scalar(255.0))
        val photoNo = maskedMeanAbs(gA, gB, all)
        // gyro-only baseline: warp A by the infinite-depth homography K R K^-1
        val kS = Mat.eye(3, 3, CvType.CV_64F)
        kS.put(0, 0, cam.fx * s); kS.put(1, 1, cam.fy * s); kS.put(0, 2, cam.cx * s); kS.put(1, 2, cam.cy * s)
        val kInv = Mat(); Core.invert(kS, kInv)
        val rM = Mat(3, 3, CvType.CV_64F); rM.put(0, 0, *m.gyroRotationCam)
        val tmp = Mat(); val hM = Mat()
        Core.gemm(kS, rM, 1.0, Mat(), 0.0, tmp)
        Core.gemm(tmp, kInv, 1.0, Mat(), 0.0, hM)
        val gWarp = Mat(); val mWarp = Mat()
        Imgproc.warpPerspective(gA, gWarp, hM, gA.size())
        Imgproc.warpPerspective(all, mWarp, hM, gA.size(), Imgproc.INTER_NEAREST)
        Imgproc.erode(mWarp, mWarp, kernel)
        val photoGyro = maskedMeanAbs(gWarp, gB, mWarp)

        // Error picture: where the guess disagrees with the real B.
        val diff = Mat()
        Core.absdiff(gGuess, gB, diff)
        val diff4 = Mat()
        diff.convertTo(diff4, CvType.CV_8U, 4.0)
        val errColor = Mat()
        Imgproc.applyColorMap(diff4, errColor, Imgproc.COLORMAP_INFERNO)
        val holes = Mat(); Core.bitwise_not(validErode, holes)
        errColor.setTo(Scalar(0.0, 0.0, 0.0), holes)

        // ── learn ──
        val wBefore = store.w.copyOf()
        var learned = store.samples.size
        if (a.size >= 80) {
            // Train with the requested magnitude and whichever direction actually fits (the accelerometer's own sign is not trusted for learning).
            val train = if (userSlidLeft) instrLeftS else instrS
            store.samples.add(subsample(train, 400))
            while (store.samples.size > 30) store.samples.removeAt(0)
            store.w = ViewSynth.fit(store.samples, ViewSynth.FLAT_PRIOR, cam, 4.0, store.w)
            learned = store.samples.size
        }
        if (!instrPx.isNaN() && !userSlidLeft) store.policyErrors["INSTRUCTED"]!!.add(instrPx)
        if (!imuPolicyPx.isNaN()) store.policyErrors["IMU"]!!.add(imuPolicyPx)
        store.history.add(
            JSONObject().put("t", System.currentTimeMillis()).put("policy", pred.policy)
                .put("guessPx", guessPx).put("gyroOnlyPx", gyroPx).put("noChangePx", idPx)
                .put("photoGuess", photoGuess).put("signWrong", signWrong).put("n", a.size / 2)
        )
        store.save()
        val trend = store.trendText()
        Diagnostics.log(TAG, "REVEAL feature error guess=${"%.1f".format(guessPx)}px gyro-only=${"%.1f".format(gyroPx)}px none=${"%.1f".format(idPx)}px signWrong=$signWrong learnedFrom=$learned")

        for (mat in listOf(bSmall, aSmall, gB, gGuess, gA, validErode, all, kS, kInv, rM, tmp, hM, gWarp, mWarp, diff, diff4, holes, kernel)) mat.release()
        return Reveal(
            a.size / 2, guessPx, gyroPx, idPx, flipPx, photoGuess, photoGyro, photoNo, pred.coverage,
            signWrong, learned, wBefore, store.w.copyOf(), trend, errColor,
            pred.policy, instrPx, imuPolicyPx, userSlidLeft
        )
    }

    private fun subsample(s: ViewSynth.Sample, maxPts: Int): ViewSynth.Sample {
        val n = s.a.size / 2
        if (n <= maxPts) return s
        val step = n.toDouble() / maxPts
        val a = DoubleArray(maxPts * 2); val b = DoubleArray(maxPts * 2)
        for (i in 0 until maxPts) {
            val j = (i * step).toInt()
            a[2 * i] = s.a[2 * j]; a[2 * i + 1] = s.a[2 * j + 1]
            b[2 * i] = s.b[2 * j]; b[2 * i + 1] = s.b[2 * j + 1]
        }
        return ViewSynth.Sample(a, b, s.focusM, s.rot, s.c)
    }

    /** Ground-truth correspondences A -> B: full-resolution ORB, ratio test, fundamental-matrix RANSAC. Only ever called after the guess. */
    private fun correspondences(eyeA: Mat, eyeB: Mat): Pair<DoubleArray, DoubleArray> {
        val ga = toGray(eyeA); val gb = toGray(eyeB)
        val orb = ORB.create(3000)
        val kpA = MatOfKeyPoint(); val kpB = MatOfKeyPoint(); val dA = Mat(); val dB = Mat()
        orb.detectAndCompute(ga, Mat(), kpA, dA)
        orb.detectAndCompute(gb, Mat(), kpB, dB)
        ga.release(); gb.release()
        if (dA.empty() || dB.empty()) return Pair(DoubleArray(0), DoubleArray(0))
        val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
        val knn = ArrayList<MatOfDMatch>()
        matcher.knnMatch(dA, dB, knn, 2)
        val ka = kpA.toArray(); val kb = kpB.toArray()
        val pa = ArrayList<Point>(); val pb = ArrayList<Point>()
        for (mm in knn) {
            val arr = mm.toArray()
            if (arr.size >= 2 && arr[0].distance < 0.75f * arr[1].distance) {
                pa.add(ka[arr[0].queryIdx].pt); pb.add(kb[arr[0].trainIdx].pt)
            }
            mm.release()
        }
        kpA.release(); kpB.release(); dA.release(); dB.release()
        if (pa.size < 30) return Pair(DoubleArray(0), DoubleArray(0))
        val m1 = MatOfPoint2f(); m1.fromList(pa)
        val m2 = MatOfPoint2f(); m2.fromList(pb)
        val mask = Mat()
        val f = Calib3d.findFundamentalMat(m1, m2, Calib3d.FM_RANSAC, 1.5, 0.999, mask)
        val bytes = ByteArray(pa.size)
        if (f.empty()) { m1.release(); m2.release(); mask.release(); f.release(); return Pair(DoubleArray(0), DoubleArray(0)) }
        mask.get(0, 0, bytes)
        val a = ArrayList<Double>(); val b = ArrayList<Double>()
        for (i in pa.indices) if (bytes[i].toInt() != 0) { a.add(pa[i].x); a.add(pa[i].y); b.add(pb[i].x); b.add(pb[i].y) }
        m1.release(); m2.release(); mask.release(); f.release()
        return Pair(a.toDoubleArray(), b.toDoubleArray())
    }

    private fun toGray(bgr: Mat): Mat { val g = Mat(); Imgproc.cvtColor(bgr, g, Imgproc.COLOR_BGR2GRAY); return g }

    /** (I - localMean) / (localStd + eps), 8-bit. Local contrast is compared instead of raw colour, because the room's lights change colour. */
    private fun normalizeLocal(gray: Mat): Mat {
        val f = Mat(); gray.convertTo(f, CvType.CV_32F)
        val mu = Mat(); Imgproc.GaussianBlur(f, mu, Size(0.0, 0.0), 5.0)
        val diff = Mat(); Core.subtract(f, mu, diff)
        val sq = Mat(); Core.multiply(diff, diff, sq)
        val variance = Mat(); Imgproc.GaussianBlur(sq, variance, Size(0.0, 0.0), 5.0)
        val sd = Mat(); Core.sqrt(variance, sd); Core.add(sd, Scalar(8.0), sd)
        val norm = Mat(); Core.divide(diff, sd, norm)
        val out = Mat(); norm.convertTo(out, CvType.CV_8U, 48.0, 128.0)
        gray.release(); f.release(); mu.release(); diff.release(); sq.release(); variance.release(); sd.release(); norm.release()
        return out
    }

    /** Mean |a-b| over the mask, in units of 48 (one local standard deviation). */
    private fun maskedMeanAbs(a: Mat, b: Mat, mask: Mat): Double {
        val d = Mat(); Core.absdiff(a, b, d)
        val n = Core.countNonZero(mask)
        if (n < 50) { d.release(); return Double.NaN }
        val mean = Core.mean(d, mask).`val`[0] / 48.0
        d.release()
        return max(0.0, min(9.99, mean))
    }
}
