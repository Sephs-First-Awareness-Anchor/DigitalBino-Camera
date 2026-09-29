// Authored by Sunni (Sir) Morningstar and Cael Devo
package com.morningstar.stereoprobe

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.provider.Settings
import android.util.Size
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.atan

private const val TAG = "PROBE"

/** Numeric ids sort numerically ("2" < "10"), everything else after. */
val idComparator: Comparator<String> = compareBy<String>({ it.toIntOrNull() ?: Int.MAX_VALUE }, { it })

class CameraInfo(
    val id: String,
    val facing: Int?,
    val listed: Boolean,               // present in getCameraIdList()
    val logical: Boolean,              // REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA
    val physicalIds: Set<String>,      // getPhysicalCameraIds()
    val timestampSource: Int?,
    val focalLengths: List<Float>,
    val syncType: Int?,
    val streamSizes: Map<Int, List<Size>>,
    val json: JSONObject
) {
    /** Logical cameras that list this id as one of their physical cameras. */
    val memberOf = LinkedHashSet<String>()
    var inConcurrentSet = false
}

class ProbeResult(
    val cameras: List<CameraInfo>,
    val concurrentSets: List<Set<String>>,
    val concurrentFeature: Boolean,
    val concurrentError: String?,
    val json: JSONObject
) {
    val byId: Map<String, CameraInfo> = cameras.associateBy { it.id }

    fun listedRear(): List<String> =
        cameras.filter { it.listed && it.facing == CameraCharacteristics.LENS_FACING_BACK }
            .map { it.id }.sortedWith(idComparator)

    fun rearRearListedPairs(): List<Pair<String, String>> {
        val r = listedRear()
        val out = ArrayList<Pair<String, String>>()
        for (i in r.indices) for (j in i + 1 until r.size) out.add(Pair(r[i], r[j]))
        return out
    }

    /** Every 2-subset of every set Android reports as concurrently usable. */
    fun claimedPairs(): Set<Pair<String, String>> {
        val out = LinkedHashSet<Pair<String, String>>()
        for (s in concurrentSets) {
            val ids = s.sortedWith(idComparator)
            for (i in ids.indices) for (j in i + 1 until ids.size) out.add(Pair(ids[i], ids[j]))
        }
        return out
    }

    fun summary(): String {
        val sb = StringBuilder("═══ CAMERAS ═══\n")
        for (c in cameras.sortedWith(compareBy(idComparator) { it.id })) {
            val role = when {
                c.logical -> "LOGICAL(phys ${c.physicalIds.sortedWith(idComparator).joinToString(",")})"
                c.memberOf.isNotEmpty() -> "PHYSICAL(in logical ${c.memberOf.joinToString(",")})"
                else -> "STANDALONE"
            }
            val ts = when (c.timestampSource) {
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME -> "REALTIME"
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN -> "UNKNOWN"
                else -> "n/a"
            }
            sb.append("[${c.id}] ${DeviceProbe.facingName(c.facing)} $role${if (!c.listed) " HIDDEN(not in id list)" else ""}\n")
            sb.append("     f=${c.focalLengths} ts=$ts hw=${c.json.optString("hardwareLevel")} concurrentSet=${if (c.inConcurrentSet) "yes" else "no"}\n")
            sb.append("     sensor=${c.json.optString("sensorPhysicalSizeMm")} px=${c.json.optString("pixelArraySize")} syncType=${c.json.optString("logicalSyncType")}\n")
        }
        sb.append("\nconcurrent feature flag: $concurrentFeature")
        if (concurrentError != null) sb.append("  (query error: $concurrentError)")
        sb.append("\nAndroid-reported concurrent sets: ")
        sb.append(if (concurrentSets.isEmpty()) "NONE" else concurrentSets.joinToString(" ") { it.sortedWith(idComparator).toString() })
        sb.append('\n')
        return sb.toString()
    }
}

class DeviceProbe(private val context: Context) {

    private val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    companion object {
        fun facingName(f: Int?): String = when (f) {
            CameraCharacteristics.LENS_FACING_FRONT -> "FRONT"
            CameraCharacteristics.LENS_FACING_BACK -> "BACK"
            CameraCharacteristics.LENS_FACING_EXTERNAL -> "EXTERNAL"
            null -> "UNKNOWN"
            else -> "OTHER($f)"
        }

        fun formatName(f: Int): String = when (f) {
            ImageFormat.JPEG -> "JPEG"
            ImageFormat.YUV_420_888 -> "YUV_420_888"
            ImageFormat.RAW_SENSOR -> "RAW_SENSOR"
            ImageFormat.RAW10 -> "RAW10"
            ImageFormat.RAW12 -> "RAW12"
            ImageFormat.PRIVATE -> "PRIVATE"
            ImageFormat.DEPTH16 -> "DEPTH16"
            ImageFormat.DEPTH_POINT_CLOUD -> "DEPTH_POINT_CLOUD"
            ImageFormat.DEPTH_JPEG -> "DEPTH_JPEG"
            ImageFormat.HEIC -> "HEIC"
            ImageFormat.Y8 -> "Y8"
            ImageFormat.YV12 -> "YV12"
            ImageFormat.NV21 -> "NV21"
            else -> "FORMAT_$f"
        }

        fun capabilityName(c: Int): String = when (c) {
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_BACKWARD_COMPATIBLE -> "BACKWARD_COMPATIBLE"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR -> "MANUAL_SENSOR"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_POST_PROCESSING -> "MANUAL_POST_PROCESSING"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW -> "RAW"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_PRIVATE_REPROCESSING -> "PRIVATE_REPROCESSING"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_READ_SENSOR_SETTINGS -> "READ_SENSOR_SETTINGS"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_BURST_CAPTURE -> "BURST_CAPTURE"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_YUV_REPROCESSING -> "YUV_REPROCESSING"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT -> "DEPTH_OUTPUT"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_CONSTRAINED_HIGH_SPEED_VIDEO -> "CONSTRAINED_HIGH_SPEED_VIDEO"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MOTION_TRACKING -> "MOTION_TRACKING"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA -> "LOGICAL_MULTI_CAMERA"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MONOCHROME -> "MONOCHROME"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_SECURE_IMAGE_DATA -> "SECURE_IMAGE_DATA"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_SYSTEM_CAMERA -> "SYSTEM_CAMERA"
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_OFFLINE_PROCESSING -> "OFFLINE_PROCESSING"
            else -> "CAPABILITY_$c"
        }

        fun hardwareLevelName(l: Int?): String = when (l) {
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
            null -> "n/a"
            else -> "LEVEL_$l"
        }

        private fun fmtValue(v: Any?): String {
            val s = when (v) {
                null -> "null"
                is IntArray -> v.contentToString()
                is LongArray -> v.contentToString()
                is FloatArray -> v.contentToString()
                is DoubleArray -> v.contentToString()
                is BooleanArray -> v.contentToString()
                is ByteArray -> v.contentToString()
                is Array<*> -> v.contentDeepToString()
                else -> v.toString()
            }
            return if (s.length > 300) s.substring(0, 300) + "…(truncated)" else s
        }
    }

    // ───────────────────────── PHASE 0 ─────────────────────────

    fun probeDevice(): JSONObject {
        val j = JSONObject()
        j.put("manufacturer", Build.MANUFACTURER)
        j.put("brand", Build.BRAND)
        j.put("model", Build.MODEL)
        j.put("device", Build.DEVICE)
        j.put("product", Build.PRODUCT)
        j.put("board", Build.BOARD)
        j.put("hardware", Build.HARDWARE)
        j.put("fingerprint", Build.FINGERPRINT)
        j.put("display", Build.DISPLAY)
        j.put("buildId", Build.ID)
        j.put("buildType", Build.TYPE)
        j.put("buildTags", Build.TAGS)
        j.put("buildTimeMs", Build.TIME)
        j.put("androidRelease", Build.VERSION.RELEASE)
        j.put("sdkInt", Build.VERSION.SDK_INT)
        j.put("securityPatch", Build.VERSION.SECURITY_PATCH)
        j.put("incremental", Build.VERSION.INCREMENTAL)
        j.put("codename", Build.VERSION.CODENAME)
        j.put("radioVersion", Build.getRadioVersion() ?: "n/a")
        if (Build.VERSION.SDK_INT >= 31) {
            j.put("socManufacturer", Build.SOC_MANUFACTURER)
            j.put("socModel", Build.SOC_MODEL)
        }
        j.put("supportedAbis", JSONArray(Build.SUPPORTED_ABIS.toList()))
        // ANDROID_ID is app-signing-key scoped on modern Android; it identifies this handset for this app.
        j.put("androidId", Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID) ?: "n/a")
        // Informational pattern check only; the physical handset under test stays authoritative.
        j.put("modelMatchesGalaxyA16Pattern_informational", Build.MODEL.startsWith("SM-A16"))

        val pm = context.packageManager
        j.put("featureCameraConcurrent", pm.hasSystemFeature(PackageManager.FEATURE_CAMERA_CONCURRENT))
        val cam = JSONArray()
        pm.systemAvailableFeatures
            .mapNotNull { it.name }
            .filter { it.startsWith("android.hardware.camera") }
            .sorted()
            .forEach { cam.put(it) }
        j.put("cameraSystemFeatures", cam)

        Diagnostics.log(TAG, "device: ${Build.MANUFACTURER} ${Build.MODEL} Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}) build ${Build.DISPLAY}")
        return j
    }

    fun deviceSummary(j: JSONObject): String {
        val sb = StringBuilder("═══ DEVICE ═══\n")
        sb.append("${j.optString("manufacturer")} ${j.optString("model")} (${j.optString("device")})\n")
        sb.append("Android ${j.optString("androidRelease")} / API ${j.optInt("sdkInt")}  patch ${j.optString("securityPatch")}\n")
        sb.append("build ${j.optString("display")}\n")
        sb.append("hardware ${j.optString("hardware")} board ${j.optString("board")}")
        if (j.has("socModel")) sb.append("  SoC ${j.optString("socManufacturer")} ${j.optString("socModel")}")
        sb.append("\nandroidId ${j.optString("androidId")}\n")
        sb.append("camera features: ${j.optJSONArray("cameraSystemFeatures")}\n")
        return sb.toString()
    }

    // ───────────────────────── PHASE 1 ─────────────────────────

    fun probeCameras(): ProbeResult {
        val cams = ArrayList<CameraInfo>()
        val idList = try {
            manager.cameraIdList.toList()
        } catch (t: Throwable) {
            Diagnostics.error(TAG, "getCameraIdList failed", t)
            emptyList()
        }
        Diagnostics.log(TAG, "getCameraIdList = $idList")

        for (id in idList) inspect(id, true)?.let { cams.add(it) }

        // Physical ids that logical cameras report but the id list does not: "hidden" physical cameras.
        val known = cams.map { it.id }.toMutableSet()
        for (logical in cams.filter { it.logical }.toList()) {
            for (pid in logical.physicalIds) {
                if (pid !in known) {
                    Diagnostics.log(TAG, "hidden physical camera $pid (member of logical ${logical.id}); trying to read characteristics")
                    val hidden = inspect(pid, false)
                    if (hidden != null) { cams.add(hidden); known.add(pid) }
                }
            }
        }

        for (logical in cams.filter { it.logical }) {
            for (pid in logical.physicalIds) cams.firstOrNull { it.id == pid }?.memberOf?.add(logical.id)
        }

        var concurrentFeature = false
        var concurrentError: String? = null
        val sets = ArrayList<Set<String>>()
        try {
            concurrentFeature = context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_CONCURRENT)
            for (s in manager.concurrentCameraIds) sets.add(s.toSortedSet(idComparator))
        } catch (t: Throwable) {
            concurrentError = Diagnostics.describe(t)
            Diagnostics.error(TAG, "getConcurrentCameraIds failed", t)
        }
        Diagnostics.log(TAG, "concurrent camera id sets reported by Android: $sets")
        val concurrentIds = sets.flatten().toSet()
        cams.forEach { it.inConcurrentSet = it.id in concurrentIds }

        val j = JSONObject()
        val arr = JSONArray()
        cams.sortedWith(compareBy(idComparator) { it.id }).forEach { c ->
            c.json.put("memberOfLogical", JSONArray(c.memberOf.toList()))
            c.json.put("appearsInAndroidConcurrentSet", c.inConcurrentSet)
            arr.put(c.json)
        }
        j.put("cameras", arr)
        j.put("cameraIdList", JSONArray(idList))
        j.put("concurrentFeatureFlag", concurrentFeature)
        j.put("concurrentQueryError", concurrentError ?: JSONObject.NULL)
        val setsJson = JSONArray()
        sets.forEach { setsJson.put(JSONArray(it.toList())) }
        j.put("concurrentCameraIds", setsJson)
        return ProbeResult(cams, sets, concurrentFeature, concurrentError, j)
    }

    @Suppress("UNCHECKED_CAST")
    private fun readKey(c: CameraCharacteristics, k: CameraCharacteristics.Key<*>): Any? =
        c.get(k as CameraCharacteristics.Key<Any>)

    private fun inspect(id: String, listed: Boolean): CameraInfo? {
        val c = try {
            manager.getCameraCharacteristics(id)
        } catch (t: Throwable) {
            Diagnostics.error(TAG, "getCameraCharacteristics($id) failed", t)
            return null
        }
        val j = JSONObject()
        j.put("id", id)
        j.put("listedInCameraIdList", listed)

        val facing = c.get(CameraCharacteristics.LENS_FACING)
        j.put("facing", DeviceProbe.facingName(facing))

        val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toList() ?: emptyList()
        j.put("capabilities", JSONArray(caps.map { capabilityName(it) }))
        val logical = caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA)
        j.put("logicalMultiCamera", logical)

        val physicalIds: Set<String> = try { c.physicalCameraIds } catch (t: Throwable) { emptySet() }
        j.put("physicalCameraIds", JSONArray(physicalIds.sortedWith(idComparator)))

        val syncType = c.get(CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE)
        j.put(
            "logicalSyncType",
            when (syncType) {
                CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_APPROXIMATE -> "APPROXIMATE"
                CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_CALIBRATED -> "CALIBRATED"
                null -> "n/a"
                else -> "TYPE_$syncType"
            }
        )

        val hw = c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
        j.put("hardwareLevel", hardwareLevelName(hw))
        j.put("sensorOrientationDeg", c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: JSONObject.NULL)

        val focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList() ?: emptyList()
        j.put("focalLengthsMm", JSONArray(focal.map { it.toDouble() }))
        val phys = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        j.put("sensorPhysicalSizeMm", if (phys != null) "${phys.width} x ${phys.height}" else "n/a")
        val pixel = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
        j.put("pixelArraySize", if (pixel != null) "${pixel.width} x ${pixel.height}" else "n/a")
        val active = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
        j.put("activeArraySize", active?.toString() ?: "n/a")

        // Derived (not reported by Android): horizontal FOV from sensor width + focal length.
        if (phys != null && focal.isNotEmpty()) {
            val fovs = JSONArray()
            focal.forEach { f -> fovs.put(Math.round(Math.toDegrees(2.0 * atan(phys.width / (2.0 * f))) * 10.0) / 10.0) }
            j.put("derivedHorizontalFovDeg", fovs)
        }

        val ts = c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
        j.put(
            "timestampSource",
            when (ts) {
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME -> "REALTIME"
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN -> "UNKNOWN"
                null -> "n/a"
                else -> "SOURCE_$ts"
            }
        )

        j.put("minFocusDistanceDiopters", c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: JSONObject.NULL)
        j.put("hyperfocalDistanceDiopters", c.get(CameraCharacteristics.LENS_INFO_HYPERFOCAL_DISTANCE) ?: JSONObject.NULL)

        val fps = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
        j.put("aeFpsRanges", JSONArray((fps ?: emptyArray()).map { "${it.lower}-${it.upper}" }))

        // Stream configurations
        val streamSizes = HashMap<Int, List<Size>>()
        val fmts = JSONObject()
        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        if (map != null) {
            for (f in map.outputFormats) {
                val sizes = map.getOutputSizes(f)?.toList() ?: emptyList()
                streamSizes[f] = sizes
                val a = JSONArray()
                for (s in sizes) {
                    val dur = try { map.getOutputMinFrameDuration(f, s) } catch (t: Throwable) { 0L }
                    a.put("${s.width}x${s.height}" + if (dur > 0) "@${"%.1f".format(1e9 / dur)}fps" else "")
                }
                fmts.put(formatName(f), a)
            }
        }
        j.put("outputSizesByFormat", fmts)

        // Calibration / pose / distortion / depth
        j.put("lensIntrinsicCalibration_fx_fy_cx_cy_s", c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)?.let { fmtValue(it) } ?: "n/a")
        j.put("lensDistortion", c.get(CameraCharacteristics.LENS_DISTORTION)?.let { fmtValue(it) } ?: "n/a")
        j.put("lensPoseRotationQuaternion", c.get(CameraCharacteristics.LENS_POSE_ROTATION)?.let { fmtValue(it) } ?: "n/a")
        j.put("lensPoseTranslationMeters", c.get(CameraCharacteristics.LENS_POSE_TRANSLATION)?.let { fmtValue(it) } ?: "n/a")
        j.put(
            "lensPoseReference",
            when (val ref = c.get(CameraCharacteristics.LENS_POSE_REFERENCE)) {
                CameraCharacteristics.LENS_POSE_REFERENCE_PRIMARY_CAMERA -> "PRIMARY_CAMERA"
                CameraCharacteristics.LENS_POSE_REFERENCE_GYROSCOPE -> "GYROSCOPE"
                CameraCharacteristics.LENS_POSE_REFERENCE_UNDEFINED -> "UNDEFINED"
                null -> "n/a"
                else -> "REFERENCE_$ref"
            }
        )
        j.put("depthOutputCapability", caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT))
        j.put("depthIsExclusive", c.get(CameraCharacteristics.DEPTH_DEPTH_IS_EXCLUSIVE) ?: JSONObject.NULL)

        // Request / result keys, including vendor keys (anything not prefixed "android.")
        try {
            val reqKeys = c.availableCaptureRequestKeys.map { it.name }
            val resKeys = c.availableCaptureResultKeys.map { it.name }
            j.put("captureRequestKeyCount", reqKeys.size)
            j.put("captureResultKeyCount", resKeys.size)
            j.put("vendorCaptureRequestKeys", JSONArray(reqKeys.filter { !it.startsWith("android.") }))
            j.put("vendorCaptureResultKeys", JSONArray(resKeys.filter { !it.startsWith("android.") }))
            val physReq = c.availablePhysicalCameraRequestKeys?.map { it.name } ?: emptyList()
            j.put("availablePhysicalCameraRequestKeys", JSONArray(physReq))
        } catch (t: Throwable) {
            Diagnostics.error(TAG, "request/result key enumeration failed for $id", t)
        }

        // Full dump of every characteristics key (vendor keys included), values truncated.
        val all = JSONObject()
        val vendor = JSONObject()
        try {
            for (k in c.keys) {
                val v = try { fmtValue(readKey(c, k)) } catch (t: Throwable) { "unreadable: ${t.javaClass.simpleName}" }
                if (k.name == CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP.name) continue
                all.put(k.name, v)
                if (!k.name.startsWith("android.")) vendor.put(k.name, v)
            }
        } catch (t: Throwable) {
            Diagnostics.error(TAG, "characteristics key dump failed for $id", t)
        }
        j.put("allCharacteristicKeys", all)
        j.put("vendorCharacteristicKeys", vendor)

        Diagnostics.log(
            TAG,
            "camera $id facing=${facingName(facing)} logical=$logical physical=${physicalIds.sortedWith(idComparator)} " +
                "f=$focal listed=$listed vendorKeys=${vendor.length()}"
        )

        return CameraInfo(id, facing, listed, logical, physicalIds, ts, focal, syncType, streamSizes, j)
    }
}
