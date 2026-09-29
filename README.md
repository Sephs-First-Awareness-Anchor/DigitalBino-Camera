# Stereo Probe (Phases 0-3 + sequential flip measurement)
Authored by Sunni (Sir) Morningstar and Cael Devo

Question this APK answers: **can this Galaxy A16 actually see through two rear cameras at once?**

## Build
1. Open the `StereoProbe` folder in Android Studio (Ladybug or newer) and let Gradle sync.
   Or from a terminal with Gradle installed: `gradle wrapper --gradle-version 8.7` once, then `./gradlew assembleDebug`.
2. APK lands at `app/build/outputs/apk/debug/app-debug.apk`. Install with `adb install -r app-debug.apk` or copy to the phone.
3. If Studio suggests newer AGP/Kotlin versions, accepting them is fine. Nothing here depends on a specific version.

## Use
1. Launch, grant camera permission. Device + camera probe runs automatically (button 1 re-runs it).
2. **TEST ALL CAMERA PAIRS**: single-camera baselines, every pair Android claims, every rear+rear pair,
   logical-camera physical streams, and direct-open attempts on hidden physical IDs. Takes about a minute.
   Close other camera apps first (CAMERA_IN_USE will otherwise pollute results).
3. **FLIP TEST**: measures the open/grab/close/switch loop for every rear pair. Labelled sequential, never stereo.
3b. **HOLD / SUSPEND + FAST FLIP**: Camera2 has no suspend call, so this tests the nearest equivalents in BOTH open orders:
   device A open but idle, and device A open with its session closed, then tries to open B. If both can stay open it
   measures a session-swap bounce; it also times a fast flip phase by phase (open / session / first frame / close) with retry-on-busy and no artificial pauses.
4. **EXPORT**: writes JSON + TXT to Downloads/StereoProbe and opens the share sheet.

## Classification labels
TRUE STEREO CANDIDATE, CONCURRENT BUT UNCERTAIN, LOGICAL MULTI-CAMERA ONLY, UNSUPPORTED,
API-RESTRICTED / VENDOR-RESTRICTED, plus an informational CONCURRENT, NOT A REAR PAIR for front+rear combos.
"Restriction locus" text in the report is a heuristic hint, not proof.

## Reading Δt
Nearest-frame Δt between two unsynchronised streams is roughly uniform on 0..P/2 (mean about P/4).
Sensors sharing a trigger sit near zero. The verdict line spells out which one you are looking at.
Timestamps are driver-reported; the report never claims hardware sync it did not measure.

## Verification status
- Compiled with kotlinc 1.9.24 against the real android-34 android.jar: no errors.
  One deprecation warning (`TotalCaptureResult.getPhysicalCameraResults`), still functional.
- NOT yet run through Gradle or on a handset. First device run is the real test.

## Not built yet (later phases)
Dual live view UI, paired-capture gallery, recording, OpenCV calibration/rectification/disparity/depth.
Each should wait for the evidence this probe produces.
