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


---

# Bino Sweep (motion-baseline binocular capture)
Button **5 · BINO SWEEP** on the main screen. One rear camera stays open and streaming; you slide the phone sideways.

## Flow
1. Frame the subject, press **CAPTURE**: Eye A locks (exposure, white balance and focus lock with it).
2. A translucent ghost of Eye A overlays the live view. Slide the phone right, keeping its orientation.
3. The HUD shows baseline progress, pitch/yaw/roll, overlap, inliers and parallax.
4. Eye B is chosen automatically from a rolling candidate buffer, with a haptic confirmation.
5. Relative pose is estimated, the pair is rectified, disparity and depth are computed.
6. Result viewer: Eye A / Eye B / Rect (with guide lines) / Disp / Depth. Tap Disp or Depth for distance. **Export** saves PNGs + diagnostics JSON to Downloads/StereoProbe.

## Architecture
- `StereoAcquisition.kt`: `StereoAcquisitionProvider`, `StereoAcquisitionResult`, `AcquisitionSelector`. Concurrent provider is a future slot.
- `MotionBaselineProvider.kt`: camera, sweep state machine, gating, candidate buffer.
- `MotionSensors.kt`, `VisualTracker.kt`: IMU and feature tracking (each constrains the other).
- `StereoPipeline.kt`: rectification, SGBM disparity, depth. Never asks how the pair was obtained.
- `SweepActivity.kt`: UI only.

## What "genuine translation" means here
Rotation-only motion is fully explained by a homography, so the median residual after rotation compensation is about 0 for a pivot
and depth-dependent for a slide. Eye B is refused unless that parallax residual, IMU baseline, orientation, overlap, inliers, sharpness and exposure all pass.

## Verification status (be honest about it)
- Compiles with kotlinc 1.9.24 against android-34 android.jar and OpenCV 4.9.0 classes.
- Pose conventions verified on a synthetic scene with known motion (direction recovery, left/right swap, rectification, depth, pivot-vs-slide residual).
- NOT yet run through Gradle or on a handset.

## Known limits / still unverified on hardware
- **Metric scale is the weak point.** Direction comes from vision; magnitude comes from double-integrating the accelerometer, which drifts.
  Structure is reliable before scale is. The export reports IMU-vs-vision direction agreement as baseline confidence.
- Rotation hint wording ("tilt down", "rotate right") is derived from axis conventions, not yet confirmed on the phone.
- Intrinsics are derived from focal length + sensor size (no calibration, no distortion model).
- Moving subjects are only detected as "scene changed too much"; they are not masked yet.
- All thresholds live in `SweepConfig` so they can be tuned from exported diagnostics.


## v0.2.1 fixes (from the first on-device Bino Sweep log)
All eight attempts failed with "Scene changed too much". The log showed this was my code, not the scene:
- `recoverPose` silently discards points farther than ~50 baselines (about 2.5 m for a 5 cm slide). Overlap, pose-inlier count and the
  scene-changed test were all computed from that starved set. Now: far-point rejection disabled, geometry judged by essential-matrix inliers,
  overlap from the larger of two hulls, and "scene changed" requires matches to be wrong everywhere (rotation-inconsistent AND no epipolar structure).
- Integrated accelerometer displacement drifted to 5 cm with almost no sideways motion. Acceptance is now decided by measured PARALLAX (vision),
  progress bar and HUD use parallax, and IMU centimetres are labelled coarse. Drift correction (zero-velocity update) is applied when held still at Eye B.
- Rotation source is now the game rotation vector (gyro + accelerometer, no magnetometer). Bias window ends 0.3 s before Eye A so the screen tap is excluded.
- Either slide direction is accepted (direction auto-detected).
- Reset race ("CameraDevice was already closed") fixed: callbacks ignore a stopped provider, RESET is debounced.
- On failure the app saves `sweepfail_*.json` (per-frame telemetry) plus Eye A and last-frame PNGs to Downloads/StereoProbe.


## v0.2.2 (after the first complete on-device capture)
The 02:07:58 capture proved the geometry: 1,640 row-aligned matches at 0.00 px median vertical error after rectification, disparity rising
smoothly from wall (10 px) to plush (18 px) to legs (36 px), 89% sideways motion, 0.6 deg rotation. What it exposed:
- **Metric scale was wrong by an order of magnitude.** Accelerometer double integration reported 129 cm of travel over an 11.5 s sweep
  (baseline 20.1 cm); the measured disparities imply roughly 1-2 cm. Now the accelerometer is trusted only within `imuTrustSec` (2.5 s) and
  `maxPlausibleImuM` (30 cm) AND when its direction agrees with vision; otherwise depth is UNSCALED (relative, in "baselines").
  Scale then comes from, in order: short plausible IMU sweep, camera focus distance (only if the lens reports APPROXIMATE/CALIBRATED; the
  app logs `focus-distance calibration:` at start), or **Set scale**: tap a point in Disp/Depth, press Set scale, type its real distance in metres.
- **Overlap metric** was measured against the whole frame, but only ~52% of it is textured, so it hovered at 43-46% against a 45% threshold.
  Now measured against the hull of all keypoints in frame A (90.8% on the real pair); threshold 60%.
- Depth colours now use the data's own 2nd-98th percentile range, independent of the scale.
- Left/right order is decided by measured disparity sign (stereoRectify ignores the sign of T; recoverPose's sign can be wrong on noisy data).
- Local photometric normalisation before SGBM; rectifyAlpha 1.0 so nothing is cropped; valid-region mask; rectification rotation and zoom are logged.
- Vision-based "slide straight, do not arc" gate; colour-cycling lights trigger a warning.
Verified: compiles against android-34 + OpenCV 4.9.0; overlap metric replayed on the real pair; normalisation tested on synthetic pairs.
Not verified on device: the new scale UI, the focus-distance anchor, the new rectification defaults.


## v0.3.0 (from the 02:23-02:26 runs)
Findings from seven captures: 4 rectified, 2 were refused as "not mostly horizontal", 1 was degenerate. Root cause: the direction of camera
motion recovered from the essential matrix is UNSTABLE when the baseline is small compared with scene depth (on one real pair it read 13% sideways
at 540x720 and 78% at 1080x1440; the app's own runs disagreed with offline re-runs of the same pair).
- **Gyro-constrained pose** (`GyroPoseSolver.kt`, pure Kotlin): rotation held at the game-rotation-vector value, translation direction solved
  linearly (smallest eigenvector, robust reweighting, cheirality sign), then a sub-degree rotation nudge by pattern search on Sampson error.
  On the three real pairs it gave 98-100% sideways, epipolar error 0.32-0.45 px (the free essential-matrix fit was 0.48-0.52 px) and clean
  rectification (413-556 row-aligned matches, 0.76-1.04 px RMS). The Kotlin solver was verified on a JVM against a Python reference on the same data.
  It now drives the in-sweep "slide straight" gate and the final pose. The essential-matrix pose is kept as a fallback and for diagnostics
  (`poseMethod`, `gyroSampsonPx`, `essentialSampsonPx` in the export).
- Non-sideways sweeps are refused with the measured percentage instead of producing a degenerate rectification.
- Disparity search range is now adaptive (from measured feature disparities); the fixed range saturated on wide baselines.
- Degenerate rectification (zoom <= 0.15) is reported as such.
- `acceptParallaxPx` raised from 10 to 14 to get a larger baseline.
- The camera reports `LENS_INFO_FOCUS_DISTANCE_CALIBRATION = CALIBRATED`, so the focus-distance scale anchor is used. Offline, dense disparity at the
  focused centre gave baselines of 5.2 / 6.4 / 6.1 cm for three sweeps (consistent, near the 6 cm aimed for).
  Correction: an earlier estimate of 1-2 cm was biased by feature-only disparities (the plush face has no texture, so features sit on the far background).
Verified offline on real pairs: pose solver, rectification, SGBM vs feature disparity agreement (median 0.59 px). Not yet run on the phone: the new build.


## v0.4.0: the app guesses Eye B before seeing it ("Guess B")
After each sweep the app builds a guess for Eye B from **Eye A + sensor metrics only**, THEN reveals the real Eye B, scores itself, and learns.
- **No borrowed model.** The model is exact projective geometry plus a small depth prior the app fits to its own captures
  (`ViewSynth.kt`: 7 weights, Nelder-Mead with a pull toward a neutral prior, replay buffer of the last 30 captures, stored in the app's private
  `view_predictor.json`). Delete the app's data to reset learning.
- **No leakage.** `ViewPredictor.predict(eyeA, intrinsics, sensorMetrics)` is never given Eye B. `SensorMetrics` holds only gyro rotation, accelerometer
  displacement, focus distance and timing. The visually estimated pose is excluded on purpose (it is computed by matching A against B).
- **Scoreboard** (Guess B / Error buttons; also in the export): median feature-position error of the guess vs two honest baselines
  (gyro-rotation-only, and no change) plus an image error in local-contrast units, and a trend over the last captures.
- **Sensor-sign check.** If flipping the accelerometer direction makes the guess much better, that capture is flagged and learned with the corrected sign.
Known limits: baseline error and depth level multiply, so they cannot be separated from one capture (weight 0 absorbs both);
lighting changes between eyes (colour-cycling LEDs) cannot be predicted by geometry; the guess inpaints disocclusions;
learning from a handful of captures of one room will not generalise to other places; the image-error metric is weakly discriminative.
Verified offline on three real pairs: Kotlin geometry reproduces the Python prototype exactly; prior fitted on two captures predicted the third
(sign corrected) to 10.5 px vs 25.2 px for gyro-only. Not yet run on the phone.
