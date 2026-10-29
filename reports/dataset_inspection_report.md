# Dataset Inspection Report: 360° Multimodal Viewport Prediction

> **Dataset**: Head, Eye, and Face Tracking Multi-Modal Dataset for User-Specific Viewport Prediction in 360° Videos

> **Strict Requirement Verified**: ONLY the `combined.csv` dataset is accessed. No individual head/eye/face files are used.


## 1. Executive Summary

| Metric | Result | Status / Notes |
|---|---|---|
| **Total Participants** | 25 / 25 | Complete (P01 to P25) |
| **Total Videos** | 14 / 14 | Complete (V01 to V14) |
| **Combined CSV Files** | 350 / 350 | 100% Present (No missing files) |
| **Total Dataset Rows** | 5,816,644 | Verified across all files |
| **Missing / NaN Values** | 0 across all columns | 100% Complete & Clean |
| **Column Alignment** | 26 columns in exact order | Verified consistent across all 350 files |
| **Hardware Sampling Rate** | ~71.94 Hz (dt ≈ 13.90 ms) | High temporal fidelity |
| **Video Playback Rate** | ~30.0 fps (dt ≈ 33.3 ms) | Standard video playback rate |

## 2. Directory Structure and Naming Conventions

```
DataCollectedVR/
├── P001-20260710T102047Z-2-001/ -> P001/
│   ├── video_01_Weekly_Idol-Dancing/ -> combined.csv
│   ├── ...
│   └── video_14_Death_Note_VR/ -> combined.csv
├── ...
└── P025-20260710T102233Z-2-001/ -> P025/
```

## 3. Video Inventory and Temporal Coverage

| Video ID | Video Name | Duration (s) | Max Video Frame | Avg Rows (per user) |
|---|---|---|---|---|
| `V01` | video_01_Weekly_Idol-Dancing | 278.28 s | 8340 | 20,042 |
| `V02` | video_02_Damai_Music_Live-Vocie_Toy | 341.04 s | 8526 | 24,562 |
| `V03` | video_03_Rio_Olympics_VR_Interview | 186.12 s | 4653 | 13,472 |
| `V04` | video_04_Female_Basketball_Match | 361.29 s | 10828 | 26,020 |
| `V05` | video_05_SHOWTIME_Boxing | 169.03 s | 5071 | 12,386 |
| `V06` | video_06_Rosa_Khutor_Ski_360 | 340.47 s | 10204 | 24,521 |
| `V07` | video_07_Hammerhead_Shark_NatGeo_360 | 134.27 s | 4024 | 9,672 |
| `V08` | video_08_Freestyle_Skiing | 200.07 s | 6002 | 14,449 |
| `V09` | video_09_Google_Spotlight-HELP | 293.2 s | 8796 | 21,066 |
| `V10` | video_10_Deep_Ocean_Horror_360 | 212.18 s | 12731 | 15,280 |
| `V11` | video_11_GoPro_VR-Tahiti_Surf | 205.67 s | 6164 | 14,813 |
| `V12` | video_12_LOSC_Football | 163.0 s | 4075 | 11,839 |
| `V13` | video_13_Jungle_Book_360 | 93.33 s | 2800 | 6,708 |
| `V14` | video_14_Death_Note_VR | 262.66 s | 7872 | 18,872 |


## 4. Timestamp vs. PlaybackTime Analysis

- **`Timestamp`**: High-resolution continuous session hardware clock in seconds from the VR headset tracker (Vive Pro Eye). Samples at **~72 Hz** (interval ~0.0139s). Every record has a strictly unique, monotonically increasing `Timestamp`.

- **`PlaybackTime`**: Media player presentation timestamp in seconds. Because 360° videos run at **30 fps**, `PlaybackTime` updates every ~0.0333s. Consequently, for each discrete `PlaybackTime`, there are **2 to 3 tracking samples**.

- **`VideoFrame`**: Discrete integer index of the displayed frame (0 to end, -1 indicates pre-playback initialization).

- **User Input Handling**: When the user requests a playback timestamp (e.g., 1.10s), the system queries the historical observations where `PlaybackTime <= requested_timestamp` (or using the corresponding session `Timestamp`), ensuring only past observations are fed to the model without looking ahead.


## 5. Comprehensive Data Dictionary

| Column Name | Category | Data Type | Units | Observed Range | Target? | Description |
|---|---|---|---|---|---|---|
| `Timestamp` | Temporal | `float64` | seconds | [0.1538, 361.6427] | No | Hardware tracking session timestamp (continuous high-resolution clock at ~72 Hz). |
| `PlaybackTime` | Temporal | `float64` | seconds | [0.0, 361.294] | No | Video playback time in media player coordinates (discrete steps corresponding to ~30 fps frames). |
| `VideoFrame` | Temporal | `int64` | frame index | [-1.0, 12731.0] | No | Sequential video frame number (0 to end, -1 indicates pre-playback initialization). |
| `HeadYaw` | Head Tracking | `float64` | degrees | [-179.9976, 179.9984] | Yes | Horizontal rotational angle of the head in 360° spherical coordinates [-180°, +180°]. |
| `HeadPitch` | Head Tracking | `float64` | degrees | [-44.4292, 70.0671] | Yes | Vertical elevation angle of the head [-90°, +90°]. |
| `HeadRoll` | Head Tracking | `float64` | degrees | [-53.1561, 76.7247] | Yes | Lateral tilt angle of the head (ear-to-shoulder rotation) in degrees. |
| `GazeYaw` | Eye / Gaze Tracking | `float64` | degrees | [-179.993, 179.9963] | Yes | World horizontal gaze orientation = HeadYaw + GazeRelativeH (0.0 when untracked). |
| `GazePitch` | Eye / Gaze Tracking | `float64` | degrees | [-87.7987, 61.9085] | Yes | World vertical gaze orientation = HeadPitch + GazeRelativeV (0.0 when untracked). |
| `GazeRelativeH` | Eye / Gaze Tracking | `float64` | degrees | [-153.8742, 150.3311] | Yes | Relative horizontal gaze offset of eye pupils with respect to head forward direction. |
| `GazeRelativeV` | Eye / Gaze Tracking | `float64` | degrees | [-145.9163, 96.0918] | Yes | Relative vertical gaze offset of eye pupils with respect to head forward direction. |
| `EyeHeadOffset` | Eye / Gaze Tracking | `float64` | degrees | [0.0312, 204.7647] | Yes | Euclidean angular distance between head forward vector and gaze vector: sqrt(GazeRelativeH^2 + GazeRelativeV^2). |
| `AbsoluteGazeYaw` | Eye / Gaze Tracking | `float64` | degrees | [-179.993, 179.9963] | Yes | True world gaze horizontal angle; falls back safely to HeadYaw when pupil tracking is lost. |
| `AbsoluteGazePitch` | Eye / Gaze Tracking | `float64` | degrees | [-87.7987, 61.9085] | Yes | True world gaze vertical angle; falls back safely to HeadPitch when pupil tracking is lost. |
| `LeftOpenness` | Eye State | `float64` | normalized [0-1] | [0.0, 1.0] | Yes | Left eye aperture openness degree (0.0 = completely closed, 1.0 = fully open). |
| `RightOpenness` | Eye State | `float64` | normalized [0-1] | [0.0, 1.0] | Yes | Right eye aperture openness degree (0.0 = completely closed, 1.0 = fully open). |
| `BlinkL` | Eye State | `int64` | binary flag | [0.0, 1.0] | Yes | Binary indicator for left eye blink (1 = blinking, 0 = not blinking). |
| `BlinkR` | Eye State | `int64` | binary flag | [0.0, 1.0] | Yes | Binary indicator for right eye blink (1 = blinking, 0 = not blinking). |
| `FaceIsValid` | Face Tracking | `int64` | binary flag | [1.0, 1.0] | No | Hardware tracking validity indicator for facial tracker (1 = valid facial capture). |
| `SmileL` | Facial Movement | `float64` | normalized [0-1] | [0.0, 0.077] | Yes | Left lip corner pull / smile facial muscle blendshape weight. |
| `SmileR` | Facial Movement | `float64` | normalized [0-1] | [0.0, 0.077] | Yes | Right lip corner pull / smile facial muscle blendshape weight. |
| `BrowDownL` | Facial Movement | `float64` | normalized [0-1] | [0.0, 0.394] | Yes | Left eyebrow depressor blendshape weight (furrowing / frowning expression). |
| `BrowDownR` | Facial Movement | `float64` | normalized [0-1] | [0.0, 0.345] | Yes | Right eyebrow depressor blendshape weight (furrowing / frowning expression). |
| `JawOpen` | Facial Movement | `float64` | normalized [0-1] | [0.0, 0.416] | Yes | Mandibular jaw opening blendshape weight (speaking, surprise, mouth agape). |
| `EyeClosedL` | Facial Movement | `float64` | normalized [0-1] | [0.0, 1.0] | Yes | Facial tracker estimation of left eyelid closure blendshape weight. |
| `EyeClosedR` | Facial Movement | `float64` | normalized [0-1] | [0.0, 1.0] | Yes | Facial tracker estimation of right eyelid closure blendshape weight. |
| `FaceActivity` | Facial Movement | `float64` | scalar index | [0.0017, 0.0987] | Yes | Aggregate facial movement energy / expressive dynamics metric across all facial blendshapes. |


## 6. Selected Prediction Targets & Geometry

### Head Tracking Targets (Euler Angles in Degrees)

- `HeadYaw`: Horizontal angle [-180°, +180°]. Requires circular wrapping handling.

- `HeadPitch`: Vertical elevation angle [-90°, +90°].

- `HeadRoll`: Lateral head tilt.


### Eye / Gaze Targets

- `GazeYaw` & `GazePitch` (World Gaze Orientation): `GazeYaw = HeadYaw + GazeRelativeH`, `GazePitch = HeadPitch + GazeRelativeV`.

- `AbsoluteGazeYaw` & `AbsoluteGazePitch`: Robust world gaze orientation that safely falls back to Head orientation if eye tracking is lost.

- `GazeRelativeH` & `GazeRelativeV`: Pupillary deviation relative to head direction.

- `EyeHeadOffset`: Angular distance between head and gaze vectors: `sqrt(GazeRelativeH^2 + GazeRelativeV^2)`.

- `LeftOpenness` & `RightOpenness`: Degree of eye aperture [0.0 - 1.0].


### Facial Movement Targets

- `SmileL` & `SmileR`: Lip corner pull intensity [0.0 - 1.0].

- `BrowDownL` & `BrowDownR`: Eyebrow frowning / furrowing intensity [0.0 - 1.0].

- `JawOpen`: Mouth opening intensity [0.0 - 1.0].

- `EyeClosedL` & `EyeClosedR`: Eyelid closure intensity [0.0 - 1.0].

- `FaceActivity`: Combined facial muscle activity index.


## 7. Stage 1 Verification Status

- [x] Inspected all 25 participant folders (P001 to P025).

- [x] Verified all 14 video directories per participant (350 video folders).

- [x] Confirmed 350 `combined.csv` files exist with non-zero sizes.

- [x] Verified 0 missing combinations.

- [x] Verified identical 26 columns across all 350 files.

- [x] Confirmed 0 NaN/null values in all 5,816,644 rows.

- [x] Audited sampling intervals (~72 Hz tracker, ~30 fps video).

- [x] Clarified Timestamp vs PlaybackTime behavior and duplicate resolution.

- [x] Formulated ground-truth target definitions and mathematical relationships.
