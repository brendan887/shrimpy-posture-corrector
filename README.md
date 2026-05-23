# Shrimpy Posture Corrector MVP

Minimal live webcam MVP for MediaPipe Pose Landmarker **Full** on an Apple Silicon Mac.

## Setup

```bash
conda env create -f environment.yml
conda activate shrimpy-pose
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate shrimpy-pose
```

## Run

```bash
python live_pose_full.py
```

The script downloads `models/pose_landmarker_full.task` on first run, opens your webcam, and shows a simple skeleton overlay in an OpenCV window.

The overlay includes live shoulder angles for each arm:

- `flex`: flexion in the torso sagittal plane, where `0` is arm down, `90` is straight out in front, and `180` is overhead.
- `abd`: abduction in the torso frontal plane, where `0` is arm down, `90` is straight out to the side, and `180` is overhead.

For this MVP, start with a front-facing or 45-degree front-side camera view. The angles use MediaPipe world landmarks and a torso-relative frame, with optional in-session calibration to make the front, side, and down axes more personal.

## Camera View

Front view is best for abduction. A 45-degree view is often better for flexion because the forward arm raise is less hidden in camera depth.

Recommended 45-degree test setup:

- Put the camera at chest-ish height if possible.
- Place it 6-8 feet away so shoulders, hips, and elbows stay visible.
- Use `left-45` if the camera is at your left-front diagonal.
- Use `right-45` if the camera is at your right-front diagonal.
- Keep your torso facing your exercise direction, not turned toward the camera.

## Calibration

Press `c` in the webcam window to start calibration. The UI will guide you through:

- Neutral: stand tall with arms relaxed down.
- Forward raise: raise both arms straight forward to shoulder height.
- Side raise: raise both arms out to your sides like a T.

Each pose waits for visible landmarks and stillness, then shows a countdown before capture. Press `Space` during calibration to manually capture the current pose if needed.

The saved accuracy roadmap is in `docs/accuracy_plan.md`.

## Diagnostic Test Capture

Press `t` to start a repeatable diagnostic sequence. The UI prompts:

- Arms down.
- Arms straight in front.
- Arms out to the side in a T.
- Arms directly overhead.

For each prompt, hold the pose still. The app shows a countdown and captures automatically. Press `Space` if you want to manually capture the current pose immediately. The app saves a PNG image plus a JSON file in `captures/`. The JSON includes the image path, view mode, calibrated/uncalibrated measurement mode, raw MediaPipe landmarks, raw/smoothed measurement vectors, and measured flexion/abduction angles for a short window before and after the capture mark. This works whether calibration is active or not.

Useful options:

```bash
python live_pose_full.py --download-model-only
python live_pose_full.py --camera 1
python live_pose_full.py --width 640 --height 480
python live_pose_full.py --min-confidence 0.65
python live_pose_full.py --angle-smoothing 0.15
python live_pose_full.py --calibration-stillness 0.06
python live_pose_full.py --view left-45
python live_pose_full.py --test-window-seconds 3
python live_pose_full.py --test-countdown-seconds 2 --test-stillness 0.06
```

Press `q` or `Esc` to quit.

If macOS blocks the camera, allow Terminal, your IDE, or the Python launcher under **System Settings > Privacy & Security > Camera**.
