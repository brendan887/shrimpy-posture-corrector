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

For this MVP, stand mostly front-facing to the camera. The angles use MediaPipe world landmarks and a torso-relative frame, but flexion direction assumes webcam depth points toward the camera.

Useful options:

```bash
python live_pose_full.py --download-model-only
python live_pose_full.py --camera 1
python live_pose_full.py --width 640 --height 480
python live_pose_full.py --min-confidence 0.65
python live_pose_full.py --angle-smoothing 0.15
```

Press `q` or `Esc` to quit.

If macOS blocks the camera, allow Terminal, your IDE, or the Python launcher under **System Settings > Privacy & Security > Camera**.
