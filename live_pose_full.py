from __future__ import annotations

import argparse
import json
import math
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar
from pathlib import Path
from statistics import mean, median, pstdev

import cv2
import mediapipe as mp

from bridge import DEFAULT_HOST, DEFAULT_PORT, StatusClient
from pose_ui import VIEW_MODES, _detect_screen_size, render_frame


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_MODEL_PATH = Path("models/pose_landmarker_full.task")

LEFT_ARM = {"name": "L", "shoulder": 11, "elbow": 13}
RIGHT_ARM = {"name": "R", "shoulder": 12, "elbow": 14}
CALIBRATION_LANDMARKS = (11, 12, 13, 14, 23, 24)
ROM_LANDMARKS = {
    "L": {"shoulder": 11, "elbow": 13, "wrist": 15, "index": 19, "pinky": 17},
    "R": {"shoulder": 12, "elbow": 14, "wrist": 16, "index": 20, "pinky": 18},
}
LEFT_ARROW_KEYS = {81, 2424832, 63234}
RIGHT_ARROW_KEYS = {83, 2555904, 63235}


@dataclass(frozen=True)
class CalibrationStep:
    name: str
    instruction: str


@dataclass
class CalibrationAxes:
    down: tuple[float, float, float]
    right: tuple[float, float, float]
    forward: tuple[float, float, float]
    left_side: tuple[float, float, float]
    right_side: tuple[float, float, float]


@dataclass
class CalibrationState:
    countdown_seconds: float = 3.0
    stable_frames: int = 8
    stillness_threshold: float = 0.045
    max_unstable_frames: int = 10
    active: bool = False
    current_step: int = 0
    countdown_started_at: float | None = None
    unstable_frames: int = 0
    status: str = "Press c to calibrate"
    samples: dict[str, dict[int, tuple[float, float, float]]] = field(default_factory=dict)
    axes: CalibrationAxes | None = None
    landmark_buffer: deque = field(default_factory=lambda: deque(maxlen=30))
    steps: tuple[CalibrationStep, ...] = (
        CalibrationStep(
            "neutral",
            "Stand tall, arms relaxed at your sides, facing front.",
        ),
        CalibrationStep(
            "forward",
            "Raise both arms straight forward to shoulder height.",
        ),
        CalibrationStep(
            "side",
            "Raise both arms straight out to your sides like a T.",
        ),
    )

    def start(self) -> None:
        self.active = True
        self.current_step = 0
        self.countdown_started_at = None
        self.unstable_frames = 0
        self.samples.clear()
        self.landmark_buffer.clear()
        self.status = "Calibration started"

    def step(self) -> CalibrationStep | None:
        if not self.active or self.current_step >= len(self.steps):
            return None
        return self.steps[self.current_step]

    def reset_countdown(self, status: str) -> None:
        self.countdown_started_at = None
        self.unstable_frames = 0
        self.status = status

    def advance(self, sample: dict[int, tuple[float, float, float]]) -> None:
        step = self.step()
        if step is None:
            return

        self.samples[step.name] = sample
        self.current_step += 1
        self.countdown_started_at = None
        self.unstable_frames = 0
        self.landmark_buffer.clear()

        if self.current_step >= len(self.steps):
            self.axes = build_calibration_axes(self.samples)
            self.active = False
            self.status = (
                "Calibration complete"
                if self.axes is not None
                else "Calibration failed; press c to retry"
            )
        else:
            self.status = f"Captured {step.name}. Next pose..."


@dataclass(frozen=True)
class TestStep:
    name: str
    instruction: str
    expected: str


@dataclass
class TestCaptureState:
    pre_seconds: float = 2.0
    post_seconds: float = 2.0
    countdown_seconds: float = 3.0
    stable_frames: int = 8
    stillness_threshold: float = 0.045
    max_unstable_frames: int = 10
    active: bool = False
    current_step: int = 0
    countdown_started_at: float | None = None
    unstable_frames: int = 0
    status: str = "Press t to start test capture"
    session_id: str = ""
    capture_index: int = 0
    pending_capture: dict | None = None
    post_until: float = 0.0
    last_saved_json_path: Path | None = None
    last_saved_image_path: Path | None = None
    stillness_buffer: deque = field(default_factory=lambda: deque(maxlen=30))
    steps: tuple[TestStep, ...] = (
        TestStep(
            "arms_down",
            "Hold both arms relaxed straight down.",
            "Expected: flex near 0 deg, abd near 0 deg.",
        ),
        TestStep(
            "arms_forward",
            "Hold both arms straight in front at shoulder height.",
            "Expected: flex near 90 deg, abd near 0 deg.",
        ),
        TestStep(
            "arms_side_t",
            "Hold both arms straight out to the side like a T.",
            "Expected: flex near 0 deg, abd near 90 deg.",
        ),
        TestStep(
            "arms_overhead",
            "Hold both arms directly overhead.",
            "Expected: flex and/or abd near 180 deg.",
        ),
    )

    def start(self) -> None:
        self.active = True
        self.current_step = 0
        self.countdown_started_at = None
        self.unstable_frames = 0
        self.capture_index = 0
        self.pending_capture = None
        self.post_until = 0.0
        self.last_saved_json_path = None
        self.last_saved_image_path = None
        self.stillness_buffer.clear()
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.status = "Test started: hold pose 1"

    def step(self) -> TestStep | None:
        if not self.active or self.current_step >= len(self.steps):
            return None
        return self.steps[self.current_step]

    def finish_if_done(self) -> None:
        if self.current_step >= len(self.steps) and self.pending_capture is None:
            self.active = False
            self.countdown_started_at = None
            self.unstable_frames = 0
            self.stillness_buffer.clear()
            self.status = "Test complete. Press t to repeat."

    def reset_countdown(self, status: str) -> None:
        self.countdown_started_at = None
        self.unstable_frames = 0
        self.status = status


@dataclass(frozen=True)
class ROMSweepStep:
    name: str
    arm: str
    instruction: str


def default_rom_arm_for_view(view_mode: str) -> str:
    # The image is mirrored before MediaPipe inference, so the camera-side
    # landmark label is opposite the physical side-view name.
    camera_side_by_view = {
        "right-45": "L",
        "right-side": "L",
        "left-45": "R",
        "left-side": "R",
    }
    return camera_side_by_view.get(view_mode, "L")


def rom_step_for_arm(view_mode: str, arm: str) -> ROMSweepStep:
    view_slug = view_mode.replace("-", "_")
    arm_slug = "left" if arm == "L" else "right"
    return ROMSweepStep(
        f"{view_slug}_{arm_slug}_flexion_sweep",
        arm,
        (
            f"{arm} landmark arm: start down, sweep forward/up overhead, "
            "continue behind head, then press Space to stop."
        ),
    )


def rom_steps_for_view(view_mode: str) -> tuple[ROMSweepStep, ...]:
    return (
        rom_step_for_arm(view_mode, default_rom_arm_for_view(view_mode)),
    )


@dataclass
class ROMSweepState:
    duration_seconds: float = 10.0
    active: bool = False
    recording: bool = False
    current_step: int = 0
    session_id: str = ""
    capture_index: int = 0
    started_at: float = 0.0
    view_mode: str = "front"
    selected_arm: str = "L"
    status: str = "Press r to start ROM sweep"
    samples: list[dict] = field(default_factory=list)
    key_frames: dict[str, dict] = field(default_factory=dict)
    frame_samples: list[dict] = field(default_factory=list)
    last_image_plane_angles: dict | None = None
    last_saved_json_path: Path | None = None
    steps: tuple[ROMSweepStep, ...] = (
        ROMSweepStep(
            "left_flexion_sweep",
            "L",
            "Left arm only: start down, sweep forward/up overhead, then return down.",
        ),
        ROMSweepStep(
            "right_flexion_sweep",
            "R",
            "Right arm only: start down, sweep forward/up overhead, then return down.",
        ),
    )

    def start(self, view_mode: str) -> None:
        self.active = True
        self.recording = False
        self.view_mode = view_mode
        self.selected_arm = default_rom_arm_for_view(view_mode)
        self.steps = rom_steps_for_view(view_mode)
        self.current_step = 0
        self.capture_index = 0
        self.started_at = 0.0
        self.samples.clear()
        self.key_frames.clear()
        self.frame_samples.clear()
        self.last_image_plane_angles = None
        self.last_saved_json_path = None
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        step = self.step()
        if step is None:
            self.status = "No ROM sweep configured for this view."
        elif view_mode == "front":
            self.status = f"ROM test started. Arrows select arm; Space records {self.selected_arm}."
        else:
            self.status = f"ROM test started for {view_mode}. Arrows select arm; Space records {self.selected_arm}."

    def step(self) -> ROMSweepStep | None:
        if not self.active or self.current_step >= len(self.steps):
            return None
        return self.steps[self.current_step]

    def select_arm(self, arm: str) -> None:
        if arm not in {"L", "R"} or self.recording:
            return
        self.selected_arm = arm
        self.steps = (rom_step_for_arm(self.view_mode, arm),)
        self.current_step = 0
        self.samples.clear()
        self.key_frames.clear()
        self.frame_samples.clear()
        self.last_image_plane_angles = None
        self.status = f"Selected {arm} landmark arm. Press Space to record."

    def begin_recording(self, now: float) -> None:
        step = self.step()
        if step is None:
            self.finish_if_done()
            return
        self.recording = True
        self.started_at = now
        self.samples.clear()
        self.key_frames.clear()
        self.frame_samples.clear()
        self.last_image_plane_angles = None
        self.status = f"Recording {step.name}. Press Space again to stop."

    def finish_if_done(self) -> None:
        if self.current_step >= len(self.steps):
            self.active = False
            self.recording = False
            self.status = "ROM sweep complete. Press r to repeat."


def ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe Pose Landmarker Full model to {model_path}...")
    urllib.request.urlretrieve(MODEL_URL, model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Pose Landmarker Full in live webcam mode."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to pose_landmarker_full.task. Downloaded if missing.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum detection, presence, and tracking confidence.",
    )
    parser.add_argument(
        "--download-model-only",
        action="store_true",
        help="Download the Full model, then exit without opening the webcam.",
    )
    parser.add_argument(
        "--angle-smoothing",
        type=float,
        default=0.25,
        help="EMA factor for measurement vectors. 0 is very smooth, 1 is no smoothing.",
    )
    parser.add_argument(
        "--calibration-stillness",
        type=float,
        default=0.045,
        help="Normalized image-motion tolerance for calibration. Higher is more forgiving.",
    )
    parser.add_argument(
        "--calibration-stable-frames",
        type=int,
        default=8,
        help="Recent stable frames needed before calibration countdown can start.",
    )
    parser.add_argument(
        "--view",
        choices=VIEW_MODES,
        default="front",
        help="Camera placement hint for UI guidance. 45-degree modes still use torso-relative math.",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("captures"),
        help="Directory for diagnostic test capture JSON files.",
    )
    parser.add_argument(
        "--test-window-seconds",
        type=float,
        default=2.0,
        help="Seconds before and after each test capture mark to save.",
    )
    parser.add_argument(
        "--test-countdown-seconds",
        type=float,
        default=3.0,
        help="Hold-still countdown before automatic diagnostic capture.",
    )
    parser.add_argument(
        "--test-stillness",
        type=float,
        default=0.045,
        help="Normalized image-motion tolerance for diagnostic auto-capture.",
    )
    parser.add_argument(
        "--test-stable-frames",
        type=int,
        default=8,
        help="Recent stable frames needed before diagnostic countdown can start.",
    )
    parser.add_argument(
        "--rom-sweep-seconds",
        type=float,
        default=10.0,
        help="Deprecated. ROM sweeps now record until Space is pressed again.",
    )
    parser.add_argument(
        "--rom-arm",
        choices=("L", "R"),
        default=None,
        help="Pre-select the ROM sweep arm without using the left/right arrow keys.",
    )
    parser.add_argument(
        "--robot-host",
        default=DEFAULT_HOST,
        help="Host where the piper_sequence status server is running.",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for the piper_sequence status server.",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Disable the robot status client (no connection attempts, no panel).",
    )
    parser.add_argument(
        "--no-visualize-on-quit",
        action="store_true",
        help="Skip regenerating capture summary charts when the app exits.",
    )
    return parser.parse_args()


def is_visible(landmark, threshold: float) -> bool:
    visibility = getattr(landmark, "visibility", 1.0)
    presence = getattr(landmark, "presence", 1.0)
    return visibility >= threshold and presence >= threshold


def vec_from_landmark(landmark) -> tuple[float, float, float]:
    return landmark.x, landmark.y, landmark.z


def v_add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def v_sub(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def v_scale(v, scalar: float):
    return v[0] * scalar, v[1] * scalar, v[2] * scalar


def v_dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    )


def v_norm(v) -> float:
    return math.sqrt(v_dot(v, v))


def v_normalize(v):
    length = v_norm(v)
    if length < 1e-6:
        return None
    return v_scale(v, 1.0 / length)


def image_plane_flexion_angle(shoulder, distal) -> float | None:
    if shoulder is None or distal is None:
        return None

    dx = distal.x - shoulder.x
    dy = distal.y - shoulder.y
    if math.hypot(dx, dy) < 1e-6:
        return None

    # Image-space convention for fixed side-ish camera views:
    # 0=down, 90=left in frame, 180=up, >180=past vertical toward frame-right.
    angle = math.degrees(math.atan2(-dx, dy))
    if angle < -90.0:
        angle += 360.0
    return angle


def image_plane_elbow_angle(shoulder, elbow, wrist) -> float | None:
    if shoulder is None or elbow is None or wrist is None:
        return None

    upper = (shoulder.x - elbow.x, shoulder.y - elbow.y)
    forearm = (wrist.x - elbow.x, wrist.y - elbow.y)
    upper_norm = math.hypot(*upper)
    forearm_norm = math.hypot(*forearm)
    if upper_norm < 1e-6 or forearm_norm < 1e-6:
        return None

    dot = (upper[0] * forearm[0]) + (upper[1] * forearm[1])
    cosine = min(max(dot / (upper_norm * forearm_norm), -1.0), 1.0)
    return math.degrees(math.acos(cosine))


def v_lerp(a, b, alpha: float):
    return (
        (alpha * a[0]) + ((1.0 - alpha) * b[0]),
        (alpha * a[1]) + ((1.0 - alpha) * b[1]),
        (alpha * a[2]) + ((1.0 - alpha) * b[2]),
    )


def project_onto_plane(v, normal):
    return v_sub(v, v_scale(normal, v_dot(v, normal)))


def angle_between(a, b) -> float | None:
    a_norm = v_normalize(a)
    b_norm = v_normalize(b)
    if a_norm is None or b_norm is None:
        return None
    dot = min(max(v_dot(a_norm, b_norm), -1.0), 1.0)
    return math.degrees(math.acos(dot))


def signed_plane_angle(vector, zero_axis, positive_axis, plane_normal) -> float | None:
    projected = v_normalize(project_onto_plane(vector, plane_normal))
    if projected is None:
        return None

    degrees = math.degrees(
        math.atan2(v_dot(projected, positive_axis), v_dot(projected, zero_axis))
    )
    return degrees


def average_vectors(vectors):
    if not vectors:
        return None
    total = (0.0, 0.0, 0.0)
    for vector in vectors:
        total = v_add(total, vector)
    return v_scale(total, 1.0 / len(vectors))


def arm_landmarks_visible(image_landmarks, arm, threshold: float) -> bool:
    needed = (arm["shoulder"], arm["elbow"], 11, 12, 23, 24)
    return all(is_visible(image_landmarks[idx], threshold) for idx in needed)


def get_landmark_samples(result, visibility_threshold: float):
    if (
        not result
        or not result.pose_landmarks
        or not result.pose_world_landmarks
    ):
        return None

    image_landmarks = result.pose_landmarks[0]
    world_landmarks = result.pose_world_landmarks[0]
    if not all(is_visible(image_landmarks[idx], visibility_threshold) for idx in CALIBRATION_LANDMARKS):
        return None

    image_sample = {
        idx: (
            image_landmarks[idx].x,
            image_landmarks[idx].y,
            getattr(image_landmarks[idx], "z", 0.0),
        )
        for idx in CALIBRATION_LANDMARKS
    }
    world_sample = {
        idx: vec_from_landmark(world_landmarks[idx])
        for idx in CALIBRATION_LANDMARKS
    }
    return image_sample, world_sample


def average_world_samples(samples) -> dict[int, tuple[float, float, float]]:
    averaged = {}
    for idx in CALIBRATION_LANDMARKS:
        averaged[idx] = average_vectors([sample[idx] for sample in samples])
    return averaged


def sample_stillness(image_samples) -> float | None:
    if len(image_samples) < 2:
        return None

    first = image_samples[0]
    last = image_samples[-1]
    distances = []
    for idx in CALIBRATION_LANDMARKS:
        delta = v_sub(last[idx], first[idx])
        distances.append(v_norm(delta))
    return sum(distances) / len(distances)


def torso_axes_from_sample(sample) -> tuple | None:
    left_shoulder = sample[11]
    right_shoulder = sample[12]
    left_hip = sample[23]
    right_hip = sample[24]
    shoulder_mid = v_scale(v_add(left_shoulder, right_shoulder), 0.5)
    hip_mid = v_scale(v_add(left_hip, right_hip), 0.5)

    up_axis = v_normalize(v_sub(shoulder_mid, hip_mid))
    right_axis = v_normalize(v_sub(right_shoulder, left_shoulder))
    if up_axis is None or right_axis is None:
        return None

    forward_axis = v_normalize(v_cross(right_axis, up_axis))
    if forward_axis is None:
        return None

    if forward_axis[2] > 0:
        forward_axis = v_scale(forward_axis, -1.0)

    return up_axis, right_axis, forward_axis


def upper_arm_vector(sample, arm):
    return v_normalize(v_sub(sample[arm["elbow"]], sample[arm["shoulder"]]))


def pose_matches_step(sample, step_name: str) -> tuple[bool, str]:
    axes = torso_axes_from_sample(sample)
    if axes is None:
        return False, "Move so shoulders and hips are visible"

    up_axis, _right_axis, _forward_axis = axes
    down_axis = v_scale(up_axis, -1.0)
    arm_angles = []
    for arm in (LEFT_ARM, RIGHT_ARM):
        upper_arm = upper_arm_vector(sample, arm)
        if upper_arm is None:
            return False, "Move so shoulders and elbows are visible"
        arm_angle = angle_between(upper_arm, down_axis)
        if arm_angle is not None:
            arm_angles.append(arm_angle)

    if step_name == "neutral" and any(angle > 55.0 for angle in arm_angles):
        return False, "Relax arms down at your sides"
    if step_name in {"forward", "side"} and any(angle < 45.0 for angle in arm_angles):
        return False, "Raise both arms away from your torso"

    return True, "Hold still"


def build_calibration_axes(samples: dict[str, dict[int, tuple[float, float, float]]]) -> CalibrationAxes | None:
    neutral = samples.get("neutral")
    forward = samples.get("forward")
    side = samples.get("side")
    if not neutral or not forward or not side:
        return None

    axes = torso_axes_from_sample(neutral)
    if axes is None:
        return None

    up_axis, right_axis, fallback_forward = axes
    down_axis = v_scale(up_axis, -1.0)

    forward_vectors = [
        upper_arm_vector(forward, LEFT_ARM),
        upper_arm_vector(forward, RIGHT_ARM),
    ]
    average_forward = average_vectors([vector for vector in forward_vectors if vector is not None])
    forward_axis = (
        v_normalize(project_onto_plane(average_forward, up_axis))
        if average_forward is not None
        else None
    )
    if forward_axis is None:
        forward_axis = fallback_forward
    elif v_dot(forward_axis, fallback_forward) < 0:
        forward_axis = v_scale(forward_axis, -1.0)

    left_side_axis = upper_arm_vector(side, LEFT_ARM)
    right_side_axis = upper_arm_vector(side, RIGHT_ARM)
    if left_side_axis is None:
        left_side_axis = v_scale(right_axis, -1.0)
    if right_side_axis is None:
        right_side_axis = right_axis

    left_side_axis = v_normalize(project_onto_plane(left_side_axis, up_axis))
    right_side_axis = v_normalize(project_onto_plane(right_side_axis, up_axis))
    if left_side_axis is None:
        left_side_axis = v_scale(right_axis, -1.0)
    if right_side_axis is None:
        right_side_axis = right_axis

    if v_dot(left_side_axis, v_scale(right_axis, -1.0)) < 0:
        left_side_axis = v_scale(left_side_axis, -1.0)
    if v_dot(right_side_axis, right_axis) < 0:
        right_side_axis = v_scale(right_side_axis, -1.0)

    return CalibrationAxes(
        down=down_axis,
        right=right_axis,
        forward=forward_axis,
        left_side=left_side_axis,
        right_side=right_side_axis,
    )


def update_calibration(calibration: CalibrationState, result, visibility_threshold: float, now: float) -> None:
    if not calibration.active:
        return

    step = calibration.step()
    if step is None:
        return

    samples = get_landmark_samples(result, visibility_threshold)
    if samples is None:
        calibration.landmark_buffer.clear()
        calibration.reset_countdown("Waiting for shoulders, elbows, and hips")
        return

    image_sample, world_sample = samples
    pose_ok, pose_status = pose_matches_step(world_sample, step.name)
    if not pose_ok:
        calibration.landmark_buffer.clear()
        calibration.reset_countdown(pose_status)
        return

    calibration.landmark_buffer.append((image_sample, world_sample))
    if len(calibration.landmark_buffer) < calibration.stable_frames:
        calibration.reset_countdown("Hold still")
        return

    recent = list(calibration.landmark_buffer)[-calibration.stable_frames:]
    stillness = sample_stillness([item[0] for item in recent])
    if stillness is None or stillness > calibration.stillness_threshold:
        calibration.unstable_frames += 1
        calibration.status = (
            f"Hold still: movement {stillness or 0.0:.3f}"
            f" / {calibration.stillness_threshold:.3f}"
        )
        if calibration.unstable_frames > calibration.max_unstable_frames:
            calibration.countdown_started_at = None
            calibration.unstable_frames = 0
        return

    calibration.unstable_frames = 0
    if calibration.countdown_started_at is None:
        calibration.countdown_started_at = now
        calibration.status = "Countdown started"
        return

    remaining = calibration.countdown_seconds - (now - calibration.countdown_started_at)
    calibration.status = f"Capturing in {max(math.ceil(remaining), 0)}"
    if remaining <= 0:
        calibration.advance(average_world_samples([item[1] for item in recent]))


def capture_calibration_now(calibration: CalibrationState, result, visibility_threshold: float) -> None:
    if not calibration.active:
        return

    samples = get_landmark_samples(result, visibility_threshold)
    if samples is None:
        calibration.status = "Cannot capture: landmarks are not visible"
        return

    _image_sample, world_sample = samples
    calibration.advance(world_sample)


def measurement_frame_from_result(result, visibility_threshold: float, calibration_axes: CalibrationAxes | None):
    if (
        not result
        or not result.pose_landmarks
        or not result.pose_world_landmarks
    ):
        return None

    image_landmarks = result.pose_landmarks[0]
    world_landmarks = result.pose_world_landmarks[0]

    left_shoulder = vec_from_landmark(world_landmarks[11])
    right_shoulder = vec_from_landmark(world_landmarks[12])
    left_hip = vec_from_landmark(world_landmarks[23])
    right_hip = vec_from_landmark(world_landmarks[24])
    shoulder_mid = v_scale(v_add(left_shoulder, right_shoulder), 0.5)
    hip_mid = v_scale(v_add(left_hip, right_hip), 0.5)

    if calibration_axes is None:
        up_axis = v_normalize(v_sub(shoulder_mid, hip_mid))
        right_axis = v_normalize(v_sub(right_shoulder, left_shoulder))
        if up_axis is None or right_axis is None:
            return None

        forward_axis = v_normalize(v_cross(right_axis, up_axis))
        if forward_axis is None:
            return None

        # Front/45-degree MVP assumption: anatomical front points roughly toward the camera.
        if forward_axis[2] > 0:
            forward_axis = v_scale(forward_axis, -1.0)

        down_axis = v_scale(up_axis, -1.0)
        left_side_axis = v_scale(right_axis, -1.0)
        right_side_axis = right_axis
    else:
        down_axis = calibration_axes.down
        right_axis = calibration_axes.right
        forward_axis = calibration_axes.forward
        left_side_axis = calibration_axes.left_side
        right_side_axis = calibration_axes.right_side

    frame = {
        "down": down_axis,
        "right": right_axis,
        "forward": forward_axis,
        "L_side": left_side_axis,
        "R_side": right_side_axis,
        "L_upper_arm": None,
        "R_upper_arm": None,
    }
    for arm in (LEFT_ARM, RIGHT_ARM):
        if not arm_landmarks_visible(image_landmarks, arm, visibility_threshold):
            continue

        shoulder = vec_from_landmark(world_landmarks[arm["shoulder"]])
        elbow = vec_from_landmark(world_landmarks[arm["elbow"]])
        upper_arm = v_normalize(v_sub(elbow, shoulder))
        frame[f"{arm['name']}_upper_arm"] = upper_arm

    return frame


def smooth_measurement_frame(current, previous, smoothing_factor: float):
    if current is None:
        return previous
    if previous is None:
        return current

    alpha = min(max(smoothing_factor, 0.0), 1.0)
    smoothed = {}
    for key, current_value in current.items():
        previous_value = previous.get(key)
        if current_value is None:
            smoothed[key] = previous_value
        elif previous_value is None:
            smoothed[key] = current_value
        else:
            smoothed[key] = v_normalize(v_lerp(current_value, previous_value, alpha))
    return smoothed


def angles_from_measurement_frame(frame) -> dict[str, dict[str, float | None]]:
    if frame is None:
        return {}

    angles = {}
    for arm_name in ("L", "R"):
        upper_arm = frame.get(f"{arm_name}_upper_arm")
        side_axis = frame.get(f"{arm_name}_side")
        if upper_arm is None or side_axis is None:
            angles[arm_name] = {"flexion": None, "abduction": None}
            continue

        # Flexion shares one plane_normal across both arms, so the raw signed
        # angle is mirrored on L. Negate L so both arms read +90 forward.
        flexion_value = signed_plane_angle(
            upper_arm,
            zero_axis=frame["down"],
            positive_axis=frame["forward"],
            plane_normal=frame["right"],
        )
        if arm_name == "L" and flexion_value is not None:
            flexion_value = -flexion_value

        angles[arm_name] = {
            "flexion": flexion_value,
            "abduction": signed_plane_angle(
                upper_arm,
                zero_axis=frame["down"],
                positive_axis=side_axis,
                plane_normal=frame["forward"],
            ),
        }

    return angles


def serialize_landmark(landmark) -> dict[str, float]:
    serialized = {
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
    }
    for attr in ("visibility", "presence"):
        value = getattr(landmark, attr, None)
        if value is not None:
            serialized[attr] = value
    return serialized


def serialize_landmark_lists(landmark_lists) -> list[list[dict[str, float]]]:
    if not landmark_lists:
        return []
    return [
        [serialize_landmark(landmark) for landmark in landmarks]
        for landmarks in landmark_lists
    ]


def serialize_vector_map(vector_map) -> dict:
    if vector_map is None:
        return {}
    return {
        key: list(value) if value is not None else None
        for key, value in vector_map.items()
    }


def make_diagnostic_sample(
    *,
    now: float,
    result_timestamp_ms: int,
    result,
    angles,
    raw_measurement_frame,
    smoothed_measurement_frame,
    pose_count: int,
    view_mode: str,
    calibration: CalibrationState,
    test_step_name: str | None,
    image_sample=None,
    image_plane_angles=None,
) -> dict:
    return {
        "time_monotonic": now,
        "result_timestamp_ms": result_timestamp_ms,
        "view": view_mode,
        "pose_count": pose_count,
        "calibration_active": calibration.active,
        "calibration_ready": calibration.axes is not None,
        "test_step": test_step_name,
        "angles": angles,
        "image_plane_angles": image_plane_angles or {},
        "image_sample": {
            str(idx): list(value)
            for idx, value in image_sample.items()
        } if image_sample else None,
        "raw_measurement_frame": serialize_vector_map(raw_measurement_frame),
        "smoothed_measurement_frame": serialize_vector_map(smoothed_measurement_frame),
        "pose_landmarks": serialize_landmark_lists(
            result.pose_landmarks if result else []
        ),
        "pose_world_landmarks": serialize_landmark_lists(
            result.pose_world_landmarks if result else []
        ),
    }


def get_world_landmark(result, idx: int):
    if not result or not result.pose_world_landmarks:
        return None
    world_landmarks = result.pose_world_landmarks[0]
    if idx >= len(world_landmarks):
        return None
    return vec_from_landmark(world_landmarks[idx])


def get_image_landmark(result, idx: int):
    if not result or not result.pose_landmarks:
        return None
    image_landmarks = result.pose_landmarks[0]
    if idx >= len(image_landmarks):
        return None
    return image_landmarks[idx]


def landmark_confidence(result, idx: int) -> float | None:
    landmark = get_image_landmark(result, idx)
    if landmark is None:
        return None
    visibility = getattr(landmark, "visibility", 1.0)
    presence = getattr(landmark, "presence", 1.0)
    return min(visibility, presence)


def midpoint(a, b):
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return v_scale(v_add(a, b), 0.5)


def image_midpoint(a, b):
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a

    class Point:
        pass

    point = Point()
    point.x = (a.x + b.x) * 0.5
    point.y = (a.y + b.y) * 0.5
    point.z = (getattr(a, "z", 0.0) + getattr(b, "z", 0.0)) * 0.5
    return point


def image_plane_angles_from_result(result, arm_name: str) -> dict:
    landmarks = ROM_LANDMARKS[arm_name]
    shoulder = get_image_landmark(result, landmarks["shoulder"])
    elbow = get_image_landmark(result, landmarks["elbow"])
    wrist = get_image_landmark(result, landmarks["wrist"])
    index = get_image_landmark(result, landmarks["index"])
    pinky = get_image_landmark(result, landmarks["pinky"])
    hand = image_midpoint(index, pinky) or wrist

    confidence_indices = [
        landmarks["shoulder"],
        landmarks["elbow"],
        landmarks["wrist"],
        landmarks["index"],
        landmarks["pinky"],
    ]
    confidences = [
        confidence
        for confidence in (landmark_confidence(result, idx) for idx in confidence_indices)
        if confidence is not None
    ]

    return {
        "source": "pose_landmarks_image_plane",
        "convention": "0_down_90_left_180_up",
        "humerus_flexion": image_plane_flexion_angle(shoulder, elbow),
        "reach_flexion": image_plane_flexion_angle(shoulder, hand or wrist or elbow),
        "elbow_angle": image_plane_elbow_angle(shoulder, elbow, wrist),
        "confidence": min(confidences) if confidences else None,
    }


def all_image_plane_angles_from_result(result) -> dict[str, dict]:
    return {
        "L": image_plane_angles_from_result(result, "L"),
        "R": image_plane_angles_from_result(result, "R"),
    }


def rom_vectors_from_result(result, arm_name: str) -> dict:
    landmarks = ROM_LANDMARKS[arm_name]
    shoulder = get_world_landmark(result, landmarks["shoulder"])
    elbow = get_world_landmark(result, landmarks["elbow"])
    wrist = get_world_landmark(result, landmarks["wrist"])
    index = get_world_landmark(result, landmarks["index"])
    pinky = get_world_landmark(result, landmarks["pinky"])
    hand = midpoint(index, pinky) or wrist

    humerus = v_normalize(v_sub(elbow, shoulder)) if shoulder and elbow else None
    reach_target = hand or wrist or elbow
    reach = v_normalize(v_sub(reach_target, shoulder)) if shoulder and reach_target else None

    confidence_indices = [
        landmarks["shoulder"],
        landmarks["elbow"],
        landmarks["wrist"],
        landmarks["index"],
        landmarks["pinky"],
    ]
    confidences = [
        confidence
        for confidence in (landmark_confidence(result, idx) for idx in confidence_indices)
        if confidence is not None
    ]

    return {
        "humerus": list(humerus) if humerus else None,
        "reach": list(reach) if reach else None,
        "shoulder": list(shoulder) if shoulder else None,
        "elbow": list(elbow) if elbow else None,
        "wrist": list(wrist) if wrist else None,
        "hand": list(hand) if hand else None,
        "confidence": min(confidences) if confidences else None,
    }


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * (percent / 100.0)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + ((upper_value - lower_value) * weight)


def nearest_sample(samples: list[dict], target_time: float) -> dict | None:
    if not samples:
        return None
    return min(
        samples,
        key=lambda sample: abs(sample.get("time_monotonic", target_time) - target_time),
    )


def summarize_angle_values(values: list[float], total_samples: int) -> dict:
    if not values:
        return {
            "prevailing_angle": None,
            "min": None,
            "max": None,
            "range": None,
            "mean": None,
            "stdev": None,
            "p10": None,
            "p90": None,
            "valid_samples": 0,
            "total_samples": total_samples,
            "valid_ratio": 0.0 if total_samples else None,
        }

    min_value = min(values)
    max_value = max(values)
    return {
        "prevailing_angle": median(values),
        "min": min_value,
        "max": max_value,
        "range": max_value - min_value,
        "mean": mean(values),
        "stdev": pstdev(values) if len(values) > 1 else 0.0,
        "p10": percentile(values, 10),
        "p90": percentile(values, 90),
        "valid_samples": len(values),
        "total_samples": total_samples,
        "valid_ratio": len(values) / total_samples if total_samples else None,
    }


def summarize_capture(capture: dict) -> dict:
    samples = capture.get("pre_samples", []) + capture.get("post_samples", [])
    marked_sample = nearest_sample(samples, capture.get("marked_at", 0.0))
    times = [
        sample["time_monotonic"]
        for sample in samples
        if sample.get("time_monotonic") is not None
    ]

    angles = {}
    for arm_name in ("L", "R"):
        angles[arm_name] = {}
        for angle_name in ("flexion", "abduction"):
            values = []
            for sample in samples:
                value = (
                    sample.get("angles", {})
                    .get(arm_name, {})
                    .get(angle_name)
                )
                if value is not None:
                    values.append(value)
            angles[arm_name][angle_name] = summarize_angle_values(
                values,
                total_samples=len(samples),
            )

    return {
        "source": "displayed_smoothed_angles",
        "prevailing_method": "median",
        "capture_window_sample_count": len(samples),
        "capture_window_duration_seconds": (
            max(times) - min(times)
            if len(times) >= 2
            else 0.0
            if times
            else None
        ),
        "marked_sample_time_monotonic": (
            marked_sample.get("time_monotonic")
            if marked_sample
            else None
        ),
        "marked_angles": (
            marked_sample.get("angles")
            if marked_sample
            else None
        ),
        "angles": angles,
    }


def sample_angle(sample: dict, arm_name: str, angle_name: str) -> float | None:
    return (
        sample.get("angles", {})
        .get(arm_name, {})
        .get(angle_name)
    )


def update_rom_key_frames(rom: ROMSweepState, sample: dict, frame) -> None:
    step = rom.step()
    if step is None:
        return

    sample_index = len(rom.samples) - 1
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        rom.frame_samples.append({
            "sample_index": sample_index,
            "sample": sample,
            "image_bytes": encoded.tobytes(),
            "extension": "jpg",
            "source": "recorded_sample",
        })

    flexion = sample_angle(sample, step.arm, "flexion")
    if flexion is None:
        return

    if "start" not in rom.key_frames:
        rom.key_frames["start"] = {
            "sample_index": sample_index,
            "sample": sample,
            "frame": frame.copy(),
            "source": "live_start",
        }

    current_min = rom.key_frames.get("min_flexion", {}).get("sample")
    if current_min is None or flexion < sample_angle(current_min, step.arm, "flexion"):
        rom.key_frames["min_flexion"] = {
            "sample_index": sample_index,
            "sample": sample,
            "frame": frame.copy(),
            "source": "live_min_flexion",
        }

    current_max = rom.key_frames.get("max_flexion", {}).get("sample")
    if current_max is None or flexion > sample_angle(current_max, step.arm, "flexion"):
        rom.key_frames["max_flexion"] = {
            "sample_index": sample_index,
            "sample": sample,
            "frame": frame.copy(),
            "source": "live_max_flexion",
        }

    current_near_90 = rom.key_frames.get("near_90", {}).get("sample")
    if (
        current_near_90 is None
        or abs(flexion - 90.0) < abs(sample_angle(current_near_90, step.arm, "flexion") - 90.0)
    ):
        rom.key_frames["near_90"] = {
            "sample_index": sample_index,
            "sample": sample,
            "frame": frame.copy(),
            "source": "live_near_90",
        }

    rom.key_frames["end"] = {
        "sample_index": sample_index,
        "sample": sample,
        "frame": frame.copy(),
        "source": "live_end",
    }


def summarize_rom_sweep(recording: dict) -> dict:
    arm = recording["arm"]
    samples = recording.get("samples", [])
    flexion_values = [
        sample_angle(sample, arm, "flexion")
        for sample in samples
        if sample_angle(sample, arm, "flexion") is not None
    ]
    abduction_values = [
        sample_angle(sample, arm, "abduction")
        for sample in samples
        if sample_angle(sample, arm, "abduction") is not None
    ]

    if not flexion_values:
        return {
            "arm": arm,
            "valid_samples": 0,
            "total_samples": len(samples),
            "valid_ratio": 0.0 if samples else None,
            "min_flexion": None,
            "max_flexion": None,
            "rom": None,
        }

    min_sample = min(
        samples,
        key=lambda sample: (
            sample_angle(sample, arm, "flexion")
            if sample_angle(sample, arm, "flexion") is not None
            else float("inf")
        ),
    )
    max_sample = max(
        samples,
        key=lambda sample: (
            sample_angle(sample, arm, "flexion")
            if sample_angle(sample, arm, "flexion") is not None
            else float("-inf")
        ),
    )
    min_flexion = sample_angle(min_sample, arm, "flexion")
    max_flexion = sample_angle(max_sample, arm, "flexion")
    start_time = recording.get("started_at", samples[0].get("time_monotonic", 0.0))

    return {
        "arm": arm,
        "source": "displayed_smoothed_angles",
        "sample_count": len(samples),
        "valid_samples": len(flexion_values),
        "valid_ratio": len(flexion_values) / len(samples) if samples else None,
        "duration_seconds": (
            samples[-1]["time_monotonic"] - samples[0]["time_monotonic"]
            if len(samples) >= 2
            else 0.0
        ),
        "min_flexion": min_flexion,
        "max_flexion": max_flexion,
        "rom": max_flexion - min_flexion if min_flexion is not None and max_flexion is not None else None,
        "start_flexion": sample_angle(samples[0], arm, "flexion"),
        "end_flexion": sample_angle(samples[-1], arm, "flexion"),
        "max_abduction_during_sweep": max(abduction_values) if abduction_values else None,
        "abduction_range_during_sweep": (
            max(abduction_values) - min(abduction_values)
            if abduction_values
            else None
        ),
        "min_flexion_time_offset": min_sample["time_monotonic"] - start_time,
        "max_flexion_time_offset": max_sample["time_monotonic"] - start_time,
        "min_flexion_abduction": sample_angle(min_sample, arm, "abduction"),
        "max_flexion_abduction": sample_angle(max_sample, arm, "abduction"),
        "flexion_stats": summarize_angle_values(flexion_values, len(samples)),
        "abduction_stats": summarize_angle_values(abduction_values, len(samples)),
    }


def vector_from_sample(sample: dict, arm: str, vector_name: str):
    value = (
        sample.get("rom_vectors", {})
        .get(arm, {})
        .get(vector_name)
    )
    return tuple(value) if value is not None else None


def image_plane_angle_from_sample(sample: dict, arm: str, angle_name: str) -> float | None:
    return (
        sample.get("image_plane_angles", {})
        .get(arm, {})
        .get(angle_name)
    )


def estimate_motion_plane_normal(vectors: list[tuple[float, float, float]]):
    if len(vectors) < 2:
        return None

    start = vectors[0]
    normals = []
    for vector in vectors[1:]:
        normal = v_normalize(v_cross(start, vector))
        if normal is not None:
            normals.append(normal)
    if not normals:
        return None

    reference = normals[0]
    aligned = [
        normal if v_dot(normal, reference) >= 0 else v_scale(normal, -1.0)
        for normal in normals
    ]
    return v_normalize(average_vectors(aligned))


def unwrap_degrees(values: list[float | None]) -> list[float | None]:
    unwrapped = []
    previous = None
    offset = 0.0
    for value in values:
        if value is None:
            unwrapped.append(None)
            continue
        adjusted = value + offset
        if previous is not None:
            while adjusted - previous > 180.0:
                offset -= 360.0
                adjusted = value + offset
            while adjusted - previous < -180.0:
                offset += 360.0
                adjusted = value + offset
        unwrapped.append(adjusted)
        previous = adjusted
    return unwrapped


def summarize_unwrapped_vector_sweep(recording: dict, vector_name: str) -> dict:
    arm = recording["arm"]
    samples = recording.get("samples", [])
    indexed_vectors = [
        (idx, vector)
        for idx, sample in enumerate(samples)
        if (vector := vector_from_sample(sample, arm, vector_name)) is not None
    ]
    if len(indexed_vectors) < 3:
        return {
            "vector": vector_name,
            "valid_samples": len(indexed_vectors),
            "total_samples": len(samples),
            "valid_ratio": len(indexed_vectors) / len(samples) if samples else None,
            "rom": None,
        }

    valid_indices = [idx for idx, _vector in indexed_vectors]
    vectors = [vector for _idx, vector in indexed_vectors]
    plane_normal = estimate_motion_plane_normal(vectors)
    if plane_normal is None:
        return {
            "vector": vector_name,
            "valid_samples": len(indexed_vectors),
            "total_samples": len(samples),
            "valid_ratio": len(indexed_vectors) / len(samples) if samples else None,
            "rom": None,
        }

    zero_axis = v_normalize(project_onto_plane(vectors[0], plane_normal))
    if zero_axis is None:
        return {
            "vector": vector_name,
            "valid_samples": len(indexed_vectors),
            "total_samples": len(samples),
            "valid_ratio": len(indexed_vectors) / len(samples) if samples else None,
            "rom": None,
        }
    positive_axis = v_normalize(v_cross(plane_normal, zero_axis))
    if positive_axis is None:
        return {
            "vector": vector_name,
            "valid_samples": len(indexed_vectors),
            "total_samples": len(samples),
            "valid_ratio": len(indexed_vectors) / len(samples) if samples else None,
            "rom": None,
        }

    signed_angles = [
        signed_plane_angle(
            vector,
            zero_axis=zero_axis,
            positive_axis=positive_axis,
            plane_normal=plane_normal,
        )
        for vector in vectors
    ]
    unwrapped = unwrap_degrees(signed_angles)
    numeric_unwrapped = [value for value in unwrapped if value is not None]
    if not numeric_unwrapped:
        return {
            "vector": vector_name,
            "valid_samples": len(indexed_vectors),
            "total_samples": len(samples),
            "valid_ratio": len(indexed_vectors) / len(samples) if samples else None,
            "rom": None,
        }

    if abs(min(numeric_unwrapped)) > abs(max(numeric_unwrapped)):
        unwrapped = [
            -value if value is not None else None
            for value in unwrapped
        ]
        numeric_unwrapped = [value for value in unwrapped if value is not None]
        positive_axis = v_scale(positive_axis, -1.0)

    min_value = min(numeric_unwrapped)
    max_value = max(numeric_unwrapped)
    min_valid_idx = numeric_unwrapped.index(min_value)
    max_valid_idx = numeric_unwrapped.index(max_value)
    min_sample_idx = valid_indices[min_valid_idx]
    max_sample_idx = valid_indices[max_valid_idx]
    started_at = recording.get("started_at", samples[0].get("time_monotonic", 0.0))

    trace = [
        {
            "sample_index": sample_idx,
            "time_offset": samples[sample_idx]["time_monotonic"] - started_at,
            "angle": angle,
        }
        for sample_idx, angle in zip(valid_indices, unwrapped)
        if angle is not None
    ]

    return {
        "vector": vector_name,
        "method": "sweep_plane_unwrapped",
        "plane_normal": list(plane_normal),
        "zero_axis": list(zero_axis),
        "positive_axis": list(positive_axis),
        "valid_samples": len(indexed_vectors),
        "total_samples": len(samples),
        "valid_ratio": len(indexed_vectors) / len(samples) if samples else None,
        "min_angle": min_value,
        "max_angle": max_value,
        "rom": max_value - min_value,
        "start_angle": numeric_unwrapped[0],
        "end_angle": numeric_unwrapped[-1],
        "min_time_offset": samples[min_sample_idx]["time_monotonic"] - started_at,
        "max_time_offset": samples[max_sample_idx]["time_monotonic"] - started_at,
        "min_sample_index": min_sample_idx,
        "max_sample_index": max_sample_idx,
        "trace": trace,
    }


def summarize_image_plane_angle_sweep(recording: dict, angle_name: str) -> dict:
    arm = recording["arm"]
    samples = recording.get("samples", [])
    indexed_values = [
        (idx, value)
        for idx, sample in enumerate(samples)
        if (value := image_plane_angle_from_sample(sample, arm, angle_name)) is not None
    ]
    if len(indexed_values) < 3:
        return {
            "angle": angle_name,
            "valid_samples": len(indexed_values),
            "total_samples": len(samples),
            "valid_ratio": len(indexed_values) / len(samples) if samples else None,
            "rom": None,
        }

    valid_indices = [idx for idx, _value in indexed_values]
    values = [value for _idx, value in indexed_values]
    min_value = min(values)
    max_value = max(values)
    min_valid_idx = values.index(min_value)
    max_valid_idx = values.index(max_value)
    min_sample_idx = valid_indices[min_valid_idx]
    max_sample_idx = valid_indices[max_valid_idx]
    started_at = recording.get("started_at", samples[0].get("time_monotonic", 0.0))

    trace = [
        {
            "sample_index": sample_idx,
            "time_offset": samples[sample_idx]["time_monotonic"] - started_at,
            "angle": value,
        }
        for sample_idx, value in indexed_values
    ]

    return {
        "angle": angle_name,
        "method": "image_plane_fixed_camera",
        "convention": "0_down_90_frame_left_180_up_gt180_past_vertical",
        "valid_samples": len(indexed_values),
        "total_samples": len(samples),
        "valid_ratio": len(indexed_values) / len(samples) if samples else None,
        "min_angle": min_value,
        "max_angle": max_value,
        "rom": max_value - min_value,
        "start_angle": values[0],
        "end_angle": values[-1],
        "min_time_offset": samples[min_sample_idx]["time_monotonic"] - started_at,
        "max_time_offset": samples[max_sample_idx]["time_monotonic"] - started_at,
        "min_sample_index": min_sample_idx,
        "max_sample_index": max_sample_idx,
        "trace": trace,
    }


def summarize_image_plane_rom_sweep(recording: dict) -> dict:
    return {
        "source": "pose_landmarks_image_plane",
        "goal": "fixed_camera_side_view_flexion_estimate",
        "convention": "0_down_90_frame_left_180_up_gt180_past_vertical",
        "humerus": summarize_image_plane_angle_sweep(recording, "humerus_flexion"),
        "reach": summarize_image_plane_angle_sweep(recording, "reach_flexion"),
    }


def summarize_advanced_rom_sweep(recording: dict) -> dict:
    return {
        "source": "rom_vectors_world_landmarks",
        "goal": "behind_head_flexion_rom",
        "humerus": summarize_unwrapped_vector_sweep(recording, "humerus"),
        "reach": summarize_unwrapped_vector_sweep(recording, "reach"),
    }


def frame_sample_by_index(rom: ROMSweepState, sample_index: int | None) -> dict | None:
    if sample_index is None:
        return None
    return next(
        (
            item
            for item in rom.frame_samples
            if item.get("sample_index") == sample_index
        ),
        None,
    )


def write_rom_frame(output_dir: Path, prefix: str, label: str, item: dict) -> Path:
    extension = item.get("extension")
    if item.get("image_bytes") is not None and extension:
        image_path = output_dir / f"{prefix}_{label}.{extension}"
        image_path.write_bytes(item["image_bytes"])
        return image_path

    image_path = output_dir / f"{prefix}_{label}.png"
    cv2.imwrite(str(image_path), item["frame"])
    return image_path


def save_rom_sweep(rom: ROMSweepState, output_dir: Path) -> None:
    step = rom.step()
    if step is None:
        return
    if not rom.samples:
        rom.recording = False
        rom.status = f"No samples captured for {step.name}. Press Space to retry."
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rom.capture_index += 1
    prefix = f"{rom.session_id}_rom_{rom.capture_index:02d}_{step.name}"
    recording = {
        "type": "rom_sweep",
        "session_id": rom.session_id,
        "capture_index": rom.capture_index,
        "pose": step.name,
        "arm": step.arm,
        "instruction": step.instruction,
        "view": rom.samples[-1].get("view"),
        "measurement_mode": (
            "calibrated"
            if rom.samples[-1].get("calibration_ready")
            else "uncalibrated"
        ),
        "calibration_ready_at_capture": bool(rom.samples[-1].get("calibration_ready")),
        "started_at": rom.started_at,
        "duration_target_seconds": None,
        "stop_mode": "manual_spacebar",
        "samples": rom.samples,
    }
    recording["rom_summary"] = summarize_rom_sweep(recording)
    recording["advanced_rom_summary"] = summarize_advanced_rom_sweep(recording)
    recording["image_plane_rom_summary"] = summarize_image_plane_rom_sweep(recording)

    image_paths = {}
    key_frame_metadata = {}
    selected_frames = dict(rom.key_frames)
    reach_summary = recording["advanced_rom_summary"].get("reach", {})
    humerus_summary = recording["advanced_rom_summary"].get("humerus", {})
    image_plane_summary = recording["image_plane_rom_summary"]
    started_at = recording["started_at"]
    for vector_name, summary in (("reach", reach_summary), ("humerus", humerus_summary)):
        for label, time_key in (("min", "min_time_offset"), ("max", "max_time_offset")):
            sample_index = summary.get(f"{label}_sample_index")
            selected = frame_sample_by_index(rom, sample_index)
            if selected is None:
                time_offset = summary.get(time_key)
                if time_offset is None:
                    continue
                target_time = started_at + time_offset
                selected = min(
                    rom.frame_samples,
                    key=lambda item: abs(item["sample"]["time_monotonic"] - target_time),
                ) if rom.frame_samples else None
            if selected is None:
                continue
            selected_frames[f"{vector_name}_{label}"] = {
                **selected,
                "source": f"advanced_{vector_name}_{label}",
            }
    for angle_name, summary in (
        ("image_reach", image_plane_summary.get("reach", {})),
        ("image_humerus", image_plane_summary.get("humerus", {})),
    ):
        for label, time_key in (("min", "min_time_offset"), ("max", "max_time_offset")):
            sample_index = summary.get(f"{label}_sample_index")
            selected = frame_sample_by_index(rom, sample_index)
            if selected is None:
                time_offset = summary.get(time_key)
                if time_offset is None:
                    continue
                target_time = started_at + time_offset
                selected = min(
                    rom.frame_samples,
                    key=lambda item: abs(item["sample"]["time_monotonic"] - target_time),
                ) if rom.frame_samples else None
            if selected is None:
                continue
            selected_frames[f"{angle_name}_{label}"] = {
                **selected,
                "source": f"image_plane_{angle_name}_{label}",
            }

    for label, item in selected_frames.items():
        image_path = write_rom_frame(output_dir, prefix, label, item)
        image_paths[label] = str(image_path)
        key_frame_metadata[label] = {
            "sample_index": item.get("sample_index"),
            "source": item.get("source"),
            "time_monotonic": item["sample"].get("time_monotonic"),
            "angles": item["sample"].get("angles"),
        }

    recording["image_paths"] = image_paths
    recording["key_frames"] = key_frame_metadata

    output_path = output_dir / f"{prefix}.json"
    output_path.write_text(json.dumps(recording, indent=2), encoding="utf-8")
    rom.last_saved_json_path = output_path
    rom.current_step += 1
    rom.recording = False
    rom.samples.clear()
    rom.key_frames.clear()
    rom.frame_samples.clear()
    rom.last_image_plane_angles = None
    next_step = rom.step()
    if next_step is None:
        rom.finish_if_done()
    else:
        rom.status = f"Saved {step.name}. Press Space for {next_step.name}."


def mark_test_capture(
    test_capture: TestCaptureState,
    history,
    now: float,
    frame,
    output_dir: Path,
) -> None:
    if not test_capture.active or test_capture.pending_capture is not None:
        return

    step = test_capture.step()
    if step is None:
        test_capture.finish_if_done()
        return

    pre_samples = [
        sample
        for sample in history
        if now - sample["time_monotonic"] <= test_capture.pre_seconds
    ]
    current_sample = pre_samples[-1] if pre_samples else {}
    calibration_ready = bool(current_sample.get("calibration_ready"))
    measurement_mode = "calibrated" if calibration_ready else "uncalibrated"
    view_mode = current_sample.get("view")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_index = test_capture.capture_index + 1
    image_path = output_dir / f"{test_capture.session_id}_{capture_index:02d}_{step.name}.png"
    cv2.imwrite(str(image_path), frame)

    test_capture.pending_capture = {
        "session_id": test_capture.session_id,
        "capture_index": capture_index,
        "pose": step.name,
        "instruction": step.instruction,
        "expected": step.expected,
        "view": view_mode,
        "measurement_mode": measurement_mode,
        "calibration_ready_at_capture": calibration_ready,
        "marked_at": now,
        "image_path": str(image_path),
        "pre_seconds": test_capture.pre_seconds,
        "post_seconds": test_capture.post_seconds,
        "pre_samples": pre_samples,
        "post_samples": [],
    }
    test_capture.countdown_started_at = None
    test_capture.unstable_frames = 0
    test_capture.stillness_buffer.clear()
    test_capture.post_until = now + test_capture.post_seconds
    test_capture.status = f"Recording {step.name} post-roll..."


def update_test_auto_capture(
    test_capture: TestCaptureState,
    result,
    visibility_threshold: float,
    now: float,
    history,
    frame,
    output_dir: Path,
) -> None:
    if not test_capture.active or test_capture.pending_capture is not None:
        return

    step = test_capture.step()
    if step is None:
        test_capture.finish_if_done()
        return

    samples = get_landmark_samples(result, visibility_threshold)
    if samples is None:
        test_capture.stillness_buffer.clear()
        test_capture.reset_countdown("Waiting for shoulders, elbows, and hips")
        return

    image_sample, _world_sample = samples
    test_capture.stillness_buffer.append(image_sample)
    if len(test_capture.stillness_buffer) < test_capture.stable_frames:
        test_capture.reset_countdown("Hold still")
        return

    recent = list(test_capture.stillness_buffer)[-test_capture.stable_frames:]
    stillness = sample_stillness(recent)
    if stillness is None or stillness > test_capture.stillness_threshold:
        test_capture.unstable_frames += 1
        test_capture.status = (
            f"Hold still: movement {stillness or 0.0:.3f}"
            f" / {test_capture.stillness_threshold:.3f}"
        )
        if test_capture.unstable_frames > test_capture.max_unstable_frames:
            test_capture.countdown_started_at = None
            test_capture.unstable_frames = 0
        return

    test_capture.unstable_frames = 0
    if test_capture.countdown_started_at is None:
        test_capture.countdown_started_at = now
        test_capture.status = "Countdown started"
        return

    remaining = test_capture.countdown_seconds - (now - test_capture.countdown_started_at)
    test_capture.status = f"Auto-capturing in {max(math.ceil(remaining), 0)}"
    if remaining <= 0:
        mark_test_capture(test_capture, history, now, frame, output_dir)
        test_capture.countdown_started_at = None


def update_test_capture(test_capture: TestCaptureState, sample: dict, output_dir: Path) -> None:
    if test_capture.pending_capture is None:
        return

    test_capture.pending_capture["post_samples"].append(sample)
    if sample["time_monotonic"] < test_capture.post_until:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    test_capture.capture_index += 1
    pose_name = test_capture.pending_capture["pose"]
    output_path = (
        output_dir
        / f"{test_capture.session_id}_{test_capture.capture_index:02d}_{pose_name}.json"
    )
    test_capture.pending_capture["angle_summary"] = summarize_capture(
        test_capture.pending_capture
    )
    output_path.write_text(
        json.dumps(test_capture.pending_capture, indent=2),
        encoding="utf-8",
    )

    test_capture.last_saved_json_path = output_path
    image_path = test_capture.pending_capture.get("image_path")
    test_capture.last_saved_image_path = Path(image_path) if image_path else None
    test_capture.current_step += 1
    test_capture.countdown_started_at = None
    test_capture.unstable_frames = 0
    test_capture.stillness_buffer.clear()
    test_capture.pending_capture = None
    next_step = test_capture.step()
    if next_step is None:
        test_capture.finish_if_done()
    else:
        test_capture.status = (
            f"Saved {pose_name}. Next: press Space for {next_step.name}."
        )


def generate_visualizations_on_quit(capture_dir: Path) -> None:
    json_files = list(capture_dir.glob("*.json"))
    if not json_files:
        return

    try:
        from visualize_captures import visualize

        generated = visualize(capture_dir)
        print(f"Generated {len(generated)} capture visualization(s) in {capture_dir}.")
    except Exception as exc:
        print(f"Could not generate capture visualizations: {exc}")


@dataclass
class WorkflowState:
    """Five-step patient workflow:
        1. Ready position           (Space -> 2)
        2. Baseline arm raise       (Space -> 3, only after a stable 3s hold)
        3. Grab the robot handle    (Space -> 4)
        4. Robot-assisted stretch   (Space -> 5, only after a stable 3s hold)
        5. Results / comparison     (Space -> 1, restart)

    Steps 2 and 4 watch the left-arm 2D flexion angle (camera is mirrored,
    so the patient's left arm reads on what looks like the right side of the
    screen), detect a >=3s stable hold (range within STABILITY_TOL_DEG), and
    capture the peak stable mean. The robot script still operates on the
    right-arm sequence — only the camera-side measurement was swapped.
    """
    HOLD_SECONDS: ClassVar[float] = 3.0
    STABILITY_TOL_DEG: ClassVar[float] = 8.0

    step: int = 1
    angle_buffer: deque = field(default_factory=lambda: deque(maxlen=300))
    peak_during_step: float | None = None
    last_stable_angle: float | None = None
    baseline_angle: float | None = None
    assisted_angle: float | None = None

    def reset_capture(self) -> None:
        self.angle_buffer.clear()
        self.peak_during_step = None
        self.last_stable_angle = None

    def update(self, now: float, angle: float | None) -> None:
        if self.step not in (2, 4):
            return
        if angle is None:
            self.angle_buffer.clear()
            self.last_stable_angle = None
            return

        cutoff = now - self.HOLD_SECONDS
        while self.angle_buffer and self.angle_buffer[0][0] < cutoff:
            self.angle_buffer.popleft()
        self.angle_buffer.append((now, angle))

        if not self.angle_buffer:
            self.last_stable_angle = None
            return

        values = [a for _, a in self.angle_buffer]
        if max(values) - min(values) > self.STABILITY_TOL_DEG:
            # Stability broken: reset the buffer to just this sample so the
            # countdown restarts from zero the instant the user becomes stable
            # again (rather than waiting for old unstable samples to age out).
            self.angle_buffer.clear()
            self.angle_buffer.append((now, angle))
            self.last_stable_angle = None
            return

        oldest_ts = self.angle_buffer[0][0]
        if now - oldest_ts < self.HOLD_SECONDS - 0.05:
            self.last_stable_angle = None
            return

        stable_value = sum(values) / len(values)
        self.last_stable_angle = stable_value
        if self.peak_during_step is None or stable_value > self.peak_during_step:
            self.peak_during_step = stable_value

    def hold_progress(self, now: float) -> float:
        """0..1 fraction of HOLD_SECONDS spent under the stability tolerance."""
        if self.step not in (2, 4) or not self.angle_buffer:
            return 0.0
        values = [a for _, a in self.angle_buffer]
        if max(values) - min(values) > self.STABILITY_TOL_DEG:
            return 0.0
        oldest_ts = self.angle_buffer[0][0]
        return min(1.0, (now - oldest_ts) / self.HOLD_SECONDS)

    def can_advance(self) -> bool:
        if self.step in (1, 3, 5):
            return True
        return self.peak_during_step is not None

    def advance(self, now: float) -> bool:
        if not self.can_advance():
            return False
        if self.step == 2:
            self.baseline_angle = self.peak_during_step
            self.reset_capture()
            self.step = 3
        elif self.step == 4:
            self.assisted_angle = self.peak_during_step
            self.reset_capture()
            self.step = 5
        elif self.step == 5:
            self.baseline_angle = None
            self.assisted_angle = None
            self.reset_capture()
            self.step = 1
        else:
            self.reset_capture()
            self.step += 1
        return True


WORKFLOW_STEP_TITLES = (
    "Sit in ready position",
    "Baseline arm raise",
    "Grab the handle",
    "Robot-assisted stretch",
    "Compare results",
)


def workflow_state_to_ui(
    workflow: WorkflowState,
    *,
    image_plane_angles: dict,
    robot_state,
    now: float,
    dev_overlay: str = "",
):
    """Build a UIState describing the current workflow step."""
    from pose_ui import (
        AngleReading, ChecklistItem, ComparisonRow, ROBOT_PHASE_LABELS, UIState, C,
    )

    state = UIState()
    state.dev_overlay = dev_overlay
    state.step_total = len(WORKFLOW_STEP_TITLES)
    state.step_index = workflow.step - 1

    # Header accent by step
    state.phase = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3}.get(workflow.step, 1)

    # Live angle readout: left-arm humerus 2D flexion (image plane).
    # Camera is mirrored, so the patient's left arm is the one they raise on
    # the visually-right side of the screen.
    l_2d = (image_plane_angles.get("L", {}) or {}).get("humerus_flexion")
    overlay_status = "live"
    if workflow.step in (2, 4) and workflow.last_stable_angle is not None:
        overlay_status = "hold"
    state.angles = [
        AngleReading("Flexion (2D)", "L", l_2d, target_min=120, status=overlay_status),
    ]

    # Checklist
    state.checklist = []
    for i, title in enumerate(WORKFLOW_STEP_TITLES, start=1):
        if i < workflow.step:
            status = "done"
        elif i == workflow.step:
            status = "active"
        else:
            status = "pending"
        state.checklist.append(ChecklistItem(title, status))

    # Helpers shared by step 2 and 4
    def _capture_footer_and_overlays(step_label: str, locked_label: str,
                                     next_label: str) -> None:
        progress = workflow.hold_progress(now)
        peak = workflow.peak_during_step
        if peak is not None:
            state.footer_hint = (
                f"{locked_label}: {peak:.0f}°   ·   "
                f"Space to {next_label} (hold longer to update peak)"
            )
            state.big_callout = f"PEAK: {peak:.0f}°"
            state.big_callout_color = C.SUCCESS
        elif progress >= 1.0:
            state.footer_hint = "Stable — capturing peak…"
            state.countdown_seconds = 0.0
            state.countdown_label = step_label
        elif progress > 0:
            remaining = workflow.HOLD_SECONDS * (1.0 - progress)
            state.footer_hint = (
                f"Hold steady — {remaining:0.1f}s left of the 3-second hold"
            )
            state.countdown_seconds = remaining
            state.countdown_label = step_label
        else:
            state.footer_hint = (
                "Raise your arm and hold steady for 3 seconds to capture"
            )

    if workflow.step == 1:
        state.hero_kicker = "Step 1 · Ready position"
        state.hero_title = "Sit down and face the camera"
        state.hero_bullets = [
            "Sit upright with both feet flat on the floor",
            "Rest your back on the chair",
            "Relax your arms loosely at your sides",
            "Face the camera so your full upper body is visible",
        ]
        state.footer_hint = "Press Space when you're ready to begin"

    elif workflow.step == 2:
        state.hero_kicker = "Step 2 · Baseline arm raise"
        state.hero_title = "Raise your left arm to your natural limit"
        state.hero_bullets = [
            "Raise your left arm in a relaxed motion",
            "Once past horizontal, hold at your natural limit for 3 seconds",
            "We capture your highest stable angle automatically",
            "Press Space to lock in your baseline and continue",
        ]
        state.show_arm_raise_animation = True
        # Below 90° we suppress the timer entirely (the buffer is also cleared
        # in the main loop), so the only way to start the 3s countdown is to
        # raise the arm past horizontal.
        if l_2d is None or l_2d <= 90.0:
            if workflow.peak_during_step is not None:
                # Captured already; show locked peak even if the user dropped.
                state.footer_hint = (
                    f"Baseline: {workflow.peak_during_step:.0f}°   ·   "
                    f"Space to continue"
                )
                state.big_callout = f"PEAK: {workflow.peak_during_step:.0f}°"
                state.big_callout_color = C.SUCCESS
            else:
                state.footer_hint = (
                    "Raise your arm above shoulder height to start the timer"
                )
        else:
            _capture_footer_and_overlays(
                step_label="Hold steady",
                locked_label="Baseline",
                next_label="continue",
            )

    elif workflow.step == 3:
        state.hero_kicker = "Step 3 · Grab the handle"
        state.hero_title = "Reach forward and grab the robot handle"
        state.hero_bullets = [
            "Stand or sit within reach of the robot arm",
            "Wrap your hand firmly around the handle",
            "Keep your grip relaxed but secure",
            "Press Space when you're ready for the assisted stretch",
        ]
        state.footer_hint = "Press Space when ready"

    elif workflow.step == 4:
        state.hero_kicker = "Step 4 · Robot-assisted stretch"
        state.hero_title = "Let the robot guide your left arm up"
        state.hero_bullets = [
            "Stay relaxed and follow the robot's motion",
            "We start capturing when the robot reaches the end of its sweep",
            "Hold at the stretched position for 3 seconds",
            "Press Space to lock in the assisted measurement",
        ]
        rp = getattr(robot_state, "phase", "") if robot_state is not None else ""
        connected = (robot_state is not None and getattr(robot_state, "connected", False))
        at_end = (rp == "at_end")
        if connected:
            cap_name = getattr(robot_state, "capture_name", None) or "(no capture)"
            state.robot_state = (
                f"{cap_name}  ·  {ROBOT_PHASE_LABELS.get(rp, rp) or 'idle'}"
            )
        if not at_end:
            # Robot is still moving toward the stretched position. Capture
            # buffer is cleared by the main loop while this is the case.
            if rp == "moving_to_start":
                state.footer_hint = "Robot is moving to start — stay relaxed"
            elif rp == "at_start":
                state.footer_hint = "Robot is ready — about to begin the stretch"
            elif rp == "executing":
                state.footer_hint = "Robot is stretching — follow the motion"
            elif rp == "aborted":
                state.footer_hint = "Robot aborted — see status panel"
            else:
                state.footer_hint = "Waiting for robot to reach the end position…"
        else:
            _capture_footer_and_overlays(
                step_label="Hold at limit",
                locked_label="Assisted",
                next_label="see results",
            )

    elif workflow.step == 5:
        baseline = workflow.baseline_angle
        assisted = workflow.assisted_angle
        delta = (assisted - baseline) if (baseline is not None and assisted is not None) else None
        state.hero_kicker = "Step 5 · Results"
        if delta is None:
            state.hero_title = "Results"
        elif delta > 0:
            state.hero_title = f"+{delta:.0f}° improvement with assistance"
        else:
            state.hero_title = f"{delta:.0f}° change with assistance"
        state.hero_subtitle = ""
        state.hero_bullets = None
        state.comparison = [
            ComparisonRow("Left arm 2D flexion",
                          before=baseline, after=assisted),
        ]
        state.comparison_caption = (
            "Before = your own peak raise. After = peak raise with robot assist."
        )
        state.footer_hint = "Press Space to restart   ·   q to quit"
        # Take over the whole canvas with a presenter-friendly results layout.
        state.show_fullscreen_results = True

    return state


def main() -> None:
    args = parse_args()
    ensure_model(args.model)
    if args.download_model_only:
        print(f"Model ready: {args.model}")
        return

    robot_client = None
    if not args.no_robot:
        robot_client = StatusClient(host=args.robot_host, port=args.robot_port)
        robot_client.start()
        print(f"[bridge] Robot status client connecting to "
              f"{args.robot_host}:{args.robot_port} (--no-robot to disable)")

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    latest_result = {"value": None, "timestamp_ms": 0}
    result_lock = threading.Lock()

    def on_result(result, _output_image, timestamp_ms: int) -> None:
        with result_lock:
            latest_result["value"] = result
            latest_result["timestamp_ms"] = timestamp_ms

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=args.min_confidence,
        min_pose_presence_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. On macOS, check Camera privacy permissions."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Detect the display so we can render the UI canvas at native screen
    # resolution; the fullscreen window then displays 1:1 without scaling.
    screen_w, screen_h = _detect_screen_size()
    render_w, render_h = screen_w, screen_h
    print(f">>> rendering UI at {render_w}x{render_h}")

    cv2.namedWindow("Shrimpy Pose MVP", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Shrimpy Pose MVP", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    last_timestamp_ms = 0
    previous_frame_time = time.perf_counter()
    fps = 0.0
    calibration = CalibrationState()
    calibration.stillness_threshold = args.calibration_stillness
    calibration.stable_frames = max(args.calibration_stable_frames, 2)
    calibration.landmark_buffer = deque(maxlen=max(calibration.stable_frames * 3, 30))
    test_capture = TestCaptureState(
        pre_seconds=args.test_window_seconds,
        post_seconds=args.test_window_seconds,
        countdown_seconds=args.test_countdown_seconds,
        stable_frames=max(args.test_stable_frames, 2),
        stillness_threshold=args.test_stillness,
    )
    test_capture.stillness_buffer = deque(maxlen=max(test_capture.stable_frames * 3, 30))
    rom_sweep = ROMSweepState(duration_seconds=args.rom_sweep_seconds)
    diagnostic_history = deque(maxlen=max(int(args.test_window_seconds * 90), 180))
    smoothed_measurement_frame = None
    missing_measurement_frames = 0
    workflow = WorkflowState()

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            now = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("No frame from camera; exiting.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.monotonic() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            landmarker.detect_async(mp_image, timestamp_ms)

            with result_lock:
                result = latest_result["value"]
                result_timestamp_ms = latest_result["timestamp_ms"]

            update_calibration(calibration, result, args.min_confidence, now)
            pose_count = len(result.pose_landmarks) if result and result.pose_landmarks else 0
            measurement_frame = measurement_frame_from_result(
                result,
                args.min_confidence,
                calibration.axes,
            )
            if measurement_frame is None:
                missing_measurement_frames += 1
                if missing_measurement_frames > 30:
                    smoothed_measurement_frame = None
                else:
                    smoothed_measurement_frame = smooth_measurement_frame(
                        measurement_frame,
                        smoothed_measurement_frame,
                        args.angle_smoothing,
                    )
            else:
                missing_measurement_frames = 0
                smoothed_measurement_frame = smooth_measurement_frame(
                    measurement_frame,
                    smoothed_measurement_frame,
                    args.angle_smoothing,
            )
            angles = angles_from_measurement_frame(smoothed_measurement_frame)
            active_test_step = test_capture.step()
            landmark_samples = get_landmark_samples(result, args.min_confidence)
            image_sample = landmark_samples[0] if landmark_samples else None
            image_plane_angles = all_image_plane_angles_from_result(result)
            diagnostic_sample = make_diagnostic_sample(
                now=now,
                result_timestamp_ms=result_timestamp_ms,
                result=result,
                angles=angles,
                raw_measurement_frame=measurement_frame,
                smoothed_measurement_frame=smoothed_measurement_frame,
                pose_count=pose_count,
                view_mode=args.view,
                calibration=calibration,
                test_step_name=active_test_step.name if active_test_step else None,
                image_sample=image_sample,
                image_plane_angles=image_plane_angles,
            )
            if rom_sweep.recording:
                rom_sweep.last_image_plane_angles = image_plane_angles
                diagnostic_sample["rom_vectors"] = {
                    "L": rom_vectors_from_result(result, "L"),
                    "R": rom_vectors_from_result(result, "R"),
                }
            diagnostic_history.append(diagnostic_sample)
            if rom_sweep.recording:
                rom_sweep.samples.append(diagnostic_sample)
                update_rom_key_frames(rom_sweep, diagnostic_sample, frame)
            update_test_auto_capture(
                test_capture,
                result,
                args.min_confidence,
                now,
                diagnostic_history,
                frame,
                args.capture_dir,
            )
            update_test_capture(test_capture, diagnostic_sample, args.capture_dir)

            instantaneous_fps = 1.0 / max(now - previous_frame_time, 1e-6)
            fps = instantaneous_fps if fps == 0.0 else (fps * 0.9) + (instantaneous_fps * 0.1)
            previous_frame_time = now

            robot_snapshot = robot_client.snapshot() if robot_client else None

            # Drive the patient workflow off the left-arm 2D flexion angle
            # (camera is mirrored, so the patient's left arm is the one
            # visually raised on the right side of the screen).
            # Step 2's capture only starts once the arm crosses horizontal
            # (>90°). Step 4's capture is gated on the robot being at_end.
            # Outside those, always track.
            l_2d_flex = (image_plane_angles.get("L", {}) or {}).get("humerus_flexion")
            if workflow.step == 2:
                above_horizontal = (
                    l_2d_flex is not None and l_2d_flex > 90.0
                )
                if above_horizontal:
                    workflow.update(now, l_2d_flex)
                else:
                    # Arm not raised past horizontal yet; suppress the timer.
                    workflow.angle_buffer.clear()
                    workflow.last_stable_angle = None
            elif workflow.step == 4:
                at_end = (
                    robot_snapshot is not None
                    and getattr(robot_snapshot, "phase", "") == "at_end"
                )
                if at_end:
                    workflow.update(now, l_2d_flex)
                else:
                    # Robot hasn't reached the stretched position yet; don't
                    # accumulate samples or capture a peak.
                    workflow.angle_buffer.clear()
                    workflow.last_stable_angle = None
            else:
                workflow.update(now, l_2d_flex)
            dev_overlay = (
                f"dev   ·   view={args.view}   ·   step={workflow.step}/5   ·   "
                f"poses={pose_count}   ·   fps={fps:4.1f}   ·   "
                f"ts={result_timestamp_ms}ms"
            )

            # Workflow drives the UI unless a diagnostic mode is active.
            if calibration.active or test_capture.active or rom_sweep.active:
                ui_state = None
            else:
                ui_state = workflow_state_to_ui(
                    workflow,
                    image_plane_angles=image_plane_angles,
                    robot_state=robot_snapshot,
                    now=now,
                    dev_overlay=dev_overlay,
                )

            canvas = render_frame(
                frame,
                result=result,
                visibility_threshold=args.min_confidence,
                angles=angles,
                image_plane_angles=image_plane_angles,
                calibration=calibration,
                test_capture=test_capture,
                rom=rom_sweep,
                view_mode=args.view,
                pose_count=pose_count,
                fps=fps,
                result_timestamp_ms=result_timestamp_ms,
                now=now,
                robot_state=robot_snapshot,
                ui_state=ui_state,
                render_size=(render_w, render_h),
            )

            cv2.imshow("Shrimpy Pose MVP", canvas)
            raw_key = cv2.waitKeyEx(1)
            key = raw_key & 0xFF
            if raw_key in LEFT_ARROW_KEYS and rom_sweep.active and not rom_sweep.recording:
                rom_sweep.select_arm("L")
            elif raw_key in RIGHT_ARROW_KEYS and rom_sweep.active and not rom_sweep.recording:
                rom_sweep.select_arm("R")
            elif key == ord("c"):
                calibration.start()
                smoothed_measurement_frame = None
                missing_measurement_frames = 0
            elif key == ord("t"):
                rom_sweep.active = False
                rom_sweep.recording = False
                test_capture.start()
            elif key == ord("r"):
                test_capture.active = False
                test_capture.pending_capture = None
                rom_sweep.start(args.view)
                if args.rom_arm is not None:
                    rom_sweep.select_arm(args.rom_arm)
            elif key == ord(" "):
                if calibration.active:
                    capture_calibration_now(calibration, result, args.min_confidence)
                    smoothed_measurement_frame = None
                    missing_measurement_frames = 0
                elif test_capture.active:
                    mark_test_capture(
                        test_capture,
                        diagnostic_history,
                        now,
                        frame,
                        args.capture_dir,
                    )
                    test_capture.countdown_started_at = None
                    test_capture.unstable_frames = 0
                elif rom_sweep.active:
                    if rom_sweep.recording:
                        save_rom_sweep(rom_sweep, args.capture_dir)
                    else:
                        rom_sweep.begin_recording(now)
                else:
                    prev_step = workflow.step
                    if workflow.advance(now) and robot_client is not None:
                        transition = (prev_step, workflow.step)
                        # Each workflow transition that requires a robot motion
                        # sends the corresponding command. piper_sequence_demo
                        # is parked at wait_for_command(expected) for each.
                        command = {
                            (2, 3): "start_session",
                            (3, 4): "begin_workout",
                            (4, 5): "end_session",
                        }.get(transition)
                        if command is not None:
                            sent = robot_client.send_command(command)
                            print(f"[workflow] {prev_step}->{workflow.step}: "
                                  f"sent '{command}' (delivered={sent})")
            elif key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()
    if not args.no_visualize_on_quit:
        generate_visualizations_on_quit(args.capture_dir)


if __name__ == "__main__":
    main()
