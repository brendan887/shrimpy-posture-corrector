from __future__ import annotations

import argparse
import json
import math
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, pstdev

import cv2
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_MODEL_PATH = Path("models/pose_landmarker_full.task")

POSE_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
    (27, 31),
    (28, 32),
)

LEFT_ARM = {"name": "L", "shoulder": 11, "elbow": 13}
RIGHT_ARM = {"name": "R", "shoulder": 12, "elbow": 14}
CALIBRATION_LANDMARKS = (11, 12, 13, 14, 23, 24)
VIEW_MODES = ("front", "left-45", "right-45")


VIEW_GUIDANCE = {
    "front": {
        "title": "Front-view mode",
        "instruction": "Camera centered in front. Best for abduction; flexion is more depth-sensitive.",
    },
    "left-45": {
        "title": "Left 45-degree mode",
        "instruction": "Place camera at your left-front 45 deg angle. Good compromise for flexion.",
    },
    "right-45": {
        "title": "Right 45-degree mode",
        "instruction": "Place camera at your right-front 45 deg angle. Good compromise for flexion.",
    },
}


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
    return parser.parse_args()


def is_visible(landmark, threshold: float) -> bool:
    visibility = getattr(landmark, "visibility", 1.0)
    presence = getattr(landmark, "presence", 1.0)
    return visibility >= threshold and presence >= threshold


def landmark_point(landmark, width: int, height: int) -> tuple[int, int]:
    x = min(max(landmark.x, 0.0), 1.0)
    y = min(max(landmark.y, 0.0), 1.0)
    return int(x * width), int(y * height)


def draw_pose(frame, result, visibility_threshold: float) -> int:
    if not result or not result.pose_landmarks:
        return 0

    height, width = frame.shape[:2]
    for pose_landmarks in result.pose_landmarks:
        points = [
            landmark_point(landmark, width, height)
            if is_visible(landmark, visibility_threshold)
            else None
            for landmark in pose_landmarks
        ]

        for start_idx, end_idx in POSE_CONNECTIONS:
            start = points[start_idx]
            end = points[end_idx]
            if start and end:
                cv2.line(frame, start, end, (80, 220, 255), 3, cv2.LINE_AA)

        for point in points:
            if point:
                cv2.circle(frame, point, 5, (40, 255, 120), -1, cv2.LINE_AA)
                cv2.circle(frame, point, 7, (20, 30, 20), 1, cv2.LINE_AA)

    return len(result.pose_landmarks)


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

        angles[arm_name] = {
            "flexion": signed_plane_angle(
                upper_arm,
                zero_axis=frame["down"],
                positive_axis=frame["forward"],
                plane_normal=frame["right"],
            ),
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


def format_angle(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:5.1f}"


def draw_angle_panel(frame, angles) -> None:
    panel_width, panel_height = 420, 112
    x, y = 24, 92 + int(panel_height * 0.25)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - 28),
        (x + panel_width, y + panel_height),
        (10, 24, 32),
        -1,
    )
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    cv2.putText(
        frame,
        "Shoulder angle relative to torso",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 245, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "0 down | 90 straight out | 180 overhead",
        (x, y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (190, 210, 220),
        1,
        cv2.LINE_AA,
    )

    for row, arm_name in enumerate(("L", "R")):
        arm_angles = angles.get(arm_name, {})
        text = (
            f"{arm_name}  flex {format_angle(arm_angles.get('flexion'))} deg"
            f"   abd {format_angle(arm_angles.get('abduction'))} deg"
        )
        cv2.putText(
            frame,
            text,
            (x, y + 62 + (row * 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (80, 220, 255) if arm_name == "L" else (40, 255, 120),
            2,
            cv2.LINE_AA,
        )


def draw_calibration_panel(frame, calibration: CalibrationState, view_mode: str) -> None:
    height, width = frame.shape[:2]
    panel_width, panel_height = 560, 138
    x = max(width - panel_width - 24, 24)
    y = 24

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - 12),
        (x + panel_width, y + panel_height),
        (26, 20, 12),
        -1,
    )
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    if calibration.active:
        step = calibration.step()
        title = f"Calibration {calibration.current_step + 1}/{len(calibration.steps)}"
        instruction = step.instruction if step else "Finishing calibration..."
        status = calibration.status
        color = (80, 220, 255)
    elif calibration.axes is not None:
        title = "Calibration ready"
        instruction = "Using captured neutral, forward, and side axes."
        status = "Press c to recalibrate"
        color = (40, 255, 120)
    else:
        guidance = VIEW_GUIDANCE[view_mode]
        title = guidance["title"]
        instruction = guidance["instruction"]
        status = (
            calibration.status
            if calibration.status != "Press c to calibrate"
            else "Calibration optional: press c to calibrate if readings drift"
        )
        color = (220, 245, 255)

    lines = (
        (title, 0.68, color, 2),
        (instruction, 0.52, (240, 240, 230), 1),
        (status, 0.58, (255, 230, 160), 2),
        ("Keys: c calibrate | t test | Space capture | q/Esc quit", 0.48, (210, 210, 200), 1),
    )
    for idx, (text, scale, text_color, thickness) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x, y + 18 + (idx * 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )


def draw_test_capture_panel(frame, test_capture: TestCaptureState) -> None:
    if not test_capture.active and test_capture.last_saved_json_path is None:
        return

    height, width = frame.shape[:2]
    panel_width, panel_height = 620, 128
    x = max(width - panel_width - 24, 24)
    y = max(height - panel_height - 24, 24)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - 12),
        (x + panel_width, y + panel_height),
        (12, 18, 34),
        -1,
    )
    cv2.addWeighted(overlay, 0.64, frame, 0.36, 0, frame)

    step = test_capture.step()
    if test_capture.pending_capture is not None:
        title = "Test capture recording"
        instruction = test_capture.pending_capture["instruction"]
        expected = test_capture.pending_capture["expected"]
        status = test_capture.status
        color = (80, 220, 255)
    elif step is not None:
        title = f"Test {test_capture.current_step + 1}/{len(test_capture.steps)}: {step.name}"
        instruction = step.instruction
        expected = step.expected
        status = f"{test_capture.status} | Space captures now"
        color = (80, 220, 255)
    else:
        title = "Test capture complete"
        instruction = "Press t to repeat the diagnostic sequence."
        expected = (
            f"Last saved: {test_capture.last_saved_json_path}"
            if test_capture.last_saved_json_path
            else "No capture saved yet."
        )
        status = test_capture.status
        color = (40, 255, 120)

    lines = (
        (title, 0.66, color, 2),
        (instruction, 0.54, (240, 240, 250), 1),
        (expected, 0.48, (205, 220, 255), 1),
        (status, 0.54, (255, 230, 160), 2),
    )
    for idx, (text, scale, text_color, thickness) in enumerate(lines):
        cv2.putText(
            frame,
            text[:95],
            (x, y + 18 + (idx * 29)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    ensure_model(args.model)
    if args.download_model_only:
        print(f"Model ready: {args.model}")
        return

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
    diagnostic_history = deque(maxlen=max(int(args.test_window_seconds * 90), 180))
    smoothed_measurement_frame = None
    missing_measurement_frames = 0

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
            pose_count = draw_pose(frame, result, args.min_confidence)
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
            )
            diagnostic_history.append(diagnostic_sample)
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

            cv2.putText(
                frame,
                f"MediaPipe Pose Full | view: {args.view} | poses: {pose_count} | fps: {fps:4.1f}",
                (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"result ts: {result_timestamp_ms} ms | press q or Esc to quit",
                (24, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            draw_angle_panel(frame, angles)
            draw_calibration_panel(frame, calibration, args.view)
            draw_test_capture_panel(frame, test_capture)

            cv2.imshow("Shrimpy Pose MVP", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                calibration.start()
                smoothed_measurement_frame = None
                missing_measurement_frames = 0
            elif key == ord("t"):
                test_capture.start()
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
            elif key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
