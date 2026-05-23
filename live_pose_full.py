from __future__ import annotations

import argparse
import math
import threading
import time
import urllib.request
from pathlib import Path

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
        help="EMA factor for angle readouts. 0 is very smooth, 1 is no smoothing.",
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


def project_onto_plane(v, normal):
    return v_sub(v, v_scale(normal, v_dot(v, normal)))


def signed_plane_angle(vector, zero_axis, positive_axis, plane_normal) -> float | None:
    projected = v_normalize(project_onto_plane(vector, plane_normal))
    if projected is None:
        return None

    degrees = math.degrees(
        math.atan2(v_dot(projected, positive_axis), v_dot(projected, zero_axis))
    )
    return degrees


def arm_landmarks_visible(image_landmarks, arm, threshold: float) -> bool:
    needed = (arm["shoulder"], arm["elbow"], 11, 12, 23, 24)
    return all(is_visible(image_landmarks[idx], threshold) for idx in needed)


def calculate_arm_angles(result, visibility_threshold: float) -> dict[str, dict[str, float | None]]:
    if (
        not result
        or not result.pose_landmarks
        or not result.pose_world_landmarks
    ):
        return {}

    image_landmarks = result.pose_landmarks[0]
    world_landmarks = result.pose_world_landmarks[0]

    left_shoulder = vec_from_landmark(world_landmarks[11])
    right_shoulder = vec_from_landmark(world_landmarks[12])
    left_hip = vec_from_landmark(world_landmarks[23])
    right_hip = vec_from_landmark(world_landmarks[24])
    shoulder_mid = v_scale(v_add(left_shoulder, right_shoulder), 0.5)
    hip_mid = v_scale(v_add(left_hip, right_hip), 0.5)

    up_axis = v_normalize(v_sub(shoulder_mid, hip_mid))
    right_axis = v_normalize(v_sub(right_shoulder, left_shoulder))
    if up_axis is None or right_axis is None:
        return {}

    forward_axis = v_normalize(v_cross(right_axis, up_axis))
    if forward_axis is None:
        return {}

    # MediaPipe webcam depth is negative toward the camera for a front-facing user.
    # This sign choice makes forward arm flexion report as positive.
    if forward_axis[2] > 0:
        forward_axis = v_scale(forward_axis, -1.0)

    down_axis = v_scale(up_axis, -1.0)
    angles = {}
    for arm in (LEFT_ARM, RIGHT_ARM):
        if not arm_landmarks_visible(image_landmarks, arm, visibility_threshold):
            angles[arm["name"]] = {"flexion": None, "abduction": None}
            continue

        shoulder = vec_from_landmark(world_landmarks[arm["shoulder"]])
        elbow = vec_from_landmark(world_landmarks[arm["elbow"]])
        upper_arm = v_normalize(v_sub(elbow, shoulder))
        if upper_arm is None:
            angles[arm["name"]] = {"flexion": None, "abduction": None}
            continue

        side_axis = (
            v_normalize(v_sub(left_shoulder, right_shoulder))
            if arm["name"] == "L"
            else right_axis
        )
        if side_axis is None:
            angles[arm["name"]] = {"flexion": None, "abduction": None}
            continue

        angles[arm["name"]] = {
            "flexion": signed_plane_angle(
                upper_arm,
                zero_axis=down_axis,
                positive_axis=forward_axis,
                plane_normal=right_axis,
            ),
            "abduction": signed_plane_angle(
                upper_arm,
                zero_axis=down_axis,
                positive_axis=side_axis,
                plane_normal=forward_axis,
            ),
        }

    return angles


def smooth_angles(current, previous, smoothing_factor: float):
    alpha = min(max(smoothing_factor, 0.0), 1.0)
    smoothed = {}

    for arm_name in ("L", "R"):
        smoothed[arm_name] = {}
        for angle_name in ("flexion", "abduction"):
            current_value = current.get(arm_name, {}).get(angle_name)
            previous_value = previous.get(arm_name, {}).get(angle_name)

            if current_value is None:
                smoothed[arm_name][angle_name] = previous_value
            elif previous_value is None:
                smoothed[arm_name][angle_name] = current_value
            else:
                smoothed[arm_name][angle_name] = (
                    (alpha * current_value) + ((1.0 - alpha) * previous_value)
                )

    return smoothed


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
    smoothed_angles = {}

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
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

            pose_count = draw_pose(frame, result, args.min_confidence)
            angles = calculate_arm_angles(result, args.min_confidence)
            smoothed_angles = smooth_angles(
                angles,
                smoothed_angles,
                args.angle_smoothing,
            )

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_frame_time, 1e-6)
            fps = instantaneous_fps if fps == 0.0 else (fps * 0.9) + (instantaneous_fps * 0.1)
            previous_frame_time = now

            cv2.putText(
                frame,
                f"MediaPipe Pose Full | poses: {pose_count} | fps: {fps:4.1f}",
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
            draw_angle_panel(frame, smoothed_angles)

            cv2.imshow("Shrimpy Pose MVP", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
