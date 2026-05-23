from __future__ import annotations

import argparse
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

            cv2.imshow("Shrimpy Pose MVP", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
