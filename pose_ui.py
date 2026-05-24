from __future__ import annotations

import cv2


VIEW_MODES = ("front", "left-45", "right-45", "left-side", "right-side")

VIEW_GUIDANCE = {
    "front": {
        "title": "Front-view mode",
        "instruction": "Camera centered in front. Best for abduction; flexion is more depth-sensitive.",
    },
    "left-45": {
        "title": "Left 45-degree mode",
        "instruction": "Place camera at your left-front 45 deg angle. ROM sweep uses the camera-side R landmark arm.",
    },
    "right-45": {
        "title": "Right 45-degree mode",
        "instruction": "Place camera at your right-front 45 deg angle. ROM sweep uses the camera-side L landmark arm.",
    },
    "left-side": {
        "title": "Left side-view mode",
        "instruction": "Place camera near your left side. ROM sweep uses the camera-side R landmark arm.",
    },
    "right-side": {
        "title": "Right side-view mode",
        "instruction": "Place camera near your right side. ROM sweep uses the camera-side L landmark arm.",
    },
}

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


def is_visible(landmark, threshold: float) -> bool:
    visibility = getattr(landmark, "visibility", 1.0)
    presence = getattr(landmark, "presence", 1.0)
    return visibility >= threshold and presence >= threshold


def landmark_point(landmark, width: int, height: int) -> tuple[int, int]:
    x = min(max(landmark.x, 0.0), 1.0)
    y = min(max(landmark.y, 0.0), 1.0)
    return int(x * width), int(y * height)


def draw_pose(frame, result, visibility_threshold: float) -> None:
    if not result or not result.pose_landmarks:
        return

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


def format_angle(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:5.1f}"


def draw_angle_panel(frame, angles, image_plane_angles=None) -> None:
    image_plane_angles = image_plane_angles or {}
    panel_width, panel_height = 560, 172
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
        image_angles = image_plane_angles.get(arm_name, {})
        row_y = y + 62 + (row * 54)
        text = (
            f"{arm_name}  flex {format_angle(arm_angles.get('flexion'))} deg"
            f"   abd {format_angle(arm_angles.get('abduction'))} deg"
        )
        cv2.putText(
            frame,
            text,
            (x, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (80, 220, 255) if arm_name == "L" else (40, 255, 120),
            2,
            cv2.LINE_AA,
        )
        image_text = (
            f"   2D hum {format_angle(image_angles.get('humerus_flexion'))} deg"
            f"   reach {format_angle(image_angles.get('reach_flexion'))} deg"
        )
        cv2.putText(
            frame,
            image_text,
            (x, row_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (110, 235, 255),
            2,
            cv2.LINE_AA,
        )


def draw_calibration_panel(frame, calibration, view_mode: str) -> None:
    _height, width = frame.shape[:2]
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
        ("Keys: c calibrate | t static | r ROM | Space action | q/Esc quit", 0.48, (210, 210, 200), 1),
    )
    draw_lines(frame, lines, x, y)


def draw_test_capture_panel(frame, test_capture) -> None:
    if not test_capture.active and test_capture.last_saved_json_path is None:
        return

    height, width = frame.shape[:2]
    panel_width, panel_height = 700, 156
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
    draw_lines(frame, lines, x, y, line_height=29, truncate=95)


def draw_rom_panel(frame, rom, now: float) -> None:
    if not rom.active and rom.last_saved_json_path is None:
        return

    height, _width = frame.shape[:2]
    panel_width, panel_height = 620, 128
    x = 24
    y = max(height - panel_height - 24, 24)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - 12),
        (x + panel_width, y + panel_height),
        (18, 12, 34),
        -1,
    )
    cv2.addWeighted(overlay, 0.64, frame, 0.36, 0, frame)

    step = rom.step()
    if rom.recording and step is not None:
        elapsed = max(now - rom.started_at, 0.0)
        image_plane = (rom.last_image_plane_angles or {}).get(step.arm, {})
        image_plane_text = (
            f"2D frame flex: hum {format_angle(image_plane.get('humerus_flexion'))} deg"
            f" | reach {format_angle(image_plane.get('reach_flexion'))} deg"
        )
        title = f"ROM sweep recording: {step.arm} arm"
        instruction = step.instruction
        expected = "Move slowly into max flexion. Hold briefly if useful; Space stops and saves."
        status = f"Recording {elapsed:4.1f}s | Press Space to stop"
        color = (190, 120, 255)
    elif step is not None:
        title = f"ROM sweep {rom.current_step + 1}/{len(rom.steps)}: {step.arm} arm"
        instruction = step.instruction
        expected = "Press Space to start; press Space again when the sweep is complete."
        status = rom.status
        color = (190, 120, 255)
    else:
        title = "ROM sweep complete"
        instruction = "Press r to repeat flexion ROM sweeps for the current view."
        expected = (
            f"Last saved: {rom.last_saved_json_path}"
            if rom.last_saved_json_path
            else "No ROM sweep saved yet."
        )
        status = rom.status
        color = (40, 255, 120)

    lines = (
        (title, 0.66, color, 2),
        (instruction, 0.54, (245, 240, 255), 1),
        (expected, 0.48, (220, 205, 255), 1),
        (image_plane_text if rom.recording and step is not None else "", 0.52, (110, 235, 255), 2),
        (status, 0.54, (255, 230, 160), 2),
    )
    draw_lines(frame, lines, x, y, line_height=28, truncate=105)


def draw_lines(frame, lines, x: int, y: int, line_height: int = 30, truncate: int | None = None) -> None:
    for idx, (text, scale, text_color, thickness) in enumerate(lines):
        if truncate is not None:
            text = text[:truncate]
        cv2.putText(
            frame,
            text,
            (x, y + 18 + (idx * line_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )


def draw_status(frame, view_mode: str, pose_count: int, fps: float, result_timestamp_ms: int) -> None:
    cv2.putText(
        frame,
        f"MediaPipe Pose Full | view: {view_mode} | poses: {pose_count} | fps: {fps:4.1f}",
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


def render_frame(
    frame,
    *,
    result,
    visibility_threshold: float,
    angles,
    image_plane_angles,
    calibration,
    test_capture,
    rom,
    view_mode: str,
    pose_count: int,
    fps: float,
    result_timestamp_ms: int,
    now: float,
) -> None:
    draw_pose(frame, result, visibility_threshold)
    draw_status(frame, view_mode, pose_count, fps, result_timestamp_ms)
    draw_angle_panel(frame, angles, image_plane_angles)
    draw_calibration_panel(frame, calibration, view_mode)
    draw_test_capture_panel(frame, test_capture)
    draw_rom_panel(frame, rom, now)
