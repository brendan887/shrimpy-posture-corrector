from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from live_pose_full import ROM_LANDMARKS, summarize_capture, summarize_image_plane_rom_sweep


ANGLE_KEYS = (
    ("L", "flexion", "L flex"),
    ("L", "abduction", "L abd"),
    ("R", "flexion", "R flex"),
    ("R", "abduction", "R abd"),
)
POSE_ORDER = ("arms_down", "arms_forward", "arms_side_t", "arms_overhead")


def dict_landmark_confidence(landmark: dict | None) -> float | None:
    if landmark is None:
        return None
    return min(landmark.get("visibility", 1.0), landmark.get("presence", 1.0))


def image_plane_flexion_from_dict(shoulder: dict | None, distal: dict | None) -> float | None:
    if shoulder is None or distal is None:
        return None
    dx = distal["x"] - shoulder["x"]
    dy = distal["y"] - shoulder["y"]
    if np.hypot(dx, dy) < 1e-6:
        return None
    angle = np.degrees(np.arctan2(-dx, dy))
    if angle < -90.0:
        angle += 360.0
    return float(angle)


def midpoint_landmark(a: dict | None, b: dict | None) -> dict | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return {
        "x": (a["x"] + b["x"]) * 0.5,
        "y": (a["y"] + b["y"]) * 0.5,
        "z": (a.get("z", 0.0) + b.get("z", 0.0)) * 0.5,
        "visibility": min(a.get("visibility", 1.0), b.get("visibility", 1.0)),
        "presence": min(a.get("presence", 1.0), b.get("presence", 1.0)),
    }


def image_plane_angles_from_sample(sample: dict, arm: str) -> dict:
    pose_landmarks = sample.get("pose_landmarks") or []
    if not pose_landmarks:
        return {}
    landmarks = pose_landmarks[0]
    ids = ROM_LANDMARKS[arm]
    if max(ids.values()) >= len(landmarks):
        return {}
    shoulder = landmarks[ids["shoulder"]]
    elbow = landmarks[ids["elbow"]]
    wrist = landmarks[ids["wrist"]]
    index = landmarks[ids["index"]]
    pinky = landmarks[ids["pinky"]]
    hand = midpoint_landmark(index, pinky) or wrist
    confidence_values = [
        value
        for value in (
            dict_landmark_confidence(landmarks[idx])
            for idx in ids.values()
        )
        if value is not None
    ]
    return {
        "source": "pose_landmarks_image_plane",
        "convention": "0_down_90_left_180_up",
        "humerus_flexion": image_plane_flexion_from_dict(shoulder, elbow),
        "reach_flexion": image_plane_flexion_from_dict(shoulder, hand or wrist or elbow),
        "confidence": min(confidence_values) if confidence_values else None,
    }


def ensure_image_plane_rom_summary(capture: dict) -> bool:
    if capture.get("type") != "rom_sweep" or capture.get("image_plane_rom_summary"):
        return False
    samples = capture.get("samples") or []
    for sample in samples:
        if sample.get("image_plane_angles"):
            continue
        sample["image_plane_angles"] = {
            "L": image_plane_angles_from_sample(sample, "L"),
            "R": image_plane_angles_from_sample(sample, "R"),
        }
    capture["image_plane_rom_summary"] = summarize_image_plane_rom_sweep(capture)
    return True


def load_capture(path: Path) -> dict:
    data = json.loads(path.read_text())
    changed = ensure_image_plane_rom_summary(data)
    if data.get("type") != "rom_sweep" and "angle_summary" not in data:
        data["angle_summary"] = summarize_capture(data)
        changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    data["_path"] = path
    return data


def angle_stats(capture: dict):
    return capture["angle_summary"]["angles"]


def stat_values(capture: dict):
    stats = angle_stats(capture)
    prevailing = []
    lower = []
    upper = []
    labels = []

    for arm, angle_name, label in ANGLE_KEYS:
        item = stats[arm][angle_name]
        value = item["prevailing_angle"]
        labels.append(label)
        prevailing.append(np.nan if value is None else value)
        lower.append(0.0 if value is None else max(value - item["min"], 0.0))
        upper.append(0.0 if value is None else max(item["max"] - value, 0.0))

    return labels, np.array(prevailing), np.array([lower, upper])


def capture_title(capture: dict) -> str:
    pose = capture.get("pose", "unknown")
    mode = capture.get("measurement_mode", "unknown")
    view = capture.get("view", "unknown")
    return f"{pose} | {view} | {mode}"


def plot_capture(capture: dict, output_path: Path) -> None:
    labels, prevailing, error = stat_values(capture)
    x = np.arange(len(labels))
    colors = ["#4AA8FF", "#4DD17A", "#F5A64A", "#E8667A"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x, prevailing, yerr=error, capsize=6, color=colors, alpha=0.88)
    ax.axhline(0, color="#30343b", linewidth=1)
    ax.axhline(90, color="#828892", linewidth=1, linestyle="--")
    ax.axhline(180, color="#828892", linewidth=1, linestyle=":")
    ax.set_ylim(-90, 200)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Angle (deg)")
    ax.set_title(capture_title(capture))
    ax.grid(axis="y", alpha=0.22)

    summary = capture["angle_summary"]
    subtitle = (
        f"prevailing=median | samples={summary['capture_window_sample_count']} | "
        f"window={summary['capture_window_duration_seconds']:.2f}s"
    )
    ax.text(
        0.01,
        -0.18,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_session(session_id: str, captures: list[dict], output_path: Path) -> None:
    captures_by_pose = {capture.get("pose"): capture for capture in captures}
    ordered = [
        captures_by_pose[pose]
        for pose in POSE_ORDER
        if pose in captures_by_pose
    ]
    if not ordered:
        return

    mode = ordered[0].get("measurement_mode", "unknown")
    view = ordered[0].get("view", "unknown")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    colors = ["#4AA8FF", "#4DD17A", "#F5A64A", "#E8667A"]

    for ax, capture in zip(axes, ordered):
        labels, prevailing, error = stat_values(capture)
        x = np.arange(len(labels))
        ax.bar(x, prevailing, yerr=error, capsize=4, color=colors, alpha=0.88)
        ax.axhline(0, color="#30343b", linewidth=1)
        ax.axhline(90, color="#828892", linewidth=1, linestyle="--")
        ax.axhline(180, color="#828892", linewidth=1, linestyle=":")
        ax.set_ylim(-90, 200)
        ax.set_xticks(x, labels, rotation=0)
        ax.set_title(capture.get("pose", "unknown"))
        ax.grid(axis="y", alpha=0.22)

    for ax in axes[len(ordered):]:
        ax.axis("off")

    fig.suptitle(
        f"Session {session_id} | {view} | {mode} | bar=prevailing median, whisker=min/max",
        fontsize=14,
    )
    fig.supxlabel("Angle channel")
    fig.supylabel("Angle (deg)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rom_sweep(capture: dict, output_path: Path) -> None:
    arm = capture.get("arm", "L")
    samples = capture.get("samples", [])
    if not samples:
        return

    start_time = capture.get("started_at", samples[0].get("time_monotonic", 0.0))
    times = np.array([sample.get("time_monotonic", start_time) - start_time for sample in samples])
    live_flexion = np.array([
        sample.get("angles", {}).get(arm, {}).get("flexion", np.nan)
        for sample in samples
    ], dtype=float)
    live_abduction = np.array([
        sample.get("angles", {}).get(arm, {}).get("abduction", np.nan)
        for sample in samples
    ], dtype=float)
    summary = capture.get("rom_summary", {})
    advanced = capture.get("advanced_rom_summary", {})
    image_plane = capture.get("image_plane_rom_summary", {})
    humerus = advanced.get("humerus", {})
    reach = advanced.get("reach", {})
    image_humerus = image_plane.get("humerus", {})
    image_reach = image_plane.get("reach", {})

    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)
    ax = axes[0]
    ax.plot(times, live_flexion, color="#4AA8FF", linewidth=1.8, label="live humerus flex")
    ax.plot(times, live_abduction, color="#4DD17A", linewidth=1.3, label="live abduction")

    ax.axhline(0, color="#30343b", linewidth=1)
    ax.axhline(90, color="#828892", linewidth=1, linestyle="--")
    ax.axhline(180, color="#828892", linewidth=1, linestyle=":")
    min_time = summary.get("min_flexion_time_offset")
    max_time = summary.get("max_flexion_time_offset")
    if min_time is not None:
        ax.axvline(min_time, color="#F5A64A", linestyle="--", linewidth=1.5, label="min flex")
    if max_time is not None:
        ax.axvline(max_time, color="#E8667A", linestyle="--", linewidth=1.5, label="max flex")
    ax.set_ylim(-90, 210)
    ax.set_ylabel("Angle (deg)")
    ax.grid(alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    ax.set_title("Live displayed angles")

    ax = axes[1]
    for label, item, color in (
        ("unwrapped humerus", humerus, "#4AA8FF"),
        ("unwrapped reach", reach, "#E8667A"),
    ):
        trace = item.get("trace", [])
        if not trace:
            continue
        trace_times = np.array([point["time_offset"] for point in trace])
        trace_angles = np.array([point["angle"] for point in trace])
        ax.plot(trace_times, trace_angles, color=color, linewidth=2.2, label=label)
        max_time = item.get("max_time_offset")
        if max_time is not None:
            ax.axvline(max_time, color=color, linestyle="--", linewidth=1.2, alpha=0.7)

    ax.axhline(0, color="#30343b", linewidth=1)
    ax.axhline(90, color="#828892", linewidth=1, linestyle="--")
    ax.axhline(180, color="#828892", linewidth=1, linestyle=":")
    ax.set_ylim(-30, 240)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unwrapped angle (deg)")
    ax.grid(alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    ax.set_title("Sweep-plane unwrapped ROM")

    ax = axes[2]
    for label, item, color in (
        ("2D image humerus", image_humerus, "#00A6A6"),
        ("2D image reach", image_reach, "#F28E2B"),
    ):
        trace = item.get("trace", [])
        if not trace:
            continue
        trace_times = np.array([point["time_offset"] for point in trace])
        trace_angles = np.array([point["angle"] for point in trace])
        ax.plot(trace_times, trace_angles, color=color, linewidth=2.0, label=label)
        max_time = item.get("max_time_offset")
        if max_time is not None:
            ax.axvline(max_time, color=color, linestyle="--", linewidth=1.2, alpha=0.7)

    ax.axhline(0, color="#30343b", linewidth=1)
    ax.axhline(90, color="#828892", linewidth=1, linestyle="--")
    ax.axhline(180, color="#828892", linewidth=1, linestyle=":")
    ax.set_ylim(-30, 240)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Image angle (deg)")
    ax.grid(alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    ax.set_title("2D fixed-camera flexion estimate")

    reach_rom = reach.get("rom")
    humerus_rom = humerus.get("rom")
    image_humerus_max = image_humerus.get("max_angle")
    title = (
        f"{capture.get('pose')} | {capture.get('view')} | {capture.get('measurement_mode')} | "
        f"humerus ROM={humerus_rom:.1f} deg | reach ROM={reach_rom:.1f} deg"
        + (
            f" | 2D hum max={image_humerus_max:.1f} deg"
            if image_humerus_max is not None
            else ""
        )
        if humerus_rom is not None and reach_rom is not None
        else f"{capture.get('pose')} | ROM unavailable"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def visualize(capture_dir: Path) -> list[Path]:
    captures = [load_capture(path) for path in sorted(capture_dir.glob("*.json"))]
    generated = []

    sessions = defaultdict(list)
    for capture in captures:
        json_path = capture["_path"]
        if capture.get("type") == "rom_sweep":
            rom_output = json_path.with_name(f"{json_path.stem}_summary.png")
            plot_rom_sweep(capture, rom_output)
            generated.append(rom_output)
            continue

        capture_output = json_path.with_name(f"{json_path.stem}_summary.png")
        plot_capture(capture, capture_output)
        generated.append(capture_output)
        sessions[capture.get("session_id", "unknown")].append(capture)

    for session_id, session_captures in sorted(sessions.items()):
        session_output = capture_dir / f"{session_id}_session_summary.png"
        plot_session(session_id, session_captures, session_output)
        generated.append(session_output)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual summaries for diagnostic capture JSON files."
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("captures"),
        help="Directory containing diagnostic capture JSON files.",
    )
    args = parser.parse_args()

    generated = visualize(args.capture_dir)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
