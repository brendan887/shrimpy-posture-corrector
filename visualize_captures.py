from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from live_pose_full import summarize_capture


ANGLE_KEYS = (
    ("L", "flexion", "L flex"),
    ("L", "abduction", "L abd"),
    ("R", "flexion", "R flex"),
    ("R", "abduction", "R abd"),
)
POSE_ORDER = ("arms_down", "arms_forward", "arms_side_t", "arms_overhead")


def load_capture(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("type") != "rom_sweep" and "angle_summary" not in data:
        data["angle_summary"] = summarize_capture(data)
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
    times = np.array([
        sample.get("time_monotonic", start_time) - start_time
        for sample in samples
    ])
    flexion = np.array([
        sample.get("angles", {}).get(arm, {}).get("flexion", np.nan)
        for sample in samples
    ], dtype=float)
    abduction = np.array([
        sample.get("angles", {}).get(arm, {}).get("abduction", np.nan)
        for sample in samples
    ], dtype=float)
    summary = capture.get("rom_summary", {})

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(times, flexion, color="#4AA8FF", linewidth=2.5, label=f"{arm} flexion")
    ax.plot(times, abduction, color="#4DD17A", linewidth=1.8, label=f"{arm} abduction")
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
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    title = (
        f"{capture.get('pose')} | {capture.get('view')} | {capture.get('measurement_mode')} | "
        f"ROM={summary.get('rom'):.1f} deg"
        if summary.get("rom") is not None
        else f"{capture.get('pose')} | ROM unavailable"
    )
    ax.set_title(title)
    fig.tight_layout()
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
