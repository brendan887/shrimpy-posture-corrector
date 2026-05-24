"""
Patient-facing UI for Stretch Goals -- the autonomous ROM assessment and
robot-assisted stretch.

Layout:
    - Slim header bar at top with just the phase title
    - Camera + pose skeleton on the LEFT half of the screen
    - All instructional text on the RIGHT half (kicker, title, bullets,
      workflow checklist or comparison report)
    - A small floating angle readout overlaid in the bottom-right of the
      camera area so the user can see their live measurement at a glance

A single UIState dataclass drives everything. Wiring functionality later is a
matter of populating UIState each frame.

Text rendering uses PIL (Pillow) with a TrueType system font for crisp
antialiased text and proper Unicode glyphs. All draw_text calls during a
render pass are batched and flushed in one PIL roundtrip.

Backwards-compatible exports for live_pose_full.py:
    VIEW_MODES, POSE_CONNECTIONS, draw_pose, is_visible, landmark_point,
    render_frame
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Backwards-compatible constants used by live_pose_full.py
# ---------------------------------------------------------------------------

VIEW_MODES = ("front", "left-45", "right-45", "left-side", "right-side")

VIEW_GUIDANCE = {
    "front": {
        "title": "Front-view mode",
        "instruction": "Camera centered in front. Best for abduction; flexion is more depth-sensitive.",
    },
    "left-45": {
        "title": "Left 45-degree mode",
        "instruction": "Place camera at your left-front 45 deg angle. ROM defaults to camera-side R; arrows can switch.",
    },
    "right-45": {
        "title": "Right 45-degree mode",
        "instruction": "Place camera at your right-front 45 deg angle. ROM defaults to camera-side L; arrows can switch.",
    },
    "left-side": {
        "title": "Left side-view mode",
        "instruction": "Place camera near your left side. ROM defaults to camera-side R; arrows can switch.",
    },
    "right-side": {
        "title": "Right side-view mode",
        "instruction": "Place camera near your right side. ROM defaults to camera-side L; arrows can switch.",
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


# ---------------------------------------------------------------------------
# Design tokens -- shrimp red on white
# ---------------------------------------------------------------------------

class C:
    # Surfaces -- visibly pink shrimp palette
    SURFACE_DEEP   = (224, 232, 255)   # light shrimp pink for header
    SURFACE_PANEL  = (232, 240, 255)   # lighter pink for content panels
    SURFACE_CARD   = (210, 222, 252)   # soft shrimp pink for cards
    SURFACE_HAIR   = (190, 205, 240)   # dusty pink dividers

    # Text
    TEXT_PRIMARY   = ( 45,  40,  55)
    TEXT_SECONDARY = (110, 100, 110)
    TEXT_DIM       = (165, 155, 165)

    # Brand + states (BGR)
    ACCENT         = ( 70,  57, 230)   # shrimp red -- brand / active
    ACCENT_DEEP    = ( 45,  35, 175)
    SUCCESS        = ( 90, 175,  95)
    WARN           = ( 40, 140, 230)
    FAIL           = ( 45,  35, 145)
    ROBOT          = (160,  70, 180)

    # Skeleton (drawn directly on camera)
    BONE           = ( 70,  57, 230)
    JOINT          = ( 45,  35, 145)
    JOINT_OUTLINE  = (255, 255, 255)


# ---------------------------------------------------------------------------
# Text engine: PIL primary, cv2 fallback
# ---------------------------------------------------------------------------

_FONT_CANDIDATES_REGULAR = (
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_FONT_CANDIDATES_BOLD = (
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

_FONT_REGULAR_PATH: Optional[str] = next((p for p in _FONT_CANDIDATES_REGULAR if os.path.exists(p)), None)
_FONT_BOLD_PATH: Optional[str] = next((p for p in _FONT_CANDIDATES_BOLD if os.path.exists(p)), None)

_FONT_CACHE: dict[tuple[int, int], object] = {}


def _get_font(size: float, weight: int):
    if not _HAS_PIL:
        return None
    key = (int(round(size)), 1 if weight >= 2 else 0)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    path = _FONT_BOLD_PATH if weight >= 2 else _FONT_REGULAR_PATH
    if path is None:
        return None
    try:
        if path.endswith(".ttc") and weight >= 2:
            try:
                font = ImageFont.truetype(path, int(round(size)), index=1)
            except (OSError, IndexError):
                font = ImageFont.truetype(path, int(round(size)))
        else:
            font = ImageFont.truetype(path, int(round(size)))
    except (OSError, IOError):
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _measure_text(text: str, size: float, weight: int) -> tuple[int, int]:
    font = _get_font(size, weight)
    if font is None:
        scale = size / 30.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
        return tw, th
    try:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        return int(len(text) * size * 0.55), int(size)


@dataclass
class _TextBatch:
    ops: list = field(default_factory=list)

    def add(self, x: int, y: int, text: str, size: float, weight: int, color_bgr: tuple) -> None:
        self.ops.append((x, y, text, size, weight, color_bgr))


_current_batch: Optional[_TextBatch] = None


@contextmanager
def _text_batch(frame):
    global _current_batch
    previous = _current_batch
    if _HAS_PIL and _FONT_REGULAR_PATH is not None:
        _current_batch = _TextBatch()
    else:
        _current_batch = None
    try:
        yield
    finally:
        if _current_batch is not None and _current_batch.ops:
            _flush_batch(frame, _current_batch)
        _current_batch = previous


def _flush_batch(frame, batch: _TextBatch) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for x, y, text, size, weight, color_bgr in batch.ops:
        font = _get_font(size, weight)
        if font is None:
            continue
        rgb_color = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        kwargs = {}
        if weight >= 2 and size >= 30:
            kwargs["stroke_width"] = 1
            kwargs["stroke_fill"] = rgb_color
        try:
            draw.text((x, y), text, font=font, fill=rgb_color, **kwargs)
        except Exception:
            draw.text((x, y), text, font=font, fill=rgb_color)
    back = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    frame[:] = back


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _alpha_rect(frame, top_left, bottom_right, color, alpha: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _rounded_filled(img, top_left, bottom_right, color, radius: int = 12) -> None:
    """Filled rounded rectangle drawn directly on img."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if radius == 0:
        cv2.rectangle(img, top_left, bottom_right, color, -1, cv2.LINE_AA)
        return
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),  90, 0, 90, color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),   0, 0, 90, color, -1, cv2.LINE_AA)


def _rounded_stroke(img, top_left, bottom_right, color, thickness: int = 2, radius: int = 12) -> None:
    """Rounded rectangle outline."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if radius == 0:
        cv2.rectangle(img, top_left, bottom_right, color, thickness, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),  90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),   0, 0, 90, color, thickness, cv2.LINE_AA)


def _rounded_alpha_rect(frame, top_left, bottom_right, color, alpha: float,
                        radius: int = 12) -> None:
    overlay = frame.copy()
    _rounded_filled(overlay, top_left, bottom_right, color, radius)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


# ---------------------------------------------------------------------------
# RGBA image compositing (for sprite-sheet animations)
# ---------------------------------------------------------------------------

def composite_rgba(canvas, rgba, x: int, y: int, opacity: float = 1.0) -> None:
    """Alpha-blend an RGBA image onto canvas at (x, y). Per-pixel alpha is
    multiplied by `opacity` (1.0 = full). Out-of-bounds regions are clipped."""
    if rgba is None or opacity <= 0.0:
        return
    ih, iw = rgba.shape[:2]
    ch, cw = canvas.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(cw, x + iw)
    y2 = min(ch, y + ih)
    if x2 <= x1 or y2 <= y1:
        return
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    src = rgba[sy1:sy2, sx1:sx2]
    dst = canvas[y1:y2, x1:x2]
    alpha = (src[:, :, 3:4].astype(np.float32) / 255.0) * float(opacity)
    blended = src[:, :, :3].astype(np.float32) * alpha + dst.astype(np.float32) * (1.0 - alpha)
    canvas[y1:y2, x1:x2] = blended.astype(np.uint8)


def _key_white_to_alpha(bgr, threshold: int = 230) -> np.ndarray:
    """Convert an RGB/BGR image to BGRA, treating near-white pixels as
    transparent. Threshold is the per-channel minimum to count as background."""
    if bgr.shape[2] == 4:
        return bgr
    b, g, r = cv2.split(bgr)
    a = np.full_like(b, 255)
    white_mask = (r >= threshold) & (g >= threshold) & (b >= threshold)
    a[white_mask] = 0
    # Soften the edge: 1px erosion of the silhouette mask reduces white halo.
    fg_mask = (~white_mask).astype(np.uint8) * 255
    fg_mask = cv2.erode(fg_mask, np.ones((2, 2), np.uint8), iterations=1)
    a = np.where(fg_mask > 0, a, 0).astype(np.uint8)
    return cv2.merge([b, g, r, a])


_ARM_RAISE_FRAMES: Optional[list] = None
_ARM_RAISE_TARGET_H = 320


def _load_arm_raise_frames():
    """Load step1/2/3.png from images/, key the white background, and pad each
    frame onto a uniform-size RGBA canvas so they can be cross-faded by
    pixel-wise blending. Cached after the first call."""
    global _ARM_RAISE_FRAMES
    if _ARM_RAISE_FRAMES is not None:
        return _ARM_RAISE_FRAMES

    here = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(here, "images", f"step{i}.png") for i in (1, 2, 3)]
    if not all(os.path.exists(p) for p in paths):
        _ARM_RAISE_FRAMES = []
        return _ARM_RAISE_FRAMES

    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            _ARM_RAISE_FRAMES = []
            return _ARM_RAISE_FRAMES
        rgba = _key_white_to_alpha(img)
        ih, iw = rgba.shape[:2]
        scale = _ARM_RAISE_TARGET_H / float(ih)
        new_w = max(2, int(round(iw * scale)))
        rgba = cv2.resize(rgba, (new_w, _ARM_RAISE_TARGET_H),
                          interpolation=cv2.INTER_AREA)
        frames.append(rgba)

    # Pad each frame onto a canvas sized to the widest frame (bottom-aligned,
    # horizontally centered) so all three share the same coordinate system.
    max_w = max(f.shape[1] for f in frames)
    uniformed = []
    for f in frames:
        canvas = np.zeros((_ARM_RAISE_TARGET_H, max_w, 4), dtype=np.uint8)
        fh, fw = f.shape[:2]
        x_off = (max_w - fw) // 2
        canvas[:, x_off:x_off + fw] = f
        uniformed.append(canvas)

    _ARM_RAISE_FRAMES = uniformed
    return _ARM_RAISE_FRAMES


def draw_arm_raise_animation(frame, state: UIState, now: float) -> None:
    """Top-right corner of the camera area, when state.show_arm_raise_animation.
    Smooth palindrome cycle (0 -> 1 -> 2 -> 1 -> 0) driven by a cosine so the
    motion eases in/out at each key pose."""
    if not getattr(state, "show_arm_raise_animation", False):
        return
    frames = _load_arm_raise_frames()
    if not frames:
        return

    h, w = frame.shape[:2]
    cx, cy, cw, _ch = _camera_bounds(w, h)

    # Sinusoid 0..2..0, slowing at the extremes (acts as a natural hold).
    cycle = 3.0
    pos = 1.0 - math.cos(2.0 * math.pi * (now % cycle) / cycle)  # [0, 2]
    if pos >= 1.0:
        low_idx, high_idx, alpha_t = 1, 2, pos - 1.0
    else:
        low_idx, high_idx, alpha_t = 0, 1, pos
    alpha_t = max(0.0, min(1.0, alpha_t))

    fh, fw = frames[0].shape[:2]
    x = cx + cw - fw - 24
    y = cy + 40

    # Two-pass composite: lower-weight frame first, then higher-weight on top.
    # The two key poses overlap (same chair + torso) so silhouette ghosting is
    # minimal and the moving arm naturally crossfades.
    composite_rgba(frame, frames[low_idx],  x, y, opacity=(1.0 - alpha_t))
    composite_rgba(frame, frames[high_idx], x, y, opacity=alpha_t)


def draw_panel(
    frame, x: int, y: int, w: int, h: int,
    *,
    fill=C.SURFACE_PANEL,
    alpha: float = 0.95,
    border=None,
    border_thickness: int = 1,
    accent_top=None,
    accent_thickness: int = 4,
) -> None:
    _alpha_rect(frame, (x, y), (x + w, y + h), fill, alpha)
    if border is not None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), border, border_thickness, cv2.LINE_AA)
    if accent_top is not None:
        cv2.rectangle(frame, (x, y), (x + w, y + accent_thickness), accent_top, -1)


_ASCII_FALLBACK = str.maketrans({"·": "-", "°": "", "—": "-", "•": "-", "…": "..."})


def draw_text(
    frame, text: str, x: int, y: int,
    *,
    size: float = 22,
    color=C.TEXT_PRIMARY,
    weight: int = 1,
    align: str = "left",
) -> int:
    global _current_batch
    if _current_batch is not None:
        tw, _th = _measure_text(text, size, weight)
        if align == "center":
            x = x - tw // 2
        elif align == "right":
            x = x - tw
        _current_batch.add(x, y, text, size, weight, color)
        return x + tw

    safe = text.translate(_ASCII_FALLBACK)
    scale = size / 30.0
    (tw, th), _ = cv2.getTextSize(safe, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
    if align == "center":
        x = x - tw // 2
    elif align == "right":
        x = x - tw
    cv2.putText(frame, safe, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, weight, cv2.LINE_AA)
    return x + tw


def draw_kicker(frame, text: str, x: int, y: int, color=C.TEXT_DIM,
                size: float = 15, uppercase: bool = True) -> None:
    display = text.upper() if uppercase else text
    draw_text(frame, display, x, y, size=size, color=color, weight=2)


def draw_pill(
    frame, text: str, x: int, y: int,
    *,
    fg=C.SURFACE_PANEL,
    bg=C.ACCENT,
    pad_x: int = 14,
    pad_y: int = 7,
    size: float = 14,
) -> int:
    tw, th = _measure_text(text, size, 2)
    w = tw + pad_x * 2
    h = th + pad_y * 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1, cv2.LINE_AA)
    draw_text(frame, text, x + pad_x, y + pad_y, size=size, color=fg, weight=2)
    return x + w


def draw_progress_bar(
    frame, x: int, y: int, w: int, h: int, fraction: float,
    *, bg=C.SURFACE_CARD, fg=C.ACCENT,
) -> None:
    fraction = max(0.0, min(1.0, fraction))
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1, cv2.LINE_AA)
    fw = int(w * fraction)
    if fw > 0:
        cv2.rectangle(frame, (x, y), (x + fw, y + h), fg, -1, cv2.LINE_AA)


def _draw_wrapped(frame, text: str, x: int, y: int, *,
                  max_width: int, size: float, color, weight: int = 1) -> int:
    """Word-wrap text inside max_width. Returns the y after the last line."""
    words = text.split()
    line = ""
    cy = y
    line_h = int(size * 1.35)
    for word in words:
        candidate = (line + " " + word).strip()
        tw, _ = _measure_text(candidate, size, weight)
        if tw > max_width and line:
            draw_text(frame, line, x, cy, size=size, color=color, weight=weight)
            line = word
            cy += line_h
        else:
            line = candidate
    if line:
        draw_text(frame, line, x, cy, size=size, color=color, weight=weight)
        cy += line_h
    return cy


# ---------------------------------------------------------------------------
# Logo (top-right brand mark)
# ---------------------------------------------------------------------------

_LOGO_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "logo.png"))
_LOGO_CACHE: object = None  # cached uint8 BGRA image, or False if missing


def _load_logo():
    """Lazy-load the shrimpy logo. Returns a BGRA uint8 array or None."""
    global _LOGO_CACHE
    if _LOGO_CACHE is not None:
        return _LOGO_CACHE if _LOGO_CACHE is not False else None
    if not os.path.exists(_LOGO_PATH):
        _LOGO_CACHE = False
        return None
    img = cv2.imread(_LOGO_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        _LOGO_CACHE = False
        return None
    # Ensure 4 channels (BGRA)
    if img.shape[2] == 3:
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
        img = bgra
    _LOGO_CACHE = img
    return img


def draw_logo(frame, x: int, y: int, size: int) -> None:
    """Composite the logo at (x, y) scaled so the longer side fits `size`.
    Uses the logo's alpha channel for transparent blending."""
    logo = _load_logo()
    if logo is None:
        return
    src_h, src_w = logo.shape[:2]
    scale = size / max(src_w, src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)
    fh, fw = frame.shape[:2]
    # Clip to frame bounds
    if x >= fw or y >= fh:
        return
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + new_w), min(fh, y + new_h)
    if x1 <= x0 or y1 <= y0:
        return
    src_x0, src_y0 = x0 - x, y0 - y
    src_x1, src_y1 = src_x0 + (x1 - x0), src_y0 + (y1 - y0)
    patch = resized[src_y0:src_y1, src_x0:src_x1]
    bgr = patch[:, :, :3].astype(np.float32)
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    region = frame[y0:y1, x0:x1].astype(np.float32)
    blended = bgr * alpha + region * (1.0 - alpha)
    frame[y0:y1, x0:x1] = blended.astype(np.uint8)


# ---------------------------------------------------------------------------
# Pose skeleton
# ---------------------------------------------------------------------------

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
                cv2.line(frame, start, end, C.BONE, 4, cv2.LINE_AA)

        for point in points:
            if point:
                cv2.circle(frame, point, 6, C.JOINT, -1, cv2.LINE_AA)
                cv2.circle(frame, point, 9, C.JOINT_OUTLINE, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# UI state model -- single source of truth the colleague will populate
# ---------------------------------------------------------------------------

PHASE_LABELS = {
    0: "Get Ready",
    1: "Assessment",
    2: "Robot-Assisted Stretch",
    3: "Reassessment & Report",
}

PHASE_ACCENTS = {
    0: C.ACCENT,
    1: C.ACCENT,
    2: C.ROBOT,
    3: C.SUCCESS,
}


@dataclass
class ChecklistItem:
    label: str
    status: str = "pending"   # done | active | pending | failed
    detail: str = ""
def draw_angle_panel(frame, angles, image_plane_angles=None, now: float = 0.0) -> None:
    image_plane_angles = image_plane_angles or {}
    panel_width, panel_height = 690, 172
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
        elbow_angle = image_angles.get("elbow_angle")
        elbow_warning = elbow_angle is not None and elbow_angle < 130.0
        flash_on = int(now * 4) % 2 == 0
        elbow_color = (
            (40, 40, 255)
            if elbow_warning and flash_on
            else (80, 80, 150)
            if elbow_warning
            else (110, 235, 255)
        )
        cv2.putText(
            frame,
            f"   elbow {format_angle(elbow_angle)} deg",
            (x + 350, row_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            elbow_color,
            2,
            cv2.LINE_AA,
        )


@dataclass
class AngleReading:
    label: str
    arm: str                            # "L", "R", or "—"
    value: Optional[float] = None
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    status: str = "live"                # live | hold | pass | fail


@dataclass
class ComparisonRow:
    label: str
    before: Optional[float] = None
    after: Optional[float] = None
    target: Optional[float] = None


@dataclass
class UIState:
    """All visual state. Populate this each frame; the renderer does the rest."""
    # Header
    phase: int = 1
    step_index: int = 0
    step_total: int = 4

    # Right panel content
    hero_kicker: str = ""
    hero_title: str = "Welcome"
    hero_subtitle: str = ""
    hero_bullets: Optional[list[str]] = None

    # Workflow checklist (renders at bottom of right panel)
    checklist: list[ChecklistItem] = field(default_factory=list)

    # Live angle readout (renders as small overlay on camera area)
    angles: list[AngleReading] = field(default_factory=list)

    # Camera-area overlays
    countdown_seconds: Optional[float] = None
    countdown_label: str = ""
    big_callout: Optional[str] = None
    big_callout_color: tuple = C.ACCENT
    # When True, render the 3-frame arm-raise demo loop in the top-right
    # corner of the camera area (workflow step 2 uses this).
    show_arm_raise_animation: bool = False
    # When True, replace the camera + right-panel layout with a fullscreen
    # presenter view built from state.comparison (workflow step 5 uses this).
    show_fullscreen_results: bool = False

    # Phase-2 robot status (renders inside the right panel)
    robot_state: Optional[str] = None
    robot_safety: Optional[str] = None

    # Phase-3 comparison (takes over the right panel when set)
    comparison: Optional[list[ComparisonRow]] = None
    comparison_caption: str = ""

    # Footer hint (bottom-right of right panel)
    footer_hint: str = "Press space to continue   ·   Press q to quit"

    # Operator overlay (dev only)
    dev_overlay: Optional[str] = None


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

HEADER_H   = 108           # slim header band, sized to fit the shrimpy mark
PANEL_PAD  = 32            # interior padding for right panel
CAM_PAD    = 18            # margin between camera overlays and frame edge


def _camera_bounds(w: int, h: int) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the camera area beneath the header. Two-thirds wide."""
    cam_w = (w * 2) // 3
    return 0, HEADER_H, cam_w, h - HEADER_H


def _right_panel_bounds(w: int, h: int) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the right panel. One-third wide."""
    cam_w = (w * 2) // 3
    return cam_w, HEADER_H, w - cam_w, h - HEADER_H


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def draw_phase_header(frame, state: UIState) -> None:
    """Slim header bar: phase title left, shrimpy brand mark right."""
    h, w = frame.shape[:2]
    accent = PHASE_ACCENTS.get(state.phase, C.ACCENT)
    draw_panel(frame, 0, 0, w, HEADER_H, fill=C.SURFACE_DEEP, alpha=0.98,
               accent_top=accent, accent_thickness=5)

    if state.phase == 0:
        label = "Welcome to Stretch Goals"
    else:
        label = f"Phase {state.phase}: {PHASE_LABELS.get(state.phase, '')}"
    title_y = (HEADER_H - 38) // 2
    draw_text(frame, label, 32, title_y, size=34, color=C.TEXT_PRIMARY, weight=2)

    # Shrimpy brand mark, top right -- fills most of the header height with a
    # small margin above (the accent stripe) and below (the divider).
    logo_size = HEADER_H - 14
    draw_logo(frame, w - logo_size - 16, 7, logo_size)


# ---------------------------------------------------------------------------
# Right panel -- all instructional text lives here
# ---------------------------------------------------------------------------

def draw_right_panel(frame, state: UIState) -> None:
    """Right half of the screen: kicker + title + bullets/subtitle, with
    workflow checklist anchored to the bottom. When state.comparison is set,
    the panel switches to a before/after report instead."""
    h, w = frame.shape[:2]
    px, py, pw, ph = _right_panel_bounds(w, h)
    accent = PHASE_ACCENTS.get(state.phase, C.ACCENT)

    # Panel background
    draw_panel(frame, px, py, pw, ph, fill=C.SURFACE_PANEL, alpha=0.98,
               border=C.SURFACE_HAIR)
    # Vertical accent stripe on the left edge of the panel
    cv2.rectangle(frame, (px, py), (px + 5, py + ph), accent, -1)

    if state.comparison:
        _draw_panel_comparison(frame, state, px, py, pw, ph)
    else:
        _draw_panel_instruction(frame, state, px, py, pw, ph, accent)

    # Footer hint anchored to the bottom-right of the panel -- but only when
    # we have not already rendered it inline under the bullets.
    if not state.hero_bullets and state.footer_hint and not state.comparison:
        draw_text(frame, state.footer_hint, px + pw - PANEL_PAD, py + ph - 32,
                  size=14, color=C.TEXT_DIM, align="right")


def _draw_panel_instruction(frame, state, px, py, pw, ph, accent) -> None:
    pad = PANEL_PAD
    inner_x = px + pad + 5  # +5 for the accent stripe
    inner_w = pw - pad * 2 - 5

    cy = py + pad

    if state.hero_kicker:
        # Long kickers (like the welcome lead) render as title-case sentences
        # in the accent color; short labels render as tracked uppercase.
        long_kicker = len(state.hero_kicker) > 32
        kicker_size = 19 if long_kicker else 16
        if long_kicker:
            cy = _draw_wrapped(frame, state.hero_kicker, inner_x, cy,
                               max_width=inner_w, size=kicker_size,
                               color=accent, weight=2)
            cy += 8
            # Sub-caption sits right under the lead with the SAME styling
            # (same size, weight, color) so the two intro lines read as one
            # cohesive block.
            if state.hero_bullets and state.footer_hint:
                cy = _draw_wrapped(frame, state.footer_hint, inner_x, cy,
                                   max_width=inner_w, size=kicker_size,
                                   color=accent, weight=2)
                cy += 12
            else:
                cy += 6
        else:
            draw_kicker(frame, state.hero_kicker, inner_x, cy,
                        color=accent, size=kicker_size, uppercase=True)
            cy += 30

    # Hero title -- wrap to fit narrow panel
    cy = _draw_wrapped(frame, state.hero_title, inner_x, cy,
                       max_width=inner_w, size=28, color=C.TEXT_PRIMARY, weight=2)
    cy += 12

    if state.hero_bullets:
        for i, bullet in enumerate(state.hero_bullets):
            num_cx = inner_x + 20
            num_cy = cy + 20
            cv2.circle(frame, (num_cx, num_cy), 20, accent, -1, cv2.LINE_AA)
            draw_text(frame, str(i + 1), num_cx, num_cy - 14, size=20,
                      color=C.SURFACE_PANEL, weight=2, align="center")
            end_y = _draw_wrapped(frame, bullet, inner_x + 52, cy + 2,
                                  max_width=inner_w - 52, size=19,
                                  color=C.TEXT_PRIMARY)
            cy = max(end_y, num_cy + 22) + 8
    elif state.hero_subtitle:
        cy = _draw_wrapped(frame, state.hero_subtitle, inner_x, cy,
                           max_width=inner_w, size=19, color=C.TEXT_SECONDARY)
        cy += 10

    # Robot status: render inline within the right panel (Phase 2 only).
    # Flows naturally below the instruction so the workflow strip still fits.
    if state.phase == 2 and (state.robot_state or state.robot_safety):
        cy += 12
        cv2.line(frame, (inner_x, cy), (inner_x + inner_w, cy),
                 C.SURFACE_HAIR, 1, cv2.LINE_AA)
        cy += 14
        draw_kicker(frame, "Robot status", inner_x, cy, color=C.ROBOT, size=14)
        cy += 24
        if state.robot_state:
            cy = _draw_wrapped(frame, state.robot_state, inner_x, cy,
                               max_width=inner_w, size=17,
                               color=C.TEXT_PRIMARY, weight=2)
            cy += 6
        if state.robot_safety:
            draw_pill(frame, "SAFETY", inner_x, cy,
                      fg=C.SURFACE_PANEL, bg=C.WARN, size=12)
            draw_text(frame, state.robot_safety, inner_x + 90, cy + 4,
                      size=12, color=C.TEXT_SECONDARY)
            cy += 32

    # Workflow checklist anchored to the bottom. Reserved area sized to the
    # checklist, so short 3-step lists fit without crowding the instruction
    # content above.
    if state.checklist:
        # Bottom-padding: leave a little room above the footer-hint slot
        # (or above the panel edge when bullets are showing).
        bottom_pad = 32 if state.hero_bullets else 50
        items_h = sum(40 if i.detail else 30 for i in state.checklist)
        workflow_h = items_h + 40          # kicker (18) + paddings (~22)
        workflow_top = py + ph - workflow_h - bottom_pad
        if workflow_top > cy + 4:
            cv2.line(frame, (inner_x, workflow_top), (inner_x + inner_w, workflow_top),
                     C.SURFACE_HAIR, 1, cv2.LINE_AA)
            wy = workflow_top + 10
            draw_kicker(frame, "Workflow", inner_x, wy, color=accent, size=11)
            wy += 20
            shown = 0
            for i, item in enumerate(state.checklist):
                row_h = 40 if item.detail else 30
                if wy + row_h > py + ph - bottom_pad + 2:
                    break
                wy = _draw_checklist_row(frame, item, i + 1, inner_x, wy,
                                         inner_w, accent)
                shown += 1
            hidden = len(state.checklist) - shown
            if hidden > 0 and wy + 18 <= py + ph - bottom_pad + 16:
                draw_text(frame, f"+{hidden} more step{'s' if hidden != 1 else ''}",
                          inner_x + 38, wy + 2, size=11, color=C.TEXT_DIM)


def _status_color(status: str) -> tuple:
    return {
        "done":    C.SUCCESS,
        "active":  C.ACCENT,
        "pending": C.TEXT_DIM,
        "failed":  C.FAIL,
    }.get(status, C.TEXT_SECONDARY)


def _draw_checklist_row(frame, item: ChecklistItem, number: int,
                        x: int, y: int, w: int, accent: tuple) -> int:
    """Workflow row: small numbered circle + label. Only the active step is
    accented; pending steps are greyed out."""
    num_r = 12
    num_cx = x + num_r + 2
    num_cy = y + num_r + 2

    pending_grey = (180, 178, 188)  # warm light grey for inactive markers

    if item.status == "done":
        cv2.circle(frame, (num_cx, num_cy), num_r, accent, -1, cv2.LINE_AA)
        cv2.line(frame, (num_cx - 4, num_cy + 1),
                 (num_cx - 1, num_cy + 4), C.SURFACE_PANEL, 2, cv2.LINE_AA)
        cv2.line(frame, (num_cx - 1, num_cy + 4),
                 (num_cx + 5, num_cy - 3), C.SURFACE_PANEL, 2, cv2.LINE_AA)
    elif item.status == "failed":
        cv2.circle(frame, (num_cx, num_cy), num_r, C.FAIL, -1, cv2.LINE_AA)
        cv2.line(frame, (num_cx - 4, num_cy - 4),
                 (num_cx + 4, num_cy + 4), C.SURFACE_PANEL, 2, cv2.LINE_AA)
        cv2.line(frame, (num_cx + 4, num_cy - 4),
                 (num_cx - 4, num_cy + 4), C.SURFACE_PANEL, 2, cv2.LINE_AA)
    elif item.status == "active":
        cv2.circle(frame, (num_cx, num_cy), num_r, accent, -1, cv2.LINE_AA)
        cv2.circle(frame, (num_cx, num_cy), num_r + 3, accent, 2, cv2.LINE_AA)
        draw_text(frame, str(number), num_cx, num_cy - 9, size=13,
                  color=C.SURFACE_PANEL, weight=2, align="center")
    else:  # pending -- greyed out
        cv2.circle(frame, (num_cx, num_cy), num_r, pending_grey, -1, cv2.LINE_AA)
        draw_text(frame, str(number), num_cx, num_cy - 9, size=13,
                  color=C.SURFACE_PANEL, weight=2, align="center")

    text_color = (C.TEXT_PRIMARY if item.status == "active"
                  else C.TEXT_SECONDARY if item.status != "done"
                  else C.TEXT_DIM)
    weight = 2 if item.status == "active" else 1
    draw_text(frame, item.label, x + num_r * 2 + 14, y + 3,
              size=14, color=text_color, weight=weight)
    if item.detail:
        draw_text(frame, item.detail, x + num_r * 2 + 14, y + 22,
                  size=11, color=C.TEXT_DIM)
        return y + 40
    return y + 30


def _draw_panel_comparison(frame, state, px, py, pw, ph) -> None:
    """Phase 3 report -- before/after table replacing the instruction content."""
    pad = PANEL_PAD
    inner_x = px + pad + 5
    inner_w = pw - pad * 2 - 5

    cy = py + pad
    draw_kicker(frame, "Before / After report", inner_x, cy, color=C.SUCCESS, size=14)
    cy += 28
    draw_text(frame, "Range of motion comparison", inner_x, cy, size=28,
              color=C.TEXT_PRIMARY, weight=2)
    cy += 50

    col_label  = inner_x
    col_before = inner_x + inner_w - 280
    col_after  = inner_x + inner_w - 170
    col_delta  = inner_x + inner_w - 60
    draw_kicker(frame, "Movement", col_label, cy, color=C.TEXT_DIM, size=12)
    draw_kicker(frame, "Before",   col_before, cy, color=C.TEXT_DIM, size=12)
    draw_kicker(frame, "After",    col_after,  cy, color=C.TEXT_DIM, size=12)
    draw_kicker(frame, "Delta",    col_delta,  cy, color=C.TEXT_DIM, size=12)
    cy += 32

    for row in state.comparison or []:
        cv2.line(frame, (inner_x, cy - 8), (inner_x + inner_w, cy - 8),
                 C.SURFACE_HAIR, 1, cv2.LINE_AA)
        draw_text(frame, row.label, col_label, cy, size=17,
                  color=C.TEXT_PRIMARY, weight=2)
        before_s = "--" if row.before is None else f"{row.before:.1f}°"
        after_s = "--" if row.after is None else f"{row.after:.1f}°"
        draw_text(frame, before_s, col_before, cy, size=19,
                  color=C.TEXT_SECONDARY, weight=2)
        draw_text(frame, after_s, col_after, cy, size=19,
                  color=C.TEXT_PRIMARY, weight=2)
        if row.before is not None and row.after is not None:
            delta = row.after - row.before
            delta_color = C.SUCCESS if delta >= 0 else C.FAIL
            arrow = "+" if delta >= 0 else ""
            draw_text(frame, f"{arrow}{delta:.1f}°", col_delta, cy,
                      size=19, color=delta_color, weight=2)
        cy += 48

    if state.comparison_caption:
        draw_text(frame, state.comparison_caption, inner_x, py + ph - 70,
                  size=13, color=C.TEXT_SECONDARY)


# ---------------------------------------------------------------------------
# Camera-area overlays (live angle box, countdown, callout, robot status)
# ---------------------------------------------------------------------------

def draw_angle_overlay(frame, state: UIState) -> None:
    """Small floating angle readout in the bottom-right of the camera area.
    Shows: ARM ANGLE label, the big value, and a status pill (LIVE / HOLD /
    PASS / FAIL). Sized to read from across a room.
    """
    if not state.angles:
        return

    h, w = frame.shape[:2]
    cx, cy, cw, ch = _camera_bounds(w, h)
    height, width = frame.shape[:2]
    panel_width, panel_height = 700, 156
    x = max(width - panel_width - 24, 24)
    y = max(height - panel_height - 24, 24)

    reading = state.angles[0]
    status_color = {
        "live": C.ACCENT,
        "hold": C.WARN,
        "pass": C.SUCCESS,
        "fail": C.FAIL,
    }.get(reading.status, C.TEXT_SECONDARY)

    box_w = 280
    box_h = 150
    box_x = cx + cw - box_w - CAM_PAD
    box_y = cy + ch - box_h - CAM_PAD

    draw_panel(frame, box_x, box_y, box_w, box_h, fill=C.SURFACE_PANEL,
               alpha=0.97, border=status_color, border_thickness=2,
               accent_top=status_color, accent_thickness=5)

    pad = 20

    # Status pill in the top-right corner
    pill_text = reading.status.upper()
    pill_tw, _ = _measure_text(pill_text, 15, 2)
    pill_w = pill_tw + 24
    draw_pill(frame, pill_text,
              box_x + box_w - pill_w - pad, box_y + pad - 2,
              fg=C.SURFACE_PANEL, bg=status_color,
              pad_x=12, pad_y=5, size=15)

    # "ARM ANGLE" label above the number
    draw_kicker(frame, "Arm angle", box_x + pad, box_y + pad + 2,
                color=C.TEXT_SECONDARY, size=14)

    # The big number
    value_str = "--" if reading.value is None else f"{reading.value:.1f}°"
    draw_text(frame, value_str, box_x + pad, box_y + pad + 32,
              size=64, color=C.TEXT_PRIMARY, weight=2)


def draw_countdown(frame, state: UIState) -> None:
    """Countdown overlay centered on the camera area."""
    if state.countdown_seconds is None:
        return
    h, w = frame.shape[:2]
    cx, cy, cw, ch = _camera_bounds(w, h)

    box_w, box_h = 280, 280
    box_x = cx + (cw - box_w) // 2
    box_y = cy + (ch - box_h) // 2 - 40

    _alpha_rect(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                C.SURFACE_PANEL, 0.97)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                  C.ACCENT, 3, cv2.LINE_AA)

    if state.countdown_label:
        draw_kicker(frame, state.countdown_label, box_x + box_w // 2,
                    box_y + 18, color=C.ACCENT)

    seconds = max(0.0, state.countdown_seconds)
    big = f"{seconds:0.1f}"
    draw_text(frame, big, box_x + box_w // 2, box_y + 46, size=104,
              color=C.TEXT_PRIMARY, weight=2, align="center")

    ring_cx = box_x + box_w // 2
    ring_cy = box_y + 210
    radius = 50
    fraction = (seconds % 1.0) if seconds > 0 else 0.0
    cv2.circle(frame, (ring_cx, ring_cy), radius, C.SURFACE_HAIR, 3, cv2.LINE_AA)
    angle = int(360 * fraction)
    if angle > 0:
        cv2.ellipse(frame, (ring_cx, ring_cy), (radius, radius), -90, 0, angle,
                    C.ACCENT, 4, cv2.LINE_AA)


ROBOT_PHASE_LABELS = {
    "disconnected": "Robot offline",
    "at_home": "At home",
    "moving_to_start": "Moving to start",
    "at_start": "At start — awaiting auth",
    "executing": "Executing",
    "at_end": "At end — awaiting auth",
    "returning_home": "Returning home",
    "aborted": "Aborted",
}


def draw_big_callout(frame, state: UIState) -> None:
    """Rounded badge anchored top-left of the camera area (e.g. PEAK: 185°)."""
    if not state.big_callout:
        return
    _h, w = frame.shape[:2]
    cx, cy, _cw, _ch = _camera_bounds(w, frame.shape[0])

    text = state.big_callout
    size = 48
    tw, th = _measure_text(text, size, 2)
    pad_x, pad_y = 24, 16
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2

    # Top-left of the camera area, nudged down so it clears the dev overlay row.
    box_x = cx + 24
    box_y = cy + 40

    _rounded_alpha_rect(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                        C.SURFACE_PANEL, 0.97, radius=14)
    _rounded_stroke(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                    state.big_callout_color, thickness=3, radius=14)
    # Vertically center the text inside the box (PIL anchors at top-left).
    text_y = box_y + (box_h - th) // 2 - 4
    draw_text(frame, text, box_x + pad_x, text_y, size=size,
              color=state.big_callout_color, weight=2)


def draw_dev_overlay(frame, state: UIState) -> None:
    if not state.dev_overlay:
        return
    h, w = frame.shape[:2]
    cx, cy, cw, ch = _camera_bounds(w, h)
    draw_text(frame, state.dev_overlay, cx + CAM_PAD, cy + CAM_PAD,
              size=11, color=C.TEXT_DIM)


# ---------------------------------------------------------------------------
# Top-level render
# ---------------------------------------------------------------------------

def draw_fullscreen_results(frame, state: UIState) -> None:
    """Full-canvas presenter view: three big cards (BEFORE / AFTER / DELTA)
    sized for a projector or TV. Reads values from state.comparison[0]."""
    h, w = frame.shape[:2]
    scale = h / 1080.0
    accent = PHASE_ACCENTS.get(state.phase, C.SUCCESS)

    # Solid panel-pink background covering the whole canvas.
    cv2.rectangle(frame, (0, 0), (w, h), C.SURFACE_PANEL, -1)
    cv2.rectangle(frame, (0, 0), (w, max(6, int(10 * scale))), accent, -1)

    # ---- Header / subtitle ----
    title = state.hero_title or "Your Range of Motion Results"
    draw_text(frame, title, w // 2, int(80 * scale),
              size=int(72 * scale), color=C.TEXT_PRIMARY, weight=2, align="center")

    kicker = state.hero_kicker or ""
    if kicker:
        draw_text(frame, kicker, w // 2, int(40 * scale),
                  size=int(22 * scale), color=accent, weight=2, align="center")

    subtitle = state.comparison_caption or ""
    if subtitle:
        draw_text(frame, subtitle, w // 2, int(180 * scale),
                  size=int(28 * scale), color=C.TEXT_SECONDARY, align="center")

    # ---- Values ----
    if state.comparison:
        row = state.comparison[0]
        baseline = row.before
        assisted = row.after
        delta = (assisted - baseline) if (baseline is not None and assisted is not None) else None
        movement_label = row.label
    else:
        baseline = assisted = delta = None
        movement_label = ""

    def _fmt(v):
        return "—" if v is None else f"{v:.0f}°"

    def _fmt_delta(v):
        if v is None:
            return "—"
        return f"{'+' if v >= 0 else ''}{v:.0f}°"

    if delta is None:
        delta_color = C.TEXT_PRIMARY
        delta_caption = "—"
    elif delta > 0:
        delta_color = C.SUCCESS
        delta_caption = "improvement"
    elif delta < 0:
        delta_color = C.WARN
        delta_caption = "regression"
    else:
        delta_color = C.TEXT_PRIMARY
        delta_caption = "no change"

    cards = [
        ("BEFORE", _fmt(baseline),    "your own raise",     C.TEXT_PRIMARY),
        ("AFTER",  _fmt(assisted),    "with robot assist",  C.TEXT_PRIMARY),
        ("DELTA",  _fmt_delta(delta), delta_caption,        delta_color),
    ]

    card_w = int(500 * scale)
    card_h = int(540 * scale)
    gap    = int(50 * scale)
    total_w = card_w * 3 + gap * 2
    start_x = (w - total_w) // 2
    cards_y = int(260 * scale)
    radius = int(28 * scale)

    for i, (label, value_text, caption, value_color) in enumerate(cards):
        cx = start_x + i * (card_w + gap)
        cy = cards_y
        _rounded_filled(frame, (cx, cy), (cx + card_w, cy + card_h),
                        C.SURFACE_CARD, radius=radius)
        _rounded_stroke(frame, (cx, cy), (cx + card_w, cy + card_h),
                        C.SURFACE_HAIR, thickness=2, radius=radius)
        # Label (top of card)
        draw_text(frame, label, cx + card_w // 2, cy + int(40 * scale),
                  size=int(36 * scale), color=C.TEXT_DIM, weight=2, align="center")
        # Giant value (vertically centered-ish)
        draw_text(frame, value_text, cx + card_w // 2, cy + int(170 * scale),
                  size=int(180 * scale), color=value_color, weight=2, align="center")
        # Caption (near bottom)
        draw_text(frame, caption, cx + card_w // 2,
                  cy + card_h - int(70 * scale),
                  size=int(30 * scale), color=C.TEXT_SECONDARY, align="center")

    # ---- Movement-name footer + key hint ----
    if movement_label:
        draw_text(frame, movement_label.upper(),
                  w // 2, cards_y + card_h + int(40 * scale),
                  size=int(24 * scale), color=C.TEXT_DIM, weight=2, align="center")

    footer = state.footer_hint or "Press Space to restart   ·   q to quit"
    draw_text(frame, footer, w // 2, h - int(60 * scale),
              size=int(28 * scale), color=C.TEXT_DIM, align="center")


def render_ui(frame, state: UIState) -> None:
    """Draw the patient-facing UI on top of an existing frame."""
    if getattr(state, "show_fullscreen_results", False):
        with _text_batch(frame):
            draw_fullscreen_results(frame, state)
        return

    now = time.monotonic()
    # Sprite composites first so panels and text render cleanly above them.
    draw_arm_raise_animation(frame, state, now)
    with _text_batch(frame):
        draw_phase_header(frame, state)
        draw_right_panel(frame, state)
        draw_countdown(frame, state)
        draw_big_callout(frame, state)
        draw_angle_overlay(frame, state)
        draw_dev_overlay(frame, state)


# ---------------------------------------------------------------------------
# Backwards-compatible render_frame
# ---------------------------------------------------------------------------

def compose_camera_into_canvas(canvas, camera_frame) -> tuple[int, int, int, int]:
    """Composite the camera frame into the camera region of the canvas using a
    cover fit (scale to fill, crop overflow). Mutates canvas in place. Returns
    the camera region bounds (x, y, w, h)."""
    h, w = canvas.shape[:2]
    cam_x, cam_y, cam_w, cam_h = _camera_bounds(w, h)
    if camera_frame is None or cam_w <= 0 or cam_h <= 0:
        return cam_x, cam_y, cam_w, cam_h
    src_h, src_w = camera_frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return cam_x, cam_y, cam_w, cam_h
    scale = max(cam_w / src_w, cam_h / src_h)
    new_w = max(2, int(round(src_w * scale)))
    new_h = max(2, int(round(src_h * scale)))
    resized = cv2.resize(camera_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    crop_x = max(0, (new_w - cam_w) // 2)
    crop_y = max(0, (new_h - cam_h) // 2)
    cropped = resized[crop_y:crop_y + cam_h, crop_x:crop_x + cam_w]
    ch_actual, cw_actual = cropped.shape[:2]
    canvas[cam_y:cam_y + ch_actual, cam_x:cam_x + cw_actual] = cropped
    return cam_x, cam_y, cam_w, cam_h


def render_frame(
    frame,
    *,
    result=None,
    visibility_threshold: float = 0.5,
    angles: Optional[dict] = None,
    image_plane_angles: Optional[dict] = None,
    calibration=None,
    test_capture=None,
    rom=None,
    view_mode: str = "front",
    pose_count: int = 0,
    fps: float = 0.0,
    result_timestamp_ms: int = 0,
    now: float = 0.0,
    robot_state=None,
    ui_state: Optional[UIState] = None,
    render_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Build the full Stretch Goals UI canvas for one frame and return it.

    The caller should `cv2.imshow(window, render_frame(...))`. Pose skeleton is
    drawn on the raw camera frame before it's composited into the canvas, so
    landmarks track the camera image exactly.
    """
    if render_size is None:
        cam_h, cam_w = frame.shape[:2] if frame is not None else (720, 1280)
        render_h = max(720, cam_h)
        if cam_w > 0 and cam_h > 0:
            render_w = int(round(render_h * (cam_w / cam_h)))
        else:
            render_w = render_h * 16 // 9
    else:
        render_w, render_h = render_size
    if render_w % 2:
        render_w += 1
    if render_h % 2:
        render_h += 1

    canvas = _make_backdrop(render_w, render_h)

    if frame is not None and result is not None:
        draw_pose(frame, result, visibility_threshold)
    compose_camera_into_canvas(canvas, frame)

    if ui_state is None:
        ui_state = _legacy_state_to_ui(
            angles=angles or {},
            image_plane_angles=image_plane_angles or {},
            calibration=calibration,
            test_capture=test_capture,
            rom=rom,
            view_mode=view_mode,
            pose_count=pose_count,
            fps=fps,
            result_timestamp_ms=result_timestamp_ms,
            now=now,
            robot_state=robot_state,
        )

    render_ui(canvas, ui_state)
    return canvas


def _default_checklist() -> list:
    return [
        ChecklistItem("Get into ready position", "active"),
        ChecklistItem("3-point arm assessment", "pending"),
        ChecklistItem("Robot led stretch", "pending"),
    ]


def _legacy_state_to_ui(
    *, angles, image_plane_angles, calibration, test_capture, rom, view_mode,
    pose_count, fps, result_timestamp_ms, now, robot_state=None,
) -> UIState:
    state = UIState()
    state.dev_overlay = (
        f"dev   ·   view={view_mode}   ·   poses={pose_count}   ·   "
        f"fps={fps:4.1f}   ·   ts={result_timestamp_ms}ms"
    )

    # ---- Defaults: idle / waiting --------------------------------------
    state.phase = 0
    state.step_total = 3
    state.step_index = 0
    state.hero_kicker = "Let's get ready for your Range of Motion assessment"
    state.hero_title = "Sit down and face the camera"
    state.hero_bullets = [
        "Sit upright with both feet flat on the floor",
        "Rest your back on the chair",
        "Relax your arms loosely at your sides",
        "Face the camera so your full upper body is visible",
    ]
    state.checklist = _default_checklist()
    state.footer_hint = (
        "Press  c  calibrate   ·   t  test capture   ·   "
        "r  ROM sweep   ·   q  quit"
    )

    arm_angles_l = angles.get("L", {}) if angles else {}
    arm_angles_r = angles.get("R", {}) if angles else {}
    state.angles = [
        AngleReading("Flexion", "R", arm_angles_r.get("flexion"),
                     target_min=150, status="live"),
        AngleReading("Flexion", "L", arm_angles_l.get("flexion"),
                     target_min=150, status="live"),
    ]

    # ---- Workflow: calibration -----------------------------------------
    if calibration is not None and getattr(calibration, "active", False):
        step = calibration.step() if hasattr(calibration, "step") else None
        status_text = getattr(calibration, "status", "") or ""
        state.phase = 0
        state.hero_kicker = "Setup · Calibration"
        state.hero_title = (step.name.replace("_", " ").title()
                            if step else "Hold the calibration pose")
        state.hero_subtitle = step.instruction if step else "Stay still…"
        state.hero_bullets = None
        state.checklist = _default_checklist()
        state.checklist[0] = ChecklistItem("Calibrating cameras", "active",
                                           status_text)
        state.footer_hint = "Space captures now   ·   q quits"

    # ---- Workflow: ROM sweep -------------------------------------------
    if rom is not None and getattr(rom, "active", False):
        step = rom.step() if hasattr(rom, "step") else None
        state.phase = 1
        state.step_index = 1
        state.hero_kicker = "Phase 1 · ROM sweep"
        arm_label = f"{step.arm} arm" if step and hasattr(step, "arm") else "arm"
        state.hero_title = (f"Sweep your {arm_label} overhead"
                            if step else "ROM sweep")
        state.hero_subtitle = (step.instruction if step
                               else "Move slowly through your full range.")
        state.hero_bullets = None
        state.checklist = _default_checklist()
        state.checklist[0] = ChecklistItem("Get into ready position", "done")
        state.checklist[1] = ChecklistItem("3-point arm assessment", "active",
                                           "ROM sweep in progress")
        if getattr(rom, "recording", False):
            duration = getattr(rom, "duration_seconds", 0.0)
            started_at = getattr(rom, "started_at", now)
            remaining = max(duration - (now - started_at), 0.0)
            state.countdown_seconds = remaining
            state.countdown_label = "Recording sweep"
            state.footer_hint = "Sweep through full range   ·   Space stops early"
        else:
            state.footer_hint = "Space starts recording   ·   ← → switch arm"

    # ---- Workflow: diagnostic test capture -----------------------------
    if test_capture is not None and getattr(test_capture, "active", False):
        step = test_capture.step() if hasattr(test_capture, "step") else None
        state.phase = 1
        state.step_index = 1
        state.hero_kicker = "Phase 1 · Diagnostic capture"
        state.hero_title = step.name.title() if step else "Capture pose"
        state.hero_subtitle = step.instruction if step else ""
        state.hero_bullets = None
        state.big_callout = "HOLD"
        state.big_callout_color = C.WARN
        state.checklist = _default_checklist()
        state.checklist[0] = ChecklistItem("Get into ready position", "done")
        state.checklist[1] = ChecklistItem("3-point arm assessment", "active",
                                           "Diagnostic capture")
        state.footer_hint = "Hold steady   ·   Space captures now"

    # ---- Robot bridge overlay (phase 2) --------------------------------
    if robot_state is not None and getattr(robot_state, "connected", False):
        rp = getattr(robot_state, "phase", "disconnected")
        if rp in ("moving_to_start", "at_start", "executing",
                  "at_end", "returning_home"):
            state.phase = 2
            state.step_index = 2
            state.hero_kicker = "Phase 2 · Robot-led stretch"
            phase_titles = {
                "moving_to_start": "Robot is moving to start position",
                "at_start": "Hold the start position",
                "executing": "Robot is guiding your stretch",
                "at_end": "Final position reached",
                "returning_home": "Robot returning home",
            }
            state.hero_title = phase_titles.get(rp, ROBOT_PHASE_LABELS.get(rp, rp))
            state.hero_subtitle = None
            state.hero_bullets = None
            state.checklist = _default_checklist()
            state.checklist[0] = ChecklistItem("Get into ready position", "done")
            state.checklist[1] = ChecklistItem("3-point arm assessment", "done")
            state.checklist[2] = ChecklistItem("Robot led stretch", "active",
                                               ROBOT_PHASE_LABELS.get(rp, rp))
            detail = robot_state.step or robot_state.target or ""
            cap_name = robot_state.capture_name or "(no capture)"
            state.robot_state = (f"{cap_name}  ·  {ROBOT_PHASE_LABELS.get(rp, rp)}"
                                 + (f"  ·  {detail}" if detail else ""))
            state.footer_hint = "Robot is in control   ·   q quits"
        elif rp == "aborted":
            state.robot_state = "ROBOT ABORTED"
            state.robot_safety = robot_state.detail or "Safety stop triggered"
            state.footer_hint = "Restart the robot to retry"
        elif rp == "at_home":
            state.robot_state = (f"{robot_state.capture_name or '(no capture)'}"
                                 f"  ·  at home, ready to start")

    return state


# ---------------------------------------------------------------------------
# Demo / preview state library
# ---------------------------------------------------------------------------

def _demo_p0_setup() -> UIState:
    s = UIState()
    s.phase = 0
    s.step_total = 3
    s.step_index = 0
    s.hero_kicker = "Let's get ready for your Range of Motion assessment"
    s.hero_title = "Sit down and face the camera"
    s.hero_bullets = [
        "Sit upright with both feet flat on the floor",
        "Rest your back on the chair",
        "Relax your arms loosely at your sides",
        "Face the camera so your full upper body is visible",
    ]
    s.checklist = [
        ChecklistItem("Get into ready position", "active"),
        ChecklistItem("3-point arm assessment", "pending"),
        ChecklistItem("Robot led stretch", "pending"),
    ]
    s.angles = [
        AngleReading("Posture confidence", "—", 72.0, target_min=80, status="live"),
    ]
    s.footer_hint = ""
    return s


def _demo_flexion_exercise() -> UIState:
    """Panel 2: shoulder flexion exercise. The on-camera countdown should
    trigger once the model detects the user has held the same angle for
    three seconds (signaling they've reached their limit)."""
    s = UIState()
    s.phase = 1
    s.step_total = 3
    s.step_index = 1
    s.hero_kicker = "Shoulder Range of Motion assessment"
    s.hero_title = "Exercise 1: Flexion"
    s.hero_bullets = [
        "Slowly raise your arm in front of you and straight up.",
        "Keep going until you\u2019ve reached your limit.",
        "Hold that pose for 5 seconds.   5 4 3 2 1",
        "Done! Release arm back down your side.",
    ]
    s.checklist = [
        ChecklistItem("Get into ready position", "done"),
        ChecklistItem("3-point arm assessment", "active"),
        ChecklistItem("Robot led stretch", "pending"),
    ]
    s.angles = [
        AngleReading("Flexion", "R", 142.0, target_min=160, status="hold"),
    ]
    s.countdown_seconds = 5.0
    s.countdown_label = "Hold steady"
    s.footer_hint = ""
    return s


def _demo_p1_ready() -> UIState:
    s = UIState()
    s.phase = 1
    s.step_total = 5
    s.step_index = 0
    s.hero_kicker = "Phase 1   ·   Assessment"
    s.hero_title = "Sit tall and stay relaxed"
    s.hero_subtitle = (
        "Your posture looks good. We will now run three short movements. "
        "Each one takes about ten seconds."
    )
    s.checklist = [
        ChecklistItem("Ready position", "active", "Detecting posture…"),
        ChecklistItem("Flexion: arm forward and up", "pending"),
        ChecklistItem("Abduction: arm out to side", "pending"),
        ChecklistItem("Outward rotation", "pending"),
        ChecklistItem("Assessment results", "pending"),
    ]
    s.angles = [
        AngleReading("Posture confidence", "—", 88.0, target_min=80, status="pass"),
    ]
    return s


def _demo_p1_flexion() -> UIState:
    s = UIState()
    s.phase = 1
    s.step_total = 5
    s.step_index = 1
    s.hero_kicker = "Phase 1   ·   Movement 1 of 3"
    s.hero_title = "Raise your right arm straight forward and up"
    s.hero_subtitle = "Go as high as feels comfortable. Hold at your limit for five seconds."
    s.checklist = [
        ChecklistItem("Ready position", "done"),
        ChecklistItem("Flexion: arm forward and up", "active", "Recording right arm"),
        ChecklistItem("Abduction: arm out to side", "pending"),
        ChecklistItem("Outward rotation", "pending"),
        ChecklistItem("Assessment results", "pending"),
    ]
    s.angles = [
        AngleReading("Flexion", "R", 142.0, target_min=160, status="hold"),
        AngleReading("Flexion", "L", 38.0, status="live"),
    ]
    s.countdown_seconds = 3.4
    s.countdown_label = "Hold steady"
    return s


def _demo_p1_abduction() -> UIState:
    s = UIState()
    s.phase = 1
    s.step_total = 5
    s.step_index = 2
    s.hero_kicker = "Phase 1   ·   Movement 2 of 3"
    s.hero_title = "Raise your right arm out to the side"
    s.hero_subtitle = "Form a T shape if you can. Hold at your limit for five seconds."
    s.checklist = [
        ChecklistItem("Ready position", "done"),
        ChecklistItem("Flexion: arm forward and up", "done", "R 168°   ·   L 174°"),
        ChecklistItem("Abduction: arm out to side", "active"),
        ChecklistItem("Outward rotation", "pending"),
        ChecklistItem("Assessment results", "pending"),
    ]
    s.angles = [
        AngleReading("Abduction", "R", 102.0, target_min=150, status="live"),
    ]
    return s


def _demo_p1_rotation() -> UIState:
    s = UIState()
    s.phase = 1
    s.step_total = 5
    s.step_index = 3
    s.hero_kicker = "Phase 1   ·   Movement 3 of 3"
    s.hero_title = "Bend your elbow, then rotate your forearm outward"
    s.hero_subtitle = "Keep your elbow tucked to your side. Rotate until you reach your limit."
    s.checklist = [
        ChecklistItem("Ready position", "done"),
        ChecklistItem("Flexion: arm forward and up", "done", "R 168°   ·   L 174°"),
        ChecklistItem("Abduction: arm out to side", "done", "R 142°   ·   L 168°"),
        ChecklistItem("Outward rotation", "active"),
        ChecklistItem("Assessment results", "pending"),
    ]
    s.angles = [
        AngleReading("Outward rotation", "R", 48.0, target_min=70, status="live"),
    ]
    return s


def _demo_p1_results() -> UIState:
    s = UIState()
    s.phase = 1
    s.step_total = 5
    s.step_index = 4
    s.hero_kicker = "Phase 1   ·   Results"
    s.hero_title = "Your right arm shows limited range"
    s.hero_subtitle = (
        "Flexion is within range. Abduction and rotation fall below normal thresholds. "
        "The robot will run a corrective stretch next."
    )
    s.checklist = [
        ChecklistItem("Ready position", "done"),
        ChecklistItem("Flexion R", "done", "168°   ·   Pass"),
        ChecklistItem("Abduction R", "failed", "142°   ·   Limited"),
        ChecklistItem("Outward rotation R", "failed", "48°   ·   Limited"),
        ChecklistItem("Proceed to stretch", "active"),
    ]
    s.angles = [
        AngleReading("Abduction", "R", 142.0, target_min=150, status="fail"),
        AngleReading("Rotation", "R", 48.0, target_min=70, status="fail"),
    ]
    return s


def _demo_p2_position() -> UIState:
    s = UIState()
    s.phase = 2
    s.step_total = 5
    s.step_index = 0
    s.hero_kicker = "Phase 2   ·   Stretch setup"
    s.hero_title = "Stay seated. The robot is moving into position"
    s.hero_subtitle = "Do not move your arm. The robot is bringing the band toward your right hand."
    s.checklist = [
        ChecklistItem("Robot positioning", "active"),
        ChecklistItem("Grasp the handle", "pending"),
        ChecklistItem("Practice stop signal", "pending"),
        ChecklistItem("Overarm stretch", "pending"),
        ChecklistItem("Underarm stretch", "pending"),
    ]
    s.robot_state = "Approaching right hand   ·   12 cm"
    s.robot_safety = "Pull the handle twice at any moment to stop"
    s.angles = [
        AngleReading("Hand visibility", "R", 96.0, target_min=70, status="pass"),
    ]
    return s


def _demo_p2_grasp() -> UIState:
    s = UIState()
    s.phase = 2
    s.step_total = 5
    s.step_index = 1
    s.hero_kicker = "Phase 2   ·   Hand-off"
    s.hero_title = "Grasp the handle firmly"
    s.hero_subtitle = "Wrap your fingers around the handle. Keep your arm relaxed."
    s.checklist = [
        ChecklistItem("Robot positioning", "done"),
        ChecklistItem("Grasp the handle", "active"),
        ChecklistItem("Practice stop signal", "pending"),
        ChecklistItem("Overarm stretch", "pending"),
        ChecklistItem("Underarm stretch", "pending"),
    ]
    s.robot_state = "Holding at hand position   ·   waiting for grip"
    s.robot_safety = "Pull the handle twice at any moment to stop"
    s.big_callout = "GRASP"
    s.big_callout_color = C.ROBOT
    return s


def _demo_p2_trigger() -> UIState:
    s = UIState()
    s.phase = 2
    s.step_total = 5
    s.step_index = 2
    s.hero_kicker = "Phase 2   ·   Safety check"
    s.hero_title = "Practice the stop signal"
    s.hero_subtitle = (
        "Pull the handle twice quickly. This is how you tell the robot to stop at any time."
    )
    s.checklist = [
        ChecklistItem("Robot positioning", "done"),
        ChecklistItem("Grasp the handle", "done"),
        ChecklistItem("Practice stop signal", "active", "Waiting for double pull…"),
        ChecklistItem("Overarm stretch", "pending"),
        ChecklistItem("Underarm stretch", "pending"),
    ]
    s.robot_state = "Listening for double pull   ·   1 of 2 detected"
    s.robot_safety = "Two firm tugs in under one second"
    s.angles = [
        AngleReading("Band tension", "R", 22.0, target_min=15, target_max=80, status="live"),
    ]
    return s


def _demo_p2_overarm() -> UIState:
    s = UIState()
    s.phase = 2
    s.step_total = 5
    s.step_index = 3
    s.hero_kicker = "Phase 2   ·   Stretch 1 of 2"
    s.hero_title = "Overarm stretch in progress"
    s.hero_subtitle = (
        "The robot is lifting your arm overhead toward 180°. "
        "Breathe out. Pull the handle twice to stop."
    )
    s.checklist = [
        ChecklistItem("Robot positioning", "done"),
        ChecklistItem("Grasp the handle", "done"),
        ChecklistItem("Practice stop signal", "done"),
        ChecklistItem("Overarm stretch", "active", "Holding at max for 7s"),
        ChecklistItem("Underarm stretch", "pending"),
    ]
    s.robot_state = "Stretching   ·   holding at 162° (limit detected)"
    s.robot_safety = "Pull twice to stop   ·   Arm pressure within safe range"
    s.angles = [
        AngleReading("Flexion", "R", 162.0, target_min=170, status="hold"),
        AngleReading("Band tension", "R", 41.0, target_min=15, target_max=70, status="live"),
    ]
    s.countdown_seconds = 6.2
    s.countdown_label = "Hold stretch"
    return s


def _demo_p2_underarm() -> UIState:
    s = UIState()
    s.phase = 2
    s.step_total = 5
    s.step_index = 4
    s.hero_kicker = "Phase 2   ·   Stretch 2 of 2"
    s.hero_title = "Underarm stretch in progress"
    s.hero_subtitle = (
        "The robot is drawing your arm backward and down toward 0°. "
        "Keep your arm straight."
    )
    s.checklist = [
        ChecklistItem("Robot positioning", "done"),
        ChecklistItem("Grasp the handle", "done"),
        ChecklistItem("Practice stop signal", "done"),
        ChecklistItem("Overarm stretch", "done", "Held 162° for 7s"),
        ChecklistItem("Underarm stretch", "active"),
    ]
    s.robot_state = "Stretching   ·   moving toward 0°"
    s.robot_safety = "Keep arm straight   ·   Pull twice to stop"
    s.angles = [
        AngleReading("Flexion", "R", 22.0, target_min=0, target_max=10, status="hold"),
        AngleReading("Elbow angle", "R", 168.0, target_min=170, status="live"),
    ]
    s.big_callout = "STRAIGHTEN ARM"
    s.big_callout_color = C.WARN
    return s


def _demo_p3_reassess() -> UIState:
    s = UIState()
    s.phase = 3
    s.step_total = 4
    s.step_index = 1
    s.hero_kicker = "Phase 3   ·   Reassessment"
    s.hero_title = "Same three movements, one more time"
    s.hero_subtitle = "Let go of the handle. Repeat each movement so we can measure your new range."
    s.checklist = [
        ChecklistItem("Release the handle", "done"),
        ChecklistItem("Flexion: arm forward and up", "active"),
        ChecklistItem("Abduction: arm out to side", "pending"),
        ChecklistItem("Outward rotation", "pending"),
    ]
    s.angles = [
        AngleReading("Flexion", "R", 158.0, target_min=160, status="live"),
    ]
    return s


def _demo_p3_report() -> UIState:
    s = UIState()
    s.phase = 3
    s.step_total = 4
    s.step_index = 3
    s.hero_kicker = "Phase 3   ·   Report"
    s.hero_title = "Session complete"
    s.hero_subtitle = "Saving the report. You can take a screenshot or have it emailed to your provider."
    s.checklist = [
        ChecklistItem("Release the handle", "done"),
        ChecklistItem("Flexion: arm forward and up", "done"),
        ChecklistItem("Abduction: arm out to side", "done"),
        ChecklistItem("Outward rotation", "done"),
    ]
    s.angles = []
    s.comparison = [
        ComparisonRow("Flexion (R)", before=142.0, after=171.0, target=160.0),
        ComparisonRow("Abduction (R)", before=98.0, after=146.0, target=150.0),
        ComparisonRow("Outward rotation (R)", before=48.0, after=72.0, target=70.0),
    ]
    s.comparison_caption = "Saved to captures/ as PDF + JSON   ·   Ready to send to your provider"
    return s


DEMO_SCREENS: tuple[Callable[[], UIState], ...] = (
    _demo_p0_setup,
    _demo_flexion_exercise,
    _demo_p1_ready,
    _demo_p1_flexion,
    _demo_p1_abduction,
    _demo_p1_rotation,
    _demo_p1_results,
    _demo_p2_position,
    _demo_p2_grasp,
    _demo_p2_trigger,
    _demo_p2_overarm,
    _demo_p2_underarm,
    _demo_p3_reassess,
    _demo_p3_report,
)


# ---------------------------------------------------------------------------
# Standalone preview
# ---------------------------------------------------------------------------

def _make_backdrop(width: int = 1280, height: int = 720) -> np.ndarray:
    """Soft pink gradient with a faux silhouette in the LEFT half (camera area)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / height
        # BGR: gentle pink-to-pinker top-to-bottom gradient
        b = int(220 + 12 * (1 - t))
        g = int(228 + 14 * (1 - t))
        r = int(248 + 6 * (1 - t))
        img[y, :] = (b, g, r)
    # Camera area silhouette (only meaningful when webcam is off)
    cam_cx = width * 2 // 6  # rough center of the 2/3-wide camera area
    cv2.ellipse(img, (cam_cx, height // 2 + 60), (130, 200),
                0, 0, 360, (200, 215, 245), -1, cv2.LINE_AA)
    cv2.circle(img, (cam_cx, height // 2 - 120), 80, (200, 215, 245), -1, cv2.LINE_AA)
    return img


def _detect_screen_size() -> tuple[int, int]:
    """Best-effort screen-size detection. Falls back to 1920x1080."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.destroy()
        return sw, sh
    except Exception:
        return 1920, 1080


def _open_camera() -> Optional[cv2.VideoCapture]:
    """Open a camera, preferring the built-in laptop camera on macOS.

    macOS Continuity Camera can register an iPhone as an extra camera
    index. iPhone cameras typically report a much higher native resolution
    (>= 1920 wide) than a MacBook FaceTime HD camera (1280x720 typical).
    We probe each index, log what we find, and pick the most laptop-like
    one. If nothing matches, we just pick the lowest index that works.
    """

    def _try_open(idx: int, backend: int) -> tuple[Optional[cv2.VideoCapture], int, int]:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            return None, 0, 0
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(6):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap, frame.shape[1], frame.shape[0]
            time.sleep(0.08)
        cap.release()
        return None, 0, 0

    on_mac = platform.system() == "Darwin"
    backend = cv2.CAP_AVFOUNDATION if on_mac else cv2.CAP_ANY
    backend_name = "AVFoundation" if on_mac else "default"

    # Probe each index, record (idx, frame_w, frame_h), release the capture.
    probes: list[tuple[int, int, int]] = []
    for idx in (0, 1, 2):
        cap, fw, fh = _try_open(idx, backend)
        if cap is None:
            continue
        cap.release()
        probes.append((idx, fw, fh))
        likely_iphone = fw > 1920 or fh > 1440
        tag = " (likely iPhone)" if likely_iphone else ""
        print(f">>> camera probe: index={idx} backend={backend_name} {fw}x{fh}{tag}")

    if not probes:
        print(">>> no cameras available -- check macOS Camera permission")
        print(">>> System Settings -> Privacy & Security -> Camera")
        print(f">>>   {sys.executable}")
        return None

    # Prefer cameras whose native resolution looks like a laptop webcam.
    # Lower score wins; iPhone-likely cameras get bumped to the back.
    def score(probe: tuple[int, int, int]) -> tuple[int, int]:
        idx, fw, fh = probe
        likely_iphone = 1 if (fw > 1920 or fh > 1440) else 0
        return (likely_iphone, idx)

    probes.sort(key=score)
    chosen_idx, chosen_w, chosen_h = probes[0]
    print(f">>> using camera index {chosen_idx} ({chosen_w}x{chosen_h})")

    # Re-open the chosen camera for use.
    cap, _, _ = _try_open(chosen_idx, backend)
    return cap


def _preview_loop() -> None:
    print(
        "Stretch Goals \u00b7 UI preview\n"
        "  1..9, 0  \u2192 jump to screen 1..10\n"
        "  -, =     \u2192 previous / next screen\n"
        "  c        \u2192 toggle camera backdrop\n"
        "  q / Esc  \u2192 quit"
    )

    # Initial window covers ~92% of the screen, centered.
    screen_w, screen_h = _detect_screen_size()
    init_w = int(screen_w * 0.92)
    init_h = int(screen_h * 0.92)
    pos_x = (screen_w - init_w) // 2
    pos_y = (screen_h - init_h) // 2

    cap = _open_camera()
    use_camera = cap is not None
    idx = 0

    window = "Stretch Goals"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, init_w, init_h)
    cv2.moveWindow(window, pos_x, pos_y)

    # Cached render dimensions (rebuilt when the user resizes the window).
    cached_w, cached_h = 0, 0
    backdrop = None
    cam_x = cam_y = cam_w = cam_h = 0

    while True:
        # Detect current window size each frame so the layout responds when
        # the user drags the window edge.
        try:
            _wx, _wy, win_w, win_h = cv2.getWindowImageRect(window)
        except cv2.error:
            win_w, win_h = init_w, init_h
        if win_w <= 0 or win_h <= 0:
            win_w, win_h = init_w, init_h

        # Render at 720-line baseline matching the window aspect so the UI
        # never gets stretched. Rebuild the backdrop only when the size
        # actually changes.
        render_h = 720
        render_w = int(round(render_h * win_w / win_h))
        if render_w % 2:
            render_w += 1
        if (render_w, render_h) != (cached_w, cached_h):
            cached_w, cached_h = render_w, render_h
            backdrop = _make_backdrop(render_w, render_h)
            cam_x, cam_y, cam_w, cam_h = _camera_bounds(render_w, render_h)

        # Start each frame from the backdrop, then composite the camera
        # into the camera bounds preserving its native aspect ratio.
        canvas = backdrop.copy()
        if use_camera and cap is not None:
            ok, cam_frame = cap.read()
            if ok and cam_frame is not None:
                cam_frame = cv2.flip(cam_frame, 1)
                src_h, src_w = cam_frame.shape[:2]
                # Cover: scale uniformly so the camera fills the bounds and
                # crop whichever dimension overflows. No gaps top/bottom or
                # left/right -- the camera always fills its region.
                scale = max(cam_w / src_w, cam_h / src_h)
                new_w = max(2, int(round(src_w * scale)))
                new_h = max(2, int(round(src_h * scale)))
                cam_resized = cv2.resize(cam_frame, (new_w, new_h),
                                          interpolation=cv2.INTER_AREA)
                crop_x = max(0, (new_w - cam_w) // 2)
                crop_y = max(0, (new_h - cam_h) // 2)
                cam_cropped = cam_resized[crop_y:crop_y + cam_h,
                                          crop_x:crop_x + cam_w]
                ch_actual, cw_actual = cam_cropped.shape[:2]
                canvas[cam_y:cam_y + ch_actual,
                       cam_x:cam_x + cw_actual] = cam_cropped

        state = DEMO_SCREENS[idx]()
        if state.countdown_seconds is not None:
            state.countdown_seconds = max(
                0.0, state.countdown_seconds - (time.time() % 1) * 0.1
            )

        render_ui(canvas, state)

        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("="), ord("+"), 83):
            idx = (idx + 1) % len(DEMO_SCREENS)
        elif key in (ord("-"), ord("_"), 81):
            idx = (idx - 1) % len(DEMO_SCREENS)
        elif ord("1") <= key <= ord("9"):
            target = key - ord("1")
            if target < len(DEMO_SCREENS):
                idx = target
        elif key == ord("0"):
            if len(DEMO_SCREENS) >= 10:
                idx = 9
        elif key == ord("c"):
            if cap is not None:
                use_camera = not use_camera
            else:
                cap = _open_camera()
                use_camera = cap is not None

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    _preview_loop()
