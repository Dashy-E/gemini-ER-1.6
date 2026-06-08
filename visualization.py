"""
Visualization helpers: bounding boxes, trajectories, HUD overlay.
"""

import cv2
import numpy as np

# Color palette — one distinct BGR color per label (cycles if >12 labels)
_PALETTE = [
    (0,   220, 255),   # yellow
    (0,   165, 255),   # orange
    (255, 100,   0),   # blue
    (0,   255, 128),   # green
    (180,   0, 255),   # purple
    (0,   255, 255),   # cyan-yellow
    (255,   0, 128),   # pink-blue
    (128, 255,   0),   # lime
    (255,   0,   0),   # pure blue
    (0,     0, 255),   # pure red
    (0,   255,   0),   # pure green
    (200, 200,   0),   # teal-ish
]
_label_color_cache: dict[str, tuple] = {}


def _color_for(label: str) -> tuple:
    if label not in _label_color_cache:
        idx = len(_label_color_cache) % len(_PALETTE)
        _label_color_cache[label] = _PALETTE[idx]
    return _label_color_cache[label]


# ------------------------------------------------------------------
# Bounding boxes
# ------------------------------------------------------------------

def draw_bounding_boxes(frame, detections: list[dict]) -> None:
    """
    Draw a labeled bounding box for each detection in-place.

    Each detection dict:  {"label": str, "x1", "y1", "x2", "y2", "cx", "cy"}
    """
    for det in detections:
        label = det["label"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        color = _color_for(label)

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        text = label
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, label_y - th - baseline - 2),
                      (x1 + tw + 4, label_y + baseline), color, -1)
        cv2.putText(frame, text, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Centroid dot
        cv2.circle(frame, (det["cx"], det["cy"]), 4, color, -1)


# ------------------------------------------------------------------
# Trajectory
# ------------------------------------------------------------------

def draw_trajectory(frame, trajectory: list[dict], color: tuple | None = None) -> None:
    """
    Draw a trajectory (list of {"point": [cx,cy], "label": "0"}) on frame in-place.

    Oldest points are faint blue; newest are bright based on the provided color
    (defaults to white-to-color gradient).
    """
    if len(trajectory) < 1:
        return

    n = len(trajectory)
    base_color = color if color else (0, 200, 255)

    # Draw line segments from newest to oldest with fading alpha
    for i in range(len(trajectory) - 1):
        p1 = tuple(trajectory[i]["point"])
        p2 = tuple(trajectory[i + 1]["point"])
        alpha = 1.0 - (i / n)          # 1.0 at newest, near 0 at oldest
        c = tuple(int(ch * alpha) for ch in base_color)
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, p1, p2, c, thickness, cv2.LINE_AA)

    # Draw dots at each trajectory point
    for entry in trajectory:
        pt = tuple(entry["point"])
        idx = int(entry["label"])
        alpha = 1.0 - (idx / n)
        c = tuple(int(ch * alpha) for ch in base_color)
        radius = max(2, int(5 * alpha))
        cv2.circle(frame, pt, radius, c, -1, cv2.LINE_AA)

    # Highlight the current (newest) point
    newest = tuple(trajectory[0]["point"])
    cv2.circle(frame, newest, 6, base_color, 2, cv2.LINE_AA)

    # Label the current position
    label_text = f"[{newest[0]}, {newest[1]}]"
    cv2.putText(frame, label_text,
                (newest[0] + 8, newest[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, base_color, 1, cv2.LINE_AA)


# ------------------------------------------------------------------
# Robot-to-target line (legacy draw, kept for compatibility)
# ------------------------------------------------------------------

def draw(frame, robot_pos, obj_pos, target_label=""):
    rx, ry = robot_pos
    ox, oy = obj_pos

    cv2.circle(frame, (rx, ry), 10, (255, 0, 0), -1)
    cv2.circle(frame, (ox, oy), 5, (0, 0, 255), -1)
    cv2.line(frame, (rx, ry), (ox, oy), (0, 255, 255), 2)

    if target_label:
        cv2.putText(frame, f"Target: {target_label}", (ox + 8, oy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


# ------------------------------------------------------------------
# HUD overlay
# ------------------------------------------------------------------

def draw_hud(frame, target_class: str | None, inference_fps: float = 0.0,
             num_detections: int = 0, error: str = "") -> None:
    h, w = frame.shape[:2]

    # Status line
    tracking_text = f"Tracking: {target_class}" if target_class else "Tracking: ALL objects"
    cv2.putText(frame, tracking_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    # Inference FPS and object count
    stats = f"Gemini {inference_fps:.1f} inf/s  |  {num_detections} object(s)"
    cv2.putText(frame, stats, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA)

    # Error (red, bottom-left)
    if error:
        short_err = error[:80] + ("…" if len(error) > 80 else "")
        cv2.putText(frame, f"ERR: {short_err}", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 80, 255), 1, cv2.LINE_AA)

    # Controls
    cv2.putText(frame, "q: quit  |  r: retarget  |  c: clear trajectory",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
