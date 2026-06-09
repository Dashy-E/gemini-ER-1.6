"""
Visualization helpers: bounding boxes, trajectories, HUD overlay.
"""

import cv2

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
    for det in detections:
        label = det["label"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        color = _color_for(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = label
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, label_y - th - baseline - 2),
                      (x1 + tw + 4, label_y + baseline), color, -1)
        cv2.putText(frame, text, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.circle(frame, (det["cx"], det["cy"]), 4, color, -1)


# ------------------------------------------------------------------
# Trajectory
# ------------------------------------------------------------------

def draw_trajectory(frame, trajectory: list[dict], color: tuple | None = None) -> None:
    """
    Draw a fading trajectory trail on frame in-place.

    Each entry: {"point": [cx, cy], "world": [wx, wy], "label": "0", "t": float}
    The newest point (label "0") is drawn brightest; older points fade out.
    World coordinates are annotated at the current (newest) position.
    """
    if len(trajectory) < 1:
        return

    n = len(trajectory)
    base_color = color if color else (0, 200, 255)

    # Fading line segments, newest → oldest
    for i in range(n - 1):
        p1 = tuple(trajectory[i]["point"])
        p2 = tuple(trajectory[i + 1]["point"])
        alpha = max(0.15, 1.0 - (i / max(n, 1)))
        c = tuple(int(ch * alpha) for ch in base_color)
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, p1, p2, c, thickness, cv2.LINE_AA)

    # Dots at each point
    for entry in trajectory:
        pt = tuple(entry["point"])
        idx = int(entry["label"])
        alpha = max(0.15, 1.0 - (idx / max(n, 1)))
        c = tuple(int(ch * alpha) for ch in base_color)
        radius = max(2, int(5 * alpha))
        cv2.circle(frame, pt, radius, c, -1, cv2.LINE_AA)

    # Highlight and annotate the current (newest) point
    newest_pt = tuple(trajectory[0]["point"])
    world = trajectory[0].get("world", [0.0, 0.0])
    cv2.circle(frame, newest_pt, 7, base_color, 2, cv2.LINE_AA)

    # Pixel position
    pixel_text = f"px [{newest_pt[0]}, {newest_pt[1]}]"
    cv2.putText(frame, pixel_text,
                (newest_pt[0] + 10, newest_pt[1] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, base_color, 1, cv2.LINE_AA)

    # World position
    world_text = f"world ({world[0]:.3f}, {world[1]:.3f}) m"
    cv2.putText(frame, world_text,
                (newest_pt[0] + 10, newest_pt[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, base_color, 1, cv2.LINE_AA)


# ------------------------------------------------------------------
# HUD overlay
# ------------------------------------------------------------------

def draw_hud(frame, target_class: str | None,
             inference_fps: float = 0.0,
             num_detections: int = 0,
             world_pos: list[float] | None = None,
             velocity: tuple[float, float] = (0.0, 0.0),
             is_stale: bool = False,
             error: str = "") -> None:
    h, w = frame.shape[:2]

    # Line 1 — tracking target
    tracking_text = f"Tracking: {target_class}" if target_class else "Tracking: ALL objects"
    cv2.putText(frame, tracking_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    # Line 2 — inference stats + stale warning
    stale_tag = "  !! STALE !!" if is_stale else ""
    stats = f"Gemini {inference_fps:.1f} inf/s  |  {num_detections} object(s){stale_tag}"
    stats_color = (0, 80, 255) if is_stale else (180, 255, 180)
    cv2.putText(frame, stats, (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, stats_color, 1, cv2.LINE_AA)

    # Line 3 — world position of primary target
    if world_pos is not None:
        pos_text = f"World pos: ({world_pos[0]:.3f}, {world_pos[1]:.3f}) m"
        cv2.putText(frame, pos_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1, cv2.LINE_AA)

        # Line 4 — velocity
        vx, vy = velocity
        speed = (vx ** 2 + vy ** 2) ** 0.5
        vel_text = f"Velocity:  vx={vx:.3f}  vy={vy:.3f}  |{speed:.3f}| m/s"
        cv2.putText(frame, vel_text, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1, cv2.LINE_AA)

    # Error line (red, bottom-left)
    if error:
        short_err = error[:80] + ("…" if len(error) > 80 else "")
        cv2.putText(frame, f"ERR: {short_err}", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 80, 255), 1, cv2.LINE_AA)

    # Controls reminder
    cv2.putText(frame, "q: quit  |  r: retarget  |  c: clear trajectory",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
