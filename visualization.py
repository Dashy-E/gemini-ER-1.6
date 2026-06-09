"""
Visualization helpers: bounding boxes, trajectories, HUD overlay.
"""

import math
import cv2
from mapping import pixel_to_world, WORLD_SCALE

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

def draw_trajectory(frame, trajectory: list[dict], color: tuple | None = None,
                    start_origin: tuple | None = None) -> None:
    """
    Draw a fading trajectory trail on frame in-place.

    Each entry: {"point": [cx, cy], "world": [wx, wy], "label": "0", "t": float}
    The newest point (label "0") is drawn brightest; older points fade out.
    World coordinates are annotated at the current (newest) position.

    start_origin: if given, a faded segment is drawn from it to the oldest
                  trail point so the path is anchored to the camera base.
    """
    if len(trajectory) < 1:
        return

    n = len(trajectory)
    base_color = color if color else (0, 200, 255)

    # Anchor segment: camera base → oldest trail point (faintest)
    if start_origin is not None:
        oldest_pt = tuple(trajectory[-1]["point"])
        anchor_alpha = max(0.12, 1.0 - ((n - 1) / max(n, 1)))
        ac = tuple(int(ch * anchor_alpha) for ch in base_color)
        cv2.line(frame, start_origin, oldest_pt, ac, 1, cv2.LINE_AA)
        # Small ring at the camera-base origin
        cv2.circle(frame, start_origin, 7, base_color, 2, cv2.LINE_AA)
        cv2.circle(frame, start_origin, 3, base_color, -1, cv2.LINE_AA)

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
# Approach vector  (camera-base → target)
# ------------------------------------------------------------------

def _draw_dashed_line(frame, pt1: tuple, pt2: tuple,
                      color: tuple, thickness: int = 2, dash_len: int = 12) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    pos, draw = 0.0, True
    while pos < dist:
        seg = min(dash_len, dist - pos)
        sx = int(x1 + dx * pos)
        sy = int(y1 + dy * pos)
        ex = int(x1 + dx * (pos + seg))
        ey = int(y1 + dy * (pos + seg))
        if draw:
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        pos += dash_len
        draw = not draw


def draw_approach_vector(frame, primary_det: dict,
                         world_scale: float = WORLD_SCALE) -> None:
    """
    Draw a dashed guide line from the camera base (centre-bottom of frame)
    to the ground contact point (bottom-centre of the target bounding box).

    Annotates:
      • Planar world distance  (metres)
      • Lateral offset X       (metres, + = right)
      • Depth Y                (metres ahead of camera base)
      • Bearing angle          (degrees, 0 = straight ahead)
    """
    h, w = frame.shape[:2]

    # Robot/camera base sits at centre-bottom of the frame
    origin_px = (w // 2, h - 1)

    # Use bottom-centre of bbox as the object's ground contact point
    gx = (primary_det["x1"] + primary_det["x2"]) // 2
    gy = primary_det["y2"]
    target_px = (gx, gy)

    # World coordinates (frame centre = origin; Y positive upward)
    twx, twy = pixel_to_world(gx, gy, w, h, world_scale)
    owx, owy = pixel_to_world(origin_px[0], origin_px[1], w, h, world_scale)

    lateral   = twx - owx                         # + = right of robot
    depth_fwd = twy - owy                          # + = ahead, – = behind
    planar    = math.hypot(lateral, depth_fwd)

    # Bearing: 0° = straight ahead (up in pixel space)
    dx_px = target_px[0] - origin_px[0]
    dy_px = target_px[1] - origin_px[1]
    bearing = math.degrees(math.atan2(dx_px, -dy_px))

    color = (0, 230, 140)   # bright teal-green

    # Dashed line + arrowhead
    _draw_dashed_line(frame, origin_px, target_px, color, thickness=2, dash_len=14)
    if math.hypot(dx_px, dy_px) > 20:
        cv2.arrowedLine(frame, origin_px, target_px, color, 2,
                        cv2.LINE_AA, tipLength=0.08)

    # Origin marker (camera base)
    cv2.circle(frame, origin_px, 9, color, 2, cv2.LINE_AA)
    cv2.circle(frame, origin_px, 3, color, -1, cv2.LINE_AA)

    # Target ground point marker
    cv2.circle(frame, target_px, 5, color, -1, cv2.LINE_AA)

    # Annotation block near midpoint of the line
    mid_x = (origin_px[0] + target_px[0]) // 2 + 8
    mid_y = (origin_px[1] + target_px[1]) // 2

    lines = [
        f"dist  {planar:.2f} m",
        f"lat   {lateral:+.2f} m",
        f"depth {depth_fwd:.2f} m",
        f"bear  {bearing:+.1f} deg",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (mid_x, mid_y + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


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
