"""
Per-object trajectory tracker with integrated world-coordinate mapping.

Public trajectory format (what the robot arm consumes):
    [
        {"point": [cx, cy], "label": "0"},   <- most recent
        {"point": [cx, cy], "label": "1"},
        ...
    ]

World coordinates and timestamps are stored internally and exposed via
separate methods (get_world_path, get_velocity) so the primary output
stays clean for the robot controller.
"""

import time
from collections import deque
from mapping import pixel_to_world


class TrajectoryTracker:
    """
    Tracks centroid history for multiple object labels independently.

    Args:
        max_points:  Rolling history length per object (default 50).
        frame_w:     Camera frame width in pixels.
        frame_h:     Camera frame height in pixels.
        world_scale: Metres per pixel at the operating distance.
    """

    def __init__(self, max_points: int = 50,
                 frame_w: int = 640, frame_h: int = 480,
                 world_scale: float = 0.05):
        self._max_points = max_points
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._world_scale = world_scale
        # label (str) -> deque of internal point dicts, newest first
        self._histories: dict[str, deque] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, label: str, cx: int, cy: int):
        """Append a new centroid for this label."""
        wx, wy = pixel_to_world(cx, cy, self._frame_w, self._frame_h, self._world_scale)
        key = label.lower()
        if key not in self._histories:
            self._histories[key] = deque(maxlen=self._max_points)
        self._histories[key].appendleft({
            "pixel": [cx, cy],
            "world": [round(wx, 4), round(wy, 4)],
            "t": time.time(),
        })

    def clear(self, label: str | None = None):
        if label is None:
            self._histories.clear()
        else:
            self._histories.pop(label.lower(), None)

    # ------------------------------------------------------------------
    # Public trajectory output (robot arm format)
    # ------------------------------------------------------------------

    def get(self, label: str) -> list[dict]:
        """
        Return the trajectory in robot-arm format — pure pixel points, no extras.

            [{"point": [cx, cy], "label": "0"}, ...]

        Label "0" is always the most recent position.
        """
        key = label.lower()
        history = self._histories.get(key, deque())
        return [
            {"point": entry["pixel"], "label": str(i)}
            for i, entry in enumerate(history)
        ]

    def get_all(self) -> dict[str, list[dict]]:
        """Return robot-arm format trajectories for every tracked label."""
        return {label: self.get(label) for label in self._histories}

    # ------------------------------------------------------------------
    # World-coordinate accessors (separate from robot format)
    # ------------------------------------------------------------------

    def get_world_path(self, label: str) -> list[list[float]]:
        """
        Return world-coordinate path for one label, newest first.
        Each entry: [wx, wy] in metres (or configured world_scale units).
        """
        key = label.lower()
        return [e["world"] for e in self._histories.get(key, deque())]

    def get_velocity(self, label: str, window: int = 5) -> tuple[float, float]:
        """
        Estimate instantaneous velocity in world units/second using a
        moving average over the last `window` point pairs to reduce jitter.
        Returns (vx, vy); (0.0, 0.0) if insufficient history.
        """
        key = label.lower()
        history = list(self._histories.get(key, deque()))
        if len(history) < 2:
            return (0.0, 0.0)

        pairs = min(window, len(history) - 1)
        vx_sum = vy_sum = 0.0
        valid = 0
        for i in range(pairs):
            p_new = history[i]
            p_old = history[i + 1]
            dt = p_new["t"] - p_old["t"]
            if dt <= 0:
                continue
            vx_sum += (p_new["world"][0] - p_old["world"][0]) / dt
            vy_sum += (p_new["world"][1] - p_old["world"][1]) / dt
            valid += 1

        if valid == 0:
            return (0.0, 0.0)
        return (round(vx_sum / valid, 3), round(vy_sum / valid, 3))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def tracked_labels(self) -> list[str]:
        return list(self._histories.keys())
