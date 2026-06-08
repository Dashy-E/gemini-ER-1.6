"""
Per-object trajectory tracker.

Stores the last N centroid positions for each tracked label and exposes them
in the format the visualizer and console expect:

    [
      {"point": [cx, cy], "label": "0"},   <- most recent
      {"point": [cx, cy], "label": "1"},
      ...
    ]

Label values are sequential integers starting at "0" for the newest point,
matching the example format requested.
"""

from collections import deque


class TrajectoryTracker:
    """
    Tracks centroid history for multiple object labels independently.

    Args:
        max_points: Maximum history length per object (default 50).
    """

    def __init__(self, max_points: int = 50):
        self._max_points = max_points
        # Map of label -> deque of (cx, cy) tuples, newest first
        self._histories: dict[str, deque] = {}

    def update(self, label: str, cx: int, cy: int):
        """Append a new centroid for this label."""
        key = label.lower()
        if key not in self._histories:
            self._histories[key] = deque(maxlen=self._max_points)
        self._histories[key].appendleft((cx, cy))

    def get(self, label: str) -> list[dict]:
        """
        Return trajectory for one label in the expected format:
            [{"point": [cx, cy], "label": "0"}, ...]
        Most recent point has label "0".
        """
        key = label.lower()
        history = self._histories.get(key, deque())
        return [
            {"point": list(pt), "label": str(i)}
            for i, pt in enumerate(history)
        ]

    def get_all(self) -> dict[str, list[dict]]:
        """Return trajectories for every tracked label."""
        return {label: self.get(label) for label in self._histories}

    def clear(self, label: str | None = None):
        """Clear history for one label (or all if label is None)."""
        if label is None:
            self._histories.clear()
        else:
            self._histories.pop(label.lower(), None)

    @property
    def tracked_labels(self) -> list[str]:
        return list(self._histories.keys())
