"""
Async Gemini-based object detector.

Runs Gemini inference in a background thread so the camera feed never stalls.
The main thread calls submit_frame() to queue a frame and get_detections() to
read the most recent results — both are non-blocking.

Requires: GOOGLE_API_KEY environment variable (or pass api_key directly).
Model default: gemini-2.5-flash (swap to confirmed Gemini Robotics-ER 1.6 ID once available)
"""

import os
import cv2
import json
import threading
import time

from google import genai
from google.genai import types

MODEL_ID = "gemini-2.5-flash"


def _build_prompt(target_class: str | None) -> str:
    if target_class:
        subject = f'only "{target_class}" objects (ignore everything else)'
    else:
        subject = "every distinct object visible in this image"
    return (
        f"Detect {subject}. "
        "Return ONLY a JSON array — no markdown, no explanation, no extra text. "
        "Each element: {\"label\": \"<object name>\", \"box\": [ymin, xmin, ymax, xmax]} "
        "where all four coordinates are integers normalized to the range 0-1000. "
        "Return an empty array [] if the object is not present. "
        "Example: [{\"label\": \"bottle\", \"box\": [200, 150, 700, 450]}]"
    )


def _parse_gemini_response(text: str, frame_w: int, frame_h: int) -> list[dict]:
    """Parse Gemini JSON response into pixel-space detection dicts."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            try:
                data = json.loads(stripped)
                break
            except json.JSONDecodeError:
                continue
        else:
            return []
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

    if not isinstance(data, list):
        return []

    detections = []
    for item in data:
        label = str(item.get("label", "object")).lower().strip()
        box = item.get("box", [])
        if len(box) != 4:
            continue
        try:
            ymin, xmin, ymax, xmax = [int(v) for v in box]
        except (TypeError, ValueError):
            continue

        ymin = max(0, min(1000, ymin))
        xmin = max(0, min(1000, xmin))
        ymax = max(0, min(1000, ymax))
        xmax = max(0, min(1000, xmax))

        x1 = int(xmin * frame_w / 1000)
        y1 = int(ymin * frame_h / 1000)
        x2 = int(xmax * frame_w / 1000)
        y2 = int(ymax * frame_h / 1000)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        detections.append({
            "label": label,
            "cx": cx,
            "cy": cy,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    return detections


class GeminiDetector:
    """
    Thread-safe async object detector backed by Gemini vision API.

    Usage:
        detector = GeminiDetector()
        detector.start()

        # In main loop:
        detector.submit_frame(frame)             # non-blocking
        dets = detector.get_detections()         # non-blocking, returns last result
        fps  = detector.inference_fps            # approx Gemini inference rate
        stale = detector.is_stale(timeout=1.0)  # True if no response in >timeout seconds
    """

    def __init__(self, api_key: str | None = None, model_id: str = MODEL_ID,
                 target_class: str | None = None):
        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Export it before running: set GOOGLE_API_KEY=your_key"
            )
        self._client = genai.Client(api_key=key)
        self._model_id = model_id
        self._target_class: str | None = target_class

        self._lock = threading.Lock()
        self._pending_frame = None
        self._pending_shape = None
        self._pending_target: str | None = target_class
        self._latest_detections: list[dict] = []
        self._frame_event = threading.Event()

        self._running = False
        self._thread: threading.Thread | None = None

        self._inference_count = 0
        self._last_fps_time = time.time()
        self.inference_fps = 0.0
        self.last_error: str = ""
        self._last_seen: float = 0.0   # timestamp of last successful inference

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="GeminiDetector")
        self._thread.start()

    def stop(self):
        self._running = False
        self._frame_event.set()

    @property
    def target_class(self) -> str | None:
        with self._lock:
            return self._target_class

    @target_class.setter
    def target_class(self, value: str | None):
        with self._lock:
            self._target_class = value

    def submit_frame(self, frame):
        """Queue a frame for Gemini inference (replaces any unprocessed frame)."""
        with self._lock:
            self._pending_frame = frame.copy()
            self._pending_shape = frame.shape
            self._pending_target = self._target_class
        self._frame_event.set()

    def get_detections(self, target_class: str | None = None) -> list[dict]:
        """Return the latest detection list, optionally filtered by label."""
        with self._lock:
            dets = list(self._latest_detections)
        if target_class:
            dets = [d for d in dets if d["label"] == target_class.lower()]
        return dets

    def is_stale(self, timeout: float = 1.0) -> bool:
        """
        Return True if Gemini has not produced a successful inference
        within `timeout` seconds. Signals the robot controller that
        position data should not be trusted for motion commands.
        """
        with self._lock:
            last = self._last_seen
        return last == 0.0 or (time.time() - last) > timeout

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _loop(self):
        while self._running:
            self._frame_event.wait()
            self._frame_event.clear()

            if not self._running:
                break

            frame = None
            shape = None
            snap_target = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    shape = self._pending_shape
                    snap_target = self._pending_target
                    self._pending_frame = None
                    self._pending_shape = None

            if frame is None:
                continue

            h, w = shape[:2]
            prompt = _build_prompt(snap_target)
            try:
                img_bytes = self._encode(frame)
                response = self._client.models.generate_content(
                    model=self._model_id,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    ],
                )
                dets = _parse_gemini_response(response.text, w, h)

                now = time.time()
                with self._lock:
                    self._latest_detections = dets
                    self.last_error = ""
                    self._last_seen = now
                    self._inference_count += 1
                    elapsed = now - self._last_fps_time
                    if elapsed >= 2.0:
                        self.inference_fps = self._inference_count / elapsed
                        self._inference_count = 0
                        self._last_fps_time = now

            except Exception as exc:
                err = str(exc)
                with self._lock:
                    self.last_error = err
                import logging
                logging.getLogger("gemini-er").error(f"[GeminiDetector] {err}")

    @staticmethod
    def _encode(frame) -> bytes:
        """JPEG-encode a frame at high quality to preserve fine detail."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        _, buf = cv2.imencode(".jpg", frame, encode_params)
        return buf.tobytes()
