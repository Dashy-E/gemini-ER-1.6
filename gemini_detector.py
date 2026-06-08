"""
Async Gemini-based object detector.

Runs Gemini inference in a background thread so the camera feed never stalls.
The main thread calls submit_frame() to queue a frame and get_detections() to
read the most recent results — both are non-blocking.

Requires: GOOGLE_API_KEY environment variable (or pass api_key directly).
Model default: gemini-2.0-flash  (change MODEL_ID to any vision-capable Gemini model)
"""

import os
import cv2
import json
import base64
import threading
import time

import google.generativeai as genai

MODEL_ID = "gemini-2.0-flash"

# Prompt instructs Gemini to return normalized bounding boxes [ymin, xmin, ymax, xmax]
# on a 0–1000 scale — same convention Gemini uses natively.
_DETECTION_PROMPT = (
    "Detect every distinct object visible in this image. "
    "Return ONLY a JSON array — no markdown, no explanation, no extra text. "
    "Each element: {\"label\": \"<object name>\", \"box\": [ymin, xmin, ymax, xmax]} "
    "where all four coordinates are integers normalized to the range 0-1000. "
    "Example: [{\"label\": \"bottle\", \"box\": [200, 150, 700, 450]}]"
)


def _parse_gemini_response(text: str, frame_w: int, frame_h: int) -> list[dict]:
    """Parse Gemini JSON response into pixel-space detection dicts."""
    text = text.strip()
    # Strip markdown code fences if present
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

        # Clamp to valid range
        ymin = max(0, min(1000, ymin))
        xmin = max(0, min(1000, xmin))
        ymax = max(0, min(1000, ymax))
        xmax = max(0, min(1000, xmax))

        # Convert normalized 0-1000 coords to pixel coords
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
        detector.submit_frame(frame)        # non-blocking
        dets = detector.get_detections()    # non-blocking, returns last result
        fps  = detector.inference_fps       # approx Gemini inference rate
    """

    def __init__(self, api_key: str | None = None, model_id: str = MODEL_ID):
        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Export it before running: set GOOGLE_API_KEY=your_key"
            )
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model_id)

        self._lock = threading.Lock()
        self._pending_frame = None          # frame waiting to be processed
        self._pending_shape = None
        self._latest_detections: list[dict] = []
        self._frame_event = threading.Event()

        self._running = False
        self._thread: threading.Thread | None = None

        self._inference_count = 0
        self._last_fps_time = time.time()
        self.inference_fps = 0.0
        self.last_error: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="GeminiDetector")
        self._thread.start()

    def stop(self):
        self._running = False
        self._frame_event.set()  # unblock waiting thread

    def submit_frame(self, frame):
        """Queue a frame for Gemini inference (replaces any unprocessed frame)."""
        with self._lock:
            self._pending_frame = frame.copy()
            self._pending_shape = frame.shape
        self._frame_event.set()

    def get_detections(self, target_class: str | None = None) -> list[dict]:
        """Return the latest detection list, optionally filtered by label."""
        with self._lock:
            dets = list(self._latest_detections)
        if target_class:
            dets = [d for d in dets if d["label"] == target_class.lower()]
        return dets

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _loop(self):
        while self._running:
            # Block until a new frame is queued
            self._frame_event.wait()
            self._frame_event.clear()

            if not self._running:
                break

            frame = None
            shape = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    shape = self._pending_shape
                    self._pending_frame = None
                    self._pending_shape = None

            if frame is None:
                continue

            h, w = shape[:2]
            try:
                b64 = self._encode(frame)
                response = self._model.generate_content([
                    _DETECTION_PROMPT,
                    {"mime_type": "image/jpeg", "data": b64},
                ])
                dets = _parse_gemini_response(response.text, w, h)
                with self._lock:
                    self._latest_detections = dets
                    self.last_error = ""

                # Track inference FPS
                self._inference_count += 1
                now = time.time()
                elapsed = now - self._last_fps_time
                if elapsed >= 2.0:
                    self.inference_fps = self._inference_count / elapsed
                    self._inference_count = 0
                    self._last_fps_time = now

            except Exception as exc:
                err = str(exc)
                with self._lock:
                    self.last_error = err
                print(f"[GeminiDetector] Error: {err}")

    @staticmethod
    def _encode(frame) -> str:
        """JPEG-encode a frame and return base64 string."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]
        _, buf = cv2.imencode(".jpg", frame, encode_params)
        return base64.b64encode(buf).decode("utf-8")
