# Vision-Based Robotic Manipulation System

A real-time computer vision system that uses **Google Gemini** to detect and identify objects via a laptop camera, draw bounding boxes, and track object trajectories over time.

---

## Features

- **Real-time object detection** — Gemini vision API identifies every object in the camera feed with labeled bounding boxes
- **Object trajectory tracking** — records the centroid path of the tracked object across frames
- **Async inference** — camera feed runs at full FPS; Gemini processes frames in a background thread so the display never stutters
- **Interactive target selection** — filter tracking to a specific object class (bottle, cup, person, etc.) at any time
- **World coordinate mapping** — converts pixel centroids to robot-centric world coordinates

---

## Project Structure

```
.
├── main.py               # Entry point — camera loop, key handling, orchestration
├── gemini_detector.py    # Async Gemini vision detector (background thread)
├── trajectory.py         # Per-object centroid history tracker
├── visualization.py      # Bounding boxes, trajectory overlay, HUD
├── camera.py             # Thin OpenCV VideoCapture wrapper
├── mapping.py            # Pixel-to-world coordinate conversion
└── requirements.txt      # Python dependencies
```

---

## Requirements

- Python 3.11+
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))
- A laptop or USB camera

---

## Setup

**1. Install dependencies**

```powershell
pip install -r requirements.txt
```

**2. Set your Gemini API key**

```powershell
# PowerShell (Windows)
$env:GOOGLE_API_KEY = "your_api_key_here"

# Or add it permanently via System Environment Variables
```

**3. Run**

```powershell
python main.py
```

You will be prompted to enter a target object class to track (e.g. `bottle`, `cup`, `person`). Leave blank to track all detected objects.

---

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Change target object class |
| `c` | Clear trajectory history |

---

## Trajectory Format

The tracker outputs trajectories in this format (most recent point first, label = age index):

```json
[
  {"point": [550, 610], "label": "0"},
  {"point": [500, 600], "label": "1"},
  {"point": [450, 590], "label": "2"},
  ...
  {"point": [100, 300], "label": "15"}
]
```

Printed to the console every 5 new points. The trajectory is also rendered live on the camera feed as a fading colored line.

---

## Configuration

| Location | Setting | Default | Description |
|----------|---------|---------|-------------|
| `main.py` | `TRAJECTORY_MAX_POINTS` | `50` | Max centroid history length |
| `gemini_detector.py` | `MODEL_ID` | `gemini-2.0-flash` | Gemini model to use |
| `mapping.py` | `scale` | `0.05` | Pixel-to-world unit scale factor |

To use a different Gemini model, change `MODEL_ID` in `gemini_detector.py`:

```python
MODEL_ID = "gemini-1.5-flash"   # or gemini-2.5-flash, etc.
```

---

## How It Works

```
Camera (OpenCV)
     │
     ▼
 Main Loop (full FPS)
     │
     ├──► submit_frame() ──► [Background Thread] ──► Gemini API
     │                                                    │
     │◄─────────────── latest detections ◄───────────────┘
     │
     ├──► TrajectoryTracker.update(label, cx, cy)
     │
     ├──► draw_bounding_boxes(frame, detections)
     ├──► draw_trajectory(frame, trajectory)
     └──► draw_hud(frame, ...)
```

The background thread always processes the **most recently submitted frame**, dropping any unprocessed ones in between. This keeps Gemini latency invisible to the camera display.

---

## Coordinate System

World coordinates are centered at the bottom-middle of the frame (nominal robot position):

```
world_x = (pixel_x - frame_width  / 2) * scale
world_y = (frame_height - pixel_y) * scale
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Camera capture and all drawing |
| `google-generativeai` | Gemini vision API client |
| `numpy` | Array operations (transitive via OpenCV) |
