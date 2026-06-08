import cv2
from camera import get_frame
from gemini_detector import GeminiDetector
from trajectory import TrajectoryTracker
from mapping import pixel_to_world
from visualization import draw_bounding_boxes, draw_trajectory, draw_hud, _color_for

# ── Configuration ────────────────────────────────────────────────────────────
TRAJECTORY_MAX_POINTS = 50   # how many centroid history points to keep
# Gemini is called on every available frame from the background thread.
# The main display loop always runs at full camera FPS.
# ─────────────────────────────────────────────────────────────────────────────


def prompt_target():
    val = input(
        "Enter target object to track trajectory for "
        "(e.g. bottle, cup, person) [leave blank for all]: "
    ).strip()
    return val.lower() if val else None


def main():
    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Gemini detector (background thread)
    detector = GeminiDetector()
    detector.start()
    print(f"[Gemini] Detector started (model: {detector._model.model_name})")

    # Trajectory tracker
    tracker = TrajectoryTracker(max_points=TRAJECTORY_MAX_POINTS)

    # Target class
    target_class = prompt_target()
    print(f"Targeting: {target_class or 'ALL objects'}")
    print("Controls: q = quit | r = change target | c = clear trajectory")

    while True:
        ret, frame = get_frame(cap)
        if not ret:
            print("ERROR: Frame capture failed.")
            break

        h, w = frame.shape[:2]

        # ── Submit frame to Gemini (non-blocking) ──────────────────────────
        # We always submit; the detector thread picks up the latest and drops
        # any unprocessed frame — this keeps latency minimal.
        detector.submit_frame(frame)

        # ── Retrieve latest detections ─────────────────────────────────────
        all_detections = detector.get_detections()
        target_detections = detector.get_detections(target_class)

        # ── Update trajectory for the primary tracked object ───────────────
        # If a target class is set, track that; otherwise track the first object.
        primary_label = None
        if target_detections:
            primary = target_detections[0]
            primary_label = primary["label"]
            tracker.update(primary_label, primary["cx"], primary["cy"])
        elif not target_class and all_detections:
            primary = all_detections[0]
            primary_label = primary["label"]
            tracker.update(primary_label, primary["cx"], primary["cy"])

        # ── Draw bounding boxes for ALL detections ─────────────────────────
        draw_bounding_boxes(frame, all_detections)

        # ── Draw trajectory for primary tracked object ─────────────────────
        if primary_label:
            traj = tracker.get(primary_label)
            traj_color = _color_for(primary_label)
            draw_trajectory(frame, traj, color=traj_color)

            # Print trajectory to console (only when it changes)
            if traj and len(traj) % 5 == 1:  # every 5 new points
                print(f"\n[Trajectory for '{primary_label}'] (last {len(traj)} pts):")
                import json
                print(json.dumps(traj[:10], separators=(",", ":")))

            # Print world coordinates for the primary target
            wx, wy = pixel_to_world(primary["cx"], primary["cy"], w, h)
            if len(traj) % 5 == 1:
                print(f"  World coords: ({wx:.3f}, {wy:.3f})")

        # ── HUD overlay ────────────────────────────────────────────────────
        draw_hud(
            frame,
            target_class,
            inference_fps=detector.inference_fps,
            num_detections=len(all_detections),
            error=detector.last_error,
        )

        cv2.imshow("Robotic Manipulation System — Gemini Vision", frame)

        # ── Key handling ───────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            cv2.destroyAllWindows()
            target_class = prompt_target()
            print(f"Targeting: {target_class or 'ALL objects'}")
        elif key == ord("c"):
            tracker.clear()
            print("[Trajectory] Cleared.")

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
