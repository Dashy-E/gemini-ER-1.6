import cv2
import logging
import tkinter as tk
import json
from PIL import Image, ImageTk
from camera import get_frame, verify_resolution
from gemini_detector import GeminiDetector
from trajectory import TrajectoryTracker
from visualization import draw_bounding_boxes, draw_trajectory, draw_hud, _color_for
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("gemini-er")

FRAME_W = 640
FRAME_H = 480
TRAJECTORY_MAX_POINTS = 50


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
        log.error("Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    verify_resolution(cap, FRAME_W, FRAME_H)

    # Gemini detector (background thread)
    detector = GeminiDetector()
    detector.start()
    log.info(f"Detector started (model: {detector._model_id})")

    # Trajectory tracker — knows frame size for internal pixel→world mapping
    tracker = TrajectoryTracker(
        max_points=TRAJECTORY_MAX_POINTS,
        frame_w=FRAME_W,
        frame_h=FRAME_H,
    )

    target_class = prompt_target()
    log.info(f"Targeting: {target_class or 'ALL objects'}")
    log.info("Controls: q = quit | r = change target | c = clear trajectory")

    root = tk.Tk()
    root.title("Robotic Manipulation System — Gemini Vision")
    root.resizable(False, False)
    label = tk.Label(root)
    label.pack()

    running = {"value": True}
    nonlocal_target = {"value": target_class}

    def on_key(event):
        ch = event.char.lower()
        if ch == "q":
            running["value"] = False
            root.destroy()
        elif ch == "r":
            new_target = prompt_target()
            nonlocal_target["value"] = new_target
            tracker.clear()
            log.info(f"Targeting: {new_target or 'ALL objects'}")
        elif ch == "c":
            tracker.clear()
            log.info("Trajectory cleared.")

    root.bind("<Key>", on_key)

    stale_logged = {"value": False}   # suppress repeated stale warnings

    def update():
        if not running["value"]:
            return

        target = nonlocal_target["value"]

        ret, frame = get_frame(cap)
        if not ret:
            log.error("Frame capture failed.")
            running["value"] = False
            root.destroy()
            return

        detector.submit_frame(frame)

        # Stale threshold: 3 s gives Gemini time to respond under normal API load
        stale = detector.is_stale(timeout=3.0)

        all_detections = detector.get_detections()
        target_detections = detector.get_detections(target)

        # Determine primary object
        primary = None
        if target_detections:
            primary = target_detections[0]
        elif not target and all_detections:
            primary = all_detections[0]

        # Update tracker only when detections are fresh
        if not stale and primary:
            tracker.update(primary["label"], primary["cx"], primary["cy"])
            stale_logged["value"] = False
        elif stale and not stale_logged["value"]:
            log.warning("Gemini detections are stale — tracker paused.")
            stale_logged["value"] = True

        # Always draw boxes from whatever detections we have
        draw_bounding_boxes(frame, all_detections)

        world_pos = None
        velocity = (0.0, 0.0)

        if primary:
            primary_label = primary["label"]
            traj = tracker.get(primary_label)
            if traj:
                draw_trajectory(frame, traj, color=_color_for(primary_label))
                world_path = tracker.get_world_path(primary_label)
                if world_path:
                    world_pos = world_path[0]
                velocity = tracker.get_velocity(primary_label)

                log.debug(
                    f"['{primary_label}'] {len(traj)} pts | "
                    f"world ({world_pos[0]:.3f}, {world_pos[1]:.3f}) m | "
                    f"vel vx={velocity[0]:.3f} vy={velocity[1]:.3f} m/s"
                )

        draw_hud(
            frame,
            target,
            inference_fps=detector.inference_fps,
            num_detections=len(all_detections),
            world_pos=world_pos,
            velocity=velocity,
            is_stale=stale,
            error=detector.last_error,
        )

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        root.after(1, update)

    root.after(0, update)
    root.mainloop()

    detector.stop()
    cap.release()


if __name__ == "__main__":
    main()
