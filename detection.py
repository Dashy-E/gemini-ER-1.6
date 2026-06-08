from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect(frame, target_class=None):
    results = model(frame, verbose=False)
    detections = []

    for r in results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id].lower()

            if target_class and label != target_class.lower():
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append((cx, cy, label))

    return detections