import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Config value
video_path = "data_ext/chinhkinh_nohelmet.MOV"
conf_threshold = 0.7

# Initialize DeepSort
tracker = DeepSort(max_age=5)

# Initialize YOLOv9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights="weights/model_yolov9-c3_detect_all/weights/best.pt", device=device, fuse=True)
model = AutoShape(model)

# Load class names
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size=(len(class_names), 3))

cap = cv2.VideoCapture(video_path)

# Read frames from video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects with YOLOv9
    results = model(frame)

    # Prepare list for tracking
    detections = []

    # Process detections
    for detect in results.pred[0]:
        label, confidence, bbox = detect[5], detect[4], detect[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if confidence < conf_threshold:
            continue

        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Update and get tracking information
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw bounding boxes and labels for tracked objects
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            label = "{}-{}".format(class_names[class_id], track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the frame
    resized_frame = cv2.resize(frame, (680, 1020))
    cv2.imshow("Window Tracking Result", resized_frame)

    # Exit
    if cv2.waitKey(25) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
