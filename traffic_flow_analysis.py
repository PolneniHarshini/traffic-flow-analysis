import cv2
import numpy as np
import torch
import pandas as pd
import time
from collections import defaultdict
from ultralytics import YOLO
from sort import Sort  # sort.py must be in same folder
import yt_dlp

# ------------------- SETTINGS -------------------
VIDEO_URL = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
OUTPUT_CSV = "vehicle_counts.csv"

# Lane boundaries (vertical lines for 3 lanes)
LANE_COORDS = [
    ((100, 0), (100, 720)),   # Lane 1 boundary line
    ((400, 0), (400, 720)),   # Lane 2 boundary line
    ((700, 0), (700, 720))    # Lane 3 boundary line
]

# ------------------- DOWNLOAD YOUTUBE STREAM -------------------
print("[INFO] Fetching video stream using yt-dlp...")
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(VIDEO_URL, download=False)
    video_url = info['url']

cap = cv2.VideoCapture(video_url)

# ------------------- SAVE PROCESSED VIDEO -------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# ------------------- LOAD YOLO MODEL -------------------
print("[INFO] Loading YOLO model...")
model = YOLO('yolov8n.pt')  # lightweight COCO model

# ------------------- SORT TRACKER -------------------
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

# ------------------- VARIABLES -------------------
vehicle_ids = set()
lane_counts = defaultdict(int)
records = []
frame_count = 0

def get_lane(x_center):
    """Determine lane number based on object center X."""
    if x_center < LANE_COORDS[0][0][0]:
        return 1
    elif x_center < LANE_COORDS[1][0][0]:
        return 2
    else:
        return 3

# ------------------- PROCESS VIDEO -------------------
print("[INFO] Processing video...")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    results = model(frame, stream=True)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] in ['car', 'truck', 'bus', 'motorbike']:
                x1, y1, x2, y2 = box.xyxy[0]
                detections.append([x1.item(), y1.item(), x2.item(), y2.item(), box.conf[0].item()])

    detections_np = np.array(detections) if detections else np.empty((0, 5))
    tracked_objects = tracker.update(detections_np)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        center_x = int((x1 + x2) / 2)
        lane_num = get_lane(center_x)

        if obj_id not in vehicle_ids:
            vehicle_ids.add(obj_id)
            lane_counts[lane_num] += 1
            timestamp = round(time.time() - start_time, 2)
            records.append([int(obj_id), lane_num, frame_count, timestamp])

        # Draw bounding boxes and labels
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{int(obj_id)} L:{lane_num}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw lane boundaries
    for line in LANE_COORDS:
        cv2.line(frame, line[0], line[1], (255, 0, 0), 2)

    # Show counts
    cv2.putText(frame, f"Lane 1: {lane_counts[1]}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Lane 2: {lane_counts[2]}", (300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Lane 3: {lane_counts[3]}", (550, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Write to output video
    out.write(frame)

    # Display live window (press q to quit)
    cv2.imshow("Traffic Flow Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# ------------------- SAVE CSV -------------------
df = pd.DataFrame(records, columns=["Vehicle_ID", "Lane", "Frame", "Timestamp"])
df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Results saved to {OUTPUT_CSV}")
print(f"[INFO] Processed video saved as output.mp4")
print(f"[INFO] Final Counts: {dict(lane_counts)}")
