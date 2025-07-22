# Real-Time Vehicle Detection, Tracking, Counting, and Speed Estimation Using YOLO and Supervision

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from src import Annotator, ViewTransformer, SpeedEstimator
from config import (
    IN_VIDEO_PATH,
    OUT_VIDEO_PATH,
    YOLO_MODEL_PATH,
    LINE_Y,
    SOURCE_POINTS,
    TARGET_POINTS,
    WINDOW_NAME,
)


## 1. Initialization and Setup
# -----------------------------

# Load YOLO model
model = YOLO(YOLO_MODEL_PATH)

# Get video info
video_info = sv.VideoInfo.from_video_path(IN_VIDEO_PATH)
print(video_info)

# Initialize ByteTrack tracker
tracker = sv.ByteTrack(frame_rate=video_info.fps)

# Define Line Zone for counting
offset = 55
start, end = sv.Point(offset, LINE_Y), sv.Point(video_info.width - offset, LINE_Y)
line_zone = sv.LineZone(start, end, minimum_crossing_threshold=1)

# Define Perspective Transform source & target points
SOURCE = np.array(SOURCE_POINTS)
TARGET = np.array(TARGET_POINTS)

# Initialize ViewTransformer
view_transformer = ViewTransformer(SOURCE, TARGET)

# Initialize SpeedEstimator
speed_estimator = SpeedEstimator(fps=video_info.fps, view_transformer=view_transformer)

# Initialize Annotator
annotator = Annotator(
    resolution_wh=video_info.resolution_wh,
    box_annotator=True,
    label_annotator=True,
    line_annotator=True,
    multi_class_line_annotator=True,
    trace_annotator=True,
    polygon_zone=SOURCE,
)

# Initialize Window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, video_info.width, video_info.height)


## 2. Main Processing Loop
# -------------------------
frame_generator = sv.get_video_frames_generator(IN_VIDEO_PATH)

# Video writer
with sv.VideoSink(OUT_VIDEO_PATH, video_info) as sink:
    for frame in frame_generator:
        # YOLO Detection
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Apply Tracker
        detections = tracker.update_with_detections(detections)

        # Apply Line Zone Counting
        line_zone.trigger(detections)

        # Estimate Speed
        detections = speed_estimator.update(detections)

        # Create Labels: ID + Class  + Speed
        labels = []
        for tracker_id, class_name, speed in zip(
            detections.tracker_id,
            detections.data["class_name"],
            detections.data["speed"],
        ):
            text = f"{class_name} #{tracker_id}"
            if speed != 0:
                text = f"{class_name} {speed}km/h"
            labels.append(text)

        # Annotate Frame
        annotated_frame = annotator.annotate(
            frame,
            detections,
            labels=labels,
            line_zones=[line_zone],
            multi_class_zones=[line_zone],
        )

        # Write frame to output video
        sink.write_frame(frame=annotated_frame)

        # Show real-time display
        cv2.imshow(WINDOW_NAME, annotated_frame)

        # Check if 'q' pressed or window closed
        if (
            cv2.waitKey(1) & 0xFF == ord("q")
            or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        ):
            break


cv2.destroyAllWindows()

# Final Output
print("Processing complete.")
print(f"Processed video saved at: {OUT_VIDEO_PATH}")
print(f"Total vehicles counted: {line_zone.in_count + line_zone.out_count}", end=" | ")
print(f"(In: {line_zone.in_count}, Out: {line_zone.out_count})")
print(f"Model used: {YOLO_MODEL_PATH}")
