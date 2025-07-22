# Configuration File

# Define Video Paths
IN_VIDEO_PATH = "./data/vehicles.mp4"
OUT_VIDEO_PATH = "./data/vehicles_output.mp4"

# YOLO Model Path
# To switch to a custom model, update the path below
# YOLO_MODEL_PATH = "./models/yolov8n.pt"
YOLO_MODEL_PATH = "./models/VisDrone_YOLO_x2.pt"

# Line Zone for counting vehicles (in pixels)
LINE_Y = 480

# Perspective Transform Points
# SOURCE_POINTS: Points in the original image
# TARGET_POINTS: Corresponding points in the top-down view (real-world mapping)
SOURCE_POINTS = [[450, 300], [860, 300], [1900, 720], [-660, 720]]
WIDTH, HEIGHT = 25, 100  # Dimensions in meters
TARGET_POINTS = [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]

# Display Window Settings
WINDOW_NAME = "Detection + Tracking + Counting + Speed Estimation"
