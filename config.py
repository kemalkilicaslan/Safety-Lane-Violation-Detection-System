# Safety Lane Violation Detection — Configuration

# Region of Interest (ROI) polygon — list of [x, y] points
ROI_COORDINATES = [[770, 280], [1130, 280], [1500, 900], [100, 900]]

# Counting line — two [x, y] endpoints
LINE_COORDINATES = [[375, 650], [615, 650]]

# Vehicle classes to detect (COCO class IDs) and their display names
VEHICLE_CLASSES = [2, 3, 5, 7]
VEHICLE_TYPES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Number of past positions kept per tracked object
TRACK_HISTORY_LENGTH = 30

# Model and video files
MODEL_PATH = "yolo12x.pt"
INPUT_VIDEO = "Traffic-Flow.mp4"                      # input video file name
OUTPUT_VIDEO = "Safety-Lane-Violation-Detection.mp4"  # output video file name
