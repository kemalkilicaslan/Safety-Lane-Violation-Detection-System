# Safety Lane Violation Detection

# Import the necessary libraries
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import config

# Define coordinates for ROI and counting line (loaded from config.py)
ROI_COORDINATES = np.array(config.ROI_COORDINATES, dtype=np.int32)
LINE_COORDINATES = np.array(config.LINE_COORDINATES, dtype=np.int32)

# Define allowed vehicle classes and their names (loaded from config.py)
vehicles = set(config.VEHICLE_CLASSES)
vehicle_types = config.VEHICLE_TYPES

# Object tracking and counting variables
track_history = defaultdict(lambda: deque(maxlen=config.TRACK_HISTORY_LENGTH))
object_counts = defaultdict(int)
crossed_objects = set()
all_objects_in_roi = set()

# Function to check if a point is inside the ROI
def is_point_in_roi(point, roi_coordinates):
    return cv2.pointPolygonTest(roi_coordinates, point, False) >= 0

# Function to calculate intersection point of two lines
def line_intersection(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    
    denominator = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    if denominator == 0:
        return None
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator
    
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (int(x), int(y))
    
    return None

# Function to check if an object has crossed the counting line
def counting_line(track_id, current_position, line_coordinates):
    if track_id in track_history and len(track_history[track_id]) >= 2:
        prev_position = track_history[track_id][-2]
        object_path = [prev_position, current_position]
        intersection = line_intersection(object_path, line_coordinates)
        
        if intersection:
            # Convert to 3D vectors by adding a z-coordinate of 0
            line_vec = np.array([line_coordinates[1][0] - line_coordinates[0][0], 
                               line_coordinates[1][1] - line_coordinates[0][1], 0])
            path_vec = np.array([current_position[0] - prev_position[0], 
                               current_position[1] - prev_position[1], 0])
            cross_product = np.cross(line_vec, path_vec)
            # The cross product will be in the z-direction
            direction = "right" if cross_product[2] > 0 else "left"
            return True, direction, intersection
    
    return False, None, None

# Function to dashboard
def dashboard(frame, counts, all_objects_count, crossed_objects_count):
    panel_x = 35
    panel_y = 35
    panel_width = 430
    panel_height = 390
    
    # Create semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw panel border
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (255, 255, 255), 2)
    
    # Main Title
    cv2.putText(frame, "TRAFFIC FLOW ANALYSIS", (panel_x + 20, panel_y + 30), 
               cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    
    # Draw separator line
    cv2.line(frame, (panel_x + 10, panel_y + 45), 
            (panel_x + panel_width - 10, panel_y + 45), 
            (255, 255, 255), 1)
    
    # Calculate ratio
    if all_objects_count > 0:
        ratio = (crossed_objects_count / all_objects_count) * 100
        ratio_text = f"{ratio:.1f}%"
    else:
        ratio_text = "0.0%"
    
    # Draw statistics
    cv2.putText(frame, f"Total Vehicles: {all_objects_count}", (panel_x + 30, panel_y + 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Violations: {crossed_objects_count}", (panel_x + 30, panel_y + 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw pie chart if there are objects
    if all_objects_count > 0:
        pie_chart_center = (panel_x + panel_width - 80, panel_y + 100)
        pie_chart_radius = 35
        
        # Calculate angles for pie chart
        crossed_angle = int(360 * (crossed_objects_count / all_objects_count))
        not_crossed_angle = 360 - crossed_angle
        
        # Draw pie chart
        if crossed_angle > 0:
            cv2.ellipse(frame, pie_chart_center, (pie_chart_radius, pie_chart_radius), 
                       0, 0, crossed_angle, (0, 0, 255), -1)
        
        if not_crossed_angle > 0:
            cv2.ellipse(frame, pie_chart_center, (pie_chart_radius, pie_chart_radius), 
                       0, crossed_angle, 360, (255, 50, 50), -1)
        
        # Draw pie chart border
        cv2.circle(frame, pie_chart_center, pie_chart_radius, (255, 255, 255), 0)
        
        # Draw legend
        cv2.rectangle(frame, (panel_x + 30, panel_y + 120), 
                     (panel_x + 45, panel_y + 130), (0, 0, 255), -1)
        cv2.putText(frame, "Violations", (panel_x + 50, panel_y + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.rectangle(frame, (panel_x + 30, panel_y + 140), 
                     (panel_x + 45, panel_y + 150), (255, 50, 50), -1)
        cv2.putText(frame, "No Violations", (panel_x + 50, panel_y + 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Violation Ratio
    cv2.putText(frame, f"Violation Ratio: {ratio_text}", (panel_x + 30, panel_y + 190), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw separator line
    cv2.line(frame, (panel_x + 10, panel_y + 210), 
            (panel_x + panel_width - 10, panel_y + 210), 
            (255, 255, 255), 1)
    
    # Section 2: Vehicle Counts by Type
    cv2.putText(frame, "Safety Lane Violation Detection", (panel_x + 20, panel_y + 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    y_offset = 270
    total_count = 0
    
    for class_id in sorted(vehicles):
        class_name = vehicle_types[class_id]
        count = counts.get(class_name, 0)
        total_count += count
        
        # Draw class name and count
        cv2.putText(frame, f"{class_name}:", (panel_x + 30, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{count}", (panel_x + panel_width - 50, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 22
    
    # Draw total count
    cv2.putText(frame, "Total Violations:", (panel_x + 30, panel_y + y_offset + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"{total_count}", (panel_x + panel_width - 50, panel_y + y_offset + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

# Load YOLO model (path loaded from config.py)
model = YOLO(config.MODEL_PATH)

# Open video (paths loaded from config.py)
video_capture = cv2.VideoCapture(config.INPUT_VIDEO)  # write the name of the video file here
output_file = config.OUTPUT_VIDEO  # extension the name and extension of the video file to be recorded

# Video features
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Create the VideoWriter object to save the video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Create mask for ROI
roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.fillPoly(roi_mask, [ROI_COORDINATES], 255)

while True:
    ret, frame = video_capture.read()  # read video frame

    if not ret:
        break

    # Run YOLO inference with tracking
    results = model.track(frame, persist=True, verbose=False)

    # Initialize annotated frame
    annotated_frame = frame.copy()

    # Draw ROI with 50% transparency green fill
    roi_overlay = annotated_frame.copy()
    cv2.fillPoly(roi_overlay, [ROI_COORDINATES], (0, 255, 0))
    cv2.addWeighted(roi_overlay, 0.15, annotated_frame, 0.5, 0, annotated_frame)

    # Draw ROI border and counting line
    cv2.polylines(annotated_frame, [ROI_COORDINATES], True, (0, 255, 0), 0)
    cv2.line(annotated_frame, tuple(LINE_COORDINATES[0]), tuple(LINE_COORDINATES[1]), (0, 0, 255), 5)

    # Process detections
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, class_id, confidence in zip(boxes, track_ids, class_ids, confidences):
            # Filter only allowed classes (2, 3, 5, 7)
            if class_id not in vehicles:
                continue

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box)

            # Calculate center point of object
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_point = (center_x, center_y)

            # Only process objects inside ROI
            if not is_point_in_roi(center_point, ROI_COORDINATES):
                continue

            # Track all objects in ROI
            if track_id not in all_objects_in_roi:
                all_objects_in_roi.add(track_id)

            # Update track history
            track_history[track_id].append(center_point)

            # Check if object has crossed the line
            crossed, direction, intersection_point = counting_line(track_id, center_point, LINE_COORDINATES)

            if crossed and track_id not in crossed_objects:
                crossed_objects.add(track_id)
                class_name = vehicle_types[class_id]
                object_counts[class_name] += 1

            # Determine bounding box color: RED if crossed, BLUE if not crossed yet
            box_color = (0, 0, 255) if track_id in crossed_objects else (255, 0, 0)

            # If the object has crossed the line, fill the bounding box with 20% transparent red
            if track_id in crossed_objects:
                # Create overlay for transparent fill
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Fill with red
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)  # 20% transparency

            # Draw bounding box with appropriate color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 5)

    # Dashboard
    annotated_frame = dashboard(annotated_frame, object_counts, len(all_objects_in_roi), len(crossed_objects))

    # Write the drawn frame to the video file to be saved
    output_video.write(annotated_frame)

    # Show video with safety lane violation detection
    cv2.imshow('Safety Lane Violation Detection', annotated_frame)

    # Switch off video when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all open windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()