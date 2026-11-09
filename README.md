# Safety Lane Violation Detection System

## 1. Introduction

This project implements a real-time computer vision system that detects and tracks vehicles violating emergency lane restrictions using YOLOv12 object detection. The system monitors a user-defined region of interest (ROI) and counts vehicles that cross the designated safety lane boundary. Built for traffic management and enforcement, it combines object detection, multi-object tracking, and line-crossing detection, with live dashboard analytics.

**Core Features:**

- Real-time vehicle detection and tracking using YOLOv12
- ROI-based monitoring for focused analysis
- Automatic line-crossing detection
- Live dashboard displaying violation statistics
- Color-coded visualization (blue for normal, red for violations)
- Annotated video recording
- Vehicle type classification and counting

## 2. Methodology / Approach

The system employs a multi-stage pipeline combining object detection, tracking, and spatial analysis:

**Object Detection:** YOLOv12 model detects vehicles (cars, motorcycles, buses, trucks) in each frame with high accuracy and speed.

**Multi-Object Tracking:** Objects are tracked across frames using unique IDs, maintaining consistency in vehicle identification and enabling trajectory analysis.

**ROI Filtering:** Only vehicles with center points within the user-defined region of interest are processed, reducing false alerts and focusing on relevant areas.

**Line Crossing Detection:** The system calculates if a vehicle trajectory intersects the safety lane boundary by analyzing the cross product of path vectors, determining violation direction.

**Dashboard Analytics:** Real-time statistics display total vehicles, violation counts, violation ratios, and breakdowns by vehicle type.

### 2.1 System Architecture

```
[Video Input]
    ↓
[YOLOv12 Detection] → [Multi-Object Tracking]
    ↓
[ROI Filtering]
    ↓
[Line Crossing Detection] → [Violation Classification]
    ↓
[Dashboard & Visualization]
    ↓
[Video Output]
```

### 2.2 Processing Pipeline

1. Read video frame
2. Run YOLOv12 detection with tracking persistence
3. Filter detections to vehicle classes only
4. Check if vehicle center is within ROI
5. Update tracking history with center point
6. Detect line crossing using vector cross product
7. Count violations and update statistics
8. Apply color coding and transparency based on violation status
9. Render dashboard with analytics
10. Save annotated frame to output video

## 3. Mathematical Framework

### 3.1 Cross Product for Line Crossing Detection

The system uses the cross product of 2D vectors to determine if a vehicle has crossed the safety line and in which direction.

**Vector Definitions:**

Line vector (from point A to point B on the safety line):

$$\mathbf{v}_{\text{line}} = (B_x - A_x, B_y - A_y)$$

Path vector (from previous position to current position):

$$\mathbf{v}_{\text{path}} = (P_{\text{current},x} - P_{\text{previous},x}, P_{\text{current},y} - P_{\text{previous},y})$$

**2D Cross Product (Scalar Result):**

$$\mathbf{v}_1 \times \mathbf{v}_2 = x_1 y_2 - x_2 y_1$$

For line crossing detection:

$$C = \mathbf{v}_{\text{line}} \times \mathbf{v}_{\text{path}} = (B_x - A_x)(P_{\text{current},y} - P_{\text{previous},y}) - (B_y - A_y)(P_{\text{current},x} - P_{\text{previous},x})$$

**Crossing Direction Determination:**

$$\text{Direction} = \begin{cases}
\text{Left to Right} & \text{if } C > 0 \\
\text{Right to Left} & \text{if } C < 0 \\
\text{No Crossing} & \text{if } C = 0
\end{cases}$$

**Violation Condition:**

A violation is detected when:

$$|C| > \epsilon \quad \text{and} \quad \text{sign}(C_{\text{prev}}) \neq \text{sign}(C_{\text{current}})$$

where $\epsilon$ is a small threshold to filter numerical noise.

### 3.2 Point-in-Polygon Test (ROI Filtering)
 
To determine if a vehicle center point $P = (x, y)$ is inside the ROI polygon with vertices $V = \{V_1, V_2, ..., V_n\}$, the system uses the ray-casting algorithm:
 
$$\text{Inside}(P) = \sum_{i=1}^{n} \mathbb{1}_{\text{intersect}}(P, V_i, V_{i+1}) \mod 2$$

where the indicator function is defined as:

$$\mathbb{1}_{\text{intersect}}(P, V_i, V_{i+1}) = \begin{cases} 1 & \text{if ray from } P \text{ intersects edge } (V_i, V_{i+1}) \\ 0 & \text{otherwise} \end{cases}$$

The result equals **1** (inside the polygon) or **0** (outside the polygon).

**Intersection Test:**
 
For edge from $V_i = (x_i, y_i)$ to $V_{i+1} = (x_{i+1}, y_{i+1})$:
 
$$\text{Intersects} = \begin{cases}
\text{True} & \text{if } (y_i > y) \neq (y_{i+1} > y) \text{ and } x < \frac{(x_{i+1} - x_i)(y - y_i)}{(y_{i+1} - y_i)} + x_i \\
\text{False} & \text{otherwise}
\end{cases}$$

### 3.3 Vehicle Center Point Calculation

For a bounding box with coordinates $(x_1, y_1, x_2, y_2)$:

$$P_{\text{center}} = \left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)$$

where:
- $(x_1, y_1)$ = top-left corner
- $(x_2, y_2)$ = bottom-right corner

### 3.4 Violation Statistics

**Total Violation Count:**

$$N_{\text{violations}} = \sum_{c \in \text{Classes}} N_c$$

where $N_c$ = number of violations for vehicle class $c$ (car, motorcycle, bus, truck).

**Violation Ratio:**

$$R_{\text{violation}} = \frac{N_{\text{violations}}}{N_{\text{total}}} \times 100\%$$

where $N_{\text{total}}$ = total number of tracked vehicles.

**Per-Class Violation Rate:**

$$R_c = \frac{N_c}{N_{\text{violations}}} \times 100\%$$

### 3.5 Trajectory Path Analysis

The system maintains a trajectory history buffer for each tracked vehicle:

$$\mathcal{H}_{\text{ID}} = \{P_1, P_2, ..., P_k\}$$

where:
- $\mathcal{H}$ = position history buffer for a vehicle with unique tracking ID
- $P_i$ = center point at frame $i$
- $k$ = maximum history length (30 frames)

**Path Segment:**

$$\mathbf{S}_i = P_{i+1} - P_i = (x_{i+1} - x_i, y_{i+1} - y_i)$$

**Trajectory Length (Total Distance Traveled):**

$$D_{\text{total}} = \sum_{i=1}^{k-1} \|\mathbf{S}_i\| = \sum_{i=1}^{k-1} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}$$

### 3.6 Visualization Color Coding

**Bounding Box Color:**

$$\text{Color}_{\text{box}} = \begin{cases}
(255, 0, 0) & \text{if vehicle is violator (Red)} \\
(0, 0, 255) & \text{if vehicle is normal (Blue)}
\end{cases}$$

**Transparency (Alpha Channel):**

$$\alpha = \begin{cases}
0.7 & \text{if violator (70\% opaque red fill)} \\
0.0 & \text{if normal (no fill, only border)}
\end{cases}$$

**Dashboard Metrics Display:**

$$\text{Dashboard} = \begin{bmatrix}
\text{Total Vehicles:} & N_{\text{total}} \\
\text{Violations:} & N_{\text{violations}} \\
\text{Violation Rate:} & R_{\text{violation}} \\
\text{Cars:} & N_{\text{car}} \\
\text{Motorcycles:} & N_{\text{motorcycle}} \\
\text{Buses:} & N_{\text{bus}} \\
\text{Trucks:} & N_{\text{truck}}
\end{bmatrix}$$

## 4. Requirements

```txt
opencv-python>=4.5.0
numpy>=1.21.0
ultralytics>=8.0.0
```

## 5. Installation & Configuration

### 5.1 Environment Setup

```bash
# Clone the repository
git clone https://github.com/kemalkilicaslan/Safety-Lane-Violation-Detection-System.git
cd Safety-Lane-Violation-Detection-System

# Install required packages
pip install -r requirements.txt
```

### 5.2 Project Structure

```
Safety-Lane-Violation-Detection-System/
├─ Safety-Lane-Violation-Detection-System.py
├─ config.py
├─ Dockerfile
├─ .dockerignore
├─ docker-compose.yml
├─ .gitignore
├─ README.md
├─ requirements.txt
└─ LICENSE
```

### 5.3 Required Files

- **YOLOv12 Model:** `yolo12x.pt` (automatically downloaded on first run)
- **Input Video:** Place your traffic video file in the project directory

## 6. Usage / How to Run

### 6.1 Basic Execution

```bash
python Safety-Lane-Violation-Detection-System.py
```

### 6.2 Configuration

All adjustable parameters live in `config.py`. Update them there without touching the main script:

```python
# Model and video files
MODEL_PATH = "yolo12x.pt"
INPUT_VIDEO = "Traffic-Flow.mp4"                      # input video file name
OUTPUT_VIDEO = "Safety-Lane-Violation-Detection.mp4"  # output video file name

# Region of Interest (ROI) polygon — list of [x, y] points
ROI_COORDINATES = [[770, 280], [1130, 280], [1500, 900], [100, 900]]

# Counting line — two [x, y] endpoints
LINE_COORDINATES = [[375, 650], [615, 650]]

# Vehicle classes (COCO IDs) and history length
VEHICLE_CLASSES = [2, 3, 5, 7]
TRACK_HISTORY_LENGTH = 30
```

### 6.3 Controls

- Press `q` to quit the application

### 6.4 Output

The processed video is saved as:

```
Safety-Lane-Violation-Detection.mp4
```

## 7. Docker Deployment

The system can be packaged and run in a container so that the code, its
dependencies and the runtime environment stay consistent across machines.

### Files

- `Dockerfile` - builds the image (Python 3.11 + OpenCV/ffmpeg system libraries + Python dependencies).
- `.dockerignore` - keeps large media and model files out of the build context.
- `docker-compose.yml` - runs the container and mounts the project folder for input/output.

### Build and run

```bash
docker compose up --build

# or with the Docker CLI directly
docker build -t safety-lane-violation-detection:latest .
docker run --rm -v "$(pwd)":/app safety-lane-violation-detection:latest
```

### Notes

- Input and output videos are read from and written to the project folder through
  the bind mount, so the annotated output video appears on the host after the run.
- The YOLO model weight is downloaded automatically on the first run and cached in
  the mounted volume.
- The script calls `cv2.imshow`, which needs a graphical display. On a headless
  server the output video is still written to disk; the live preview window is
  only available on Linux with X11 forwarding (see the commented lines in
  `docker-compose.yml`).

## 8. Application / Results

### 8.1 Input Video

[![Traffic Flow](https://github.com/kemalkilicaslan/Safety-Lane-Violation-Detection-System/blob/main/Traffic-Flow.webp)](https://www.youtube.com/watch?v=iNQ5blLITxk)

### 8.2 Output Video

[![Safety Lane Violation Detection](https://github.com/kemalkilicaslan/Safety-Lane-Violation-Detection-System/blob/main/Safety-Lane-Violation-Detection.webp)](https://www.youtube.com/watch?v=Mm5473bf5aA)

## 9. System Configuration

### 9.1 Vehicle Classes

| Class ID | Vehicle Type |
|----------|--------------|
| 2        | Car          |
| 3        | Motorcycle   |
| 5        | Bus          |
| 7        | Truck        |

### 9.2 Visualization Settings

- **Normal Vehicle:** Blue bounding box, no fill
- **Violating Vehicle:** Red bounding box with 70% semi-transparent red fill
- **ROI:** Green polygon with 15% transparency
- **Safety Line:** Red line marking the violation boundary

## 10. Tech Stack

### 10.1 Core Technologies

- **Language:** Python 3.7+
- **Computer Vision:** OpenCV 4.5+
- **Deep Learning:** Ultralytics YOLO 8.0+
- **Object Detection & Tracking:** YOLOv12
- **Numerical Computing:** NumPy 1.21+

### 10.2 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.5+ | Video I/O, image processing, visualization |
| ultralytics | 8.0+ | YOLOv12 model and tracking inference |
| numpy | 1.21+ | Array operations and vector mathematics |

### 10.3 Pre-trained Model

**YOLOv12 (Extra Large): `yolo12x.pt`**

- **Architecture:** YOLOv12 deep learning model
- **Classes:** 80 COCO classes including vehicles
- **Input:** Auto-resized frames
- **Tracking:** Built-in multi-object tracking with ID persistence

## 11. License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://github.com/kemalkilicaslan/Safety-Lane-Violation-Detection-System/blob/main/LICENSE).

## 12. References

1. Ultralytics [YOLOv12](https://docs.ultralytics.com/models/yolo12/) Documentation.
2. OpenCV [Object Tracking](https://docs.opencv.org/3.4/dc/d6b/group__video__track.html) Documentation.

---

### Acknowledgments

Thanks to the Ultralytics team for the YOLO framework and YOLOv12 models, and to the OpenCV community for their computer vision tools and documentation. Sample traffic footage used for demonstration purposes only.

---

**Note:** Ensure compliance with local traffic laws and privacy regulations when deploying vehicle monitoring systems. This system is intended for authorized traffic management, law enforcement, and research purposes only. Always verify the legal implications of recording vehicle data in your jurisdiction.