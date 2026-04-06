```markdown
# ⚽ Football Analysis System

A computer vision pipeline that analyzes football match footage to track players, measure speed and distance, detect ball possession, and correct for camera movement.

---

## 📌 Features

- **Player & Ball Detection** — YOLO-based object detection with ByteTrack for consistent IDs across frames
- **Team Assignment** — Automatically classifies players into two teams based on jersey color using KMeans clustering
- **Camera Movement Estimation** — Detects and compensates for camera panning using optical flow
- **Perspective Transformation** — Converts pixel coordinates to real-world field meters
- **Speed & Distance Tracking** — Calculates each player's speed (km/h) and total distance covered (m)
- **Ball Possession** — Assigns ball possession per frame and displays running team percentages
- **Video Annotation** — Draws ellipses, triangles, speed stats, and possession overlay onto output video

---

## 🗂️ Project Structure

```
├── main.py
├── utils/
│   └── bbox_utils.py               # Bounding box and distance helper functions
├── trackers/
│   └── tracker.py                  # YOLO detection + ByteTrack + annotation drawing
├── team_assigner/
│   └── team_assigner.py            # Jersey color clustering for team classification
├── player_ball_assigner/
│   └── player_ball_assigner.py     # Assigns ball possession to nearest player
├── camera_movement_estimator/
│   └── camera_movement_estimator.py # Optical flow based camera movement detection
├── view_transformer/
│   └── view_transformer.py         # Perspective transform — pixels to real world meters
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py # Player speed and distance calculation
└── output_videos/                  # Saved annotated output videos
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/football-analysis.git
cd football-analysis
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python main.py
```

Input video is read from `input_videos/`, processed through the full pipeline, and saved to `output_videos/`.

---

## 🧠 Pipeline Overview

```
Read Video Frames
      ↓
Detect & Track Objects (YOLO + ByteTrack)
      ↓
Interpolate Missing Ball Positions
      ↓
Estimate Camera Movement (Optical Flow)
      ↓
Adjust Object Positions (Remove Camera Movement)
      ↓
Transform Positions to Real World Coordinates
      ↓
Assign Players to Teams (KMeans Jersey Color)
      ↓
Assign Ball Possession (Nearest Player)
      ↓
Calculate Speed & Distance
      ↓
Draw Annotations & Save Video
```

---

## 📦 Dependencies

- `ultralytics`         — YOLO object detection
- `supervision`         — ByteTrack multi-object tracking
- `opencv-python`       — Video I/O, optical flow, drawing
- `scikit-learn`        — KMeans clustering for team assignment
- `numpy`               — Array operations
- `pandas`              — Ball position interpolation
```

