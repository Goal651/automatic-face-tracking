# FaceLock: Intelligent Real-Time Face Locking System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Performance](https://img.shields.io/badge/CPU-Optimized-success.svg)](#performance-optimization)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance, CPU-optimized facial recognition and tracking system. Built with **ArcFace ONNX**, **Haar Cascades**, and **MediaPipe**, FaceLock provides stable face locking, action detection (blinks, smiles), and persistent behavior logging without requiring a GPU.

---

## Key Features

- **Target Face Locking**: Manually select and lock onto a specific identity. The system maintains a persistent lock even if the face briefly leaves the frame.
- **High Performance**: Optimized with spatial identity caching and skip-frame recognition, achieving 30+ FPS on standard CPUs.
- **Action Detection**: Real-time detection of eye blinks, smiles, and head movements with smooth velocity tracking.
- **Automated Logging**: Generates timestamped session histories documenting every detected action for behavior analysis.
- **MQTT Servo Control**: Integrated servo positioning system with P-controller and auto-scan functionality.
- **Comprehensive Administration**: Advanced database management with cascade deletion and data cleaning capabilities.

---

## Quick Start

### 1. Installation
```bash
git clone https://github.com/goal651/FaceLocking.git
cd FaceLocking

# Set up Python 3.10 using pyenv
pyenv install 3.10 --skip-existing
pyenv local 3.10

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Setup
The system uses the **ArcFace ResNet-50** model. Download and place it in the `models/` directory:
```bash
python init_project.py

# Download ArcFace ONNX
curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
unzip -o buffalo_l.zip
cp w600k_r50.onnx models/embedder_arcface.onnx
rm buffalo_l.zip *.onnx
```

### 3. Usage
- **Enroll a User**: `python -m src.enroll` (Capture face samples for the database)
- **Run Locking System**: `python main.py` (Main interface with advanced servo control)
- **Run Recognition**: `python -m src.recognize` (Multi-face demo)
- **Database Administration**: `python admin.py` (Advanced management tools)
- **Test Utilities**: `python -m src.utils` (Camera/face detection tests)
- **Embedding Tools**: `python -m src.embed` (Face embedding visualization)
- **Threshold Evaluation**: `python -m src.evaluate` (System performance analysis)

---

## Performance Optimization

We have implemented several techniques to ensure zero lag:
- **Identity Caching**: Recognizes a face once and "follows the box" spatially, avoiding expensive AI re-calculations on every frame.
- **Recognition Skipping**: Heavy AI verification runs only once every 10 frames or upon significant movement.
- **Noise Filtering**: Minimum face size thresholds prevent the CPU from wasting cycles on background shadows.

---

## Controls

| Key | Action |
| :--- | :--- |
| **Q** | Quit System |
| **R** | Reload Face Database |
| **L** | Toggle Lock on Target |
| **+/-** | Adjust Smile Detection Threshold |
| **F1/F2** | Adjust Face Detection Sensitivity |
| **M** | Toggle Mirror Mode |
| **M** | Toggle Landmarks Display |
| **C** | Toggle Confidence Display |
| **D** | Toggle Detailed UI |
| **[/]** | Adjust Window Scaling |
| **W/S** | Increase/Decrease Window Size |
| **F** | Toggle Fullscreen Mode |
| **A** | Save Action History |
| **P** | Toggle MQTT Publishing |

### Window Features:
- **Resizable Window** - Drag edges or use W/S keys
- **Fullscreen Mode** - Press F to toggle
- **Dynamic Scaling** - Adjust window size with keyboard
- **Manual Resizing** - Window is fully resizable with mouse

---

## Project Structure

```text
├── main.py                 # Consolidated main application
├── admin.py                # Enhanced database administration
├── src/
│   ├── action_detection.py # Advanced movement, blink, smile detection
│   ├── recognize.py        # Core recognition & caching
│   ├── enroll.py           # Face database management
│   ├── embed.py            # Face embedding tools
│   ├── evaluate.py         # Threshold evaluation
│   ├── haar_5pt.py         # 5-point alignment
│   ├── mediapipe_compat.py # MediaPipe compatibility
│   ├── landmarks.py        # Landmark utilities
│   ├── serial_comm.py      # Serial communication
│   ├── align.py            # Face alignment
│   └── utils.py            # Camera/detection utilities
├── data/
│   ├── db/                 # Face embeddings (.npz)
│   ├── enroll/             # User enrollment folders
│   └── history/            # Generated action logs
└── models/                 # ONNX model files
```

---

## MQTT Integration

The system includes comprehensive MQTT servo control with advanced movement detection:
- **Advanced Movement Detection**: Uses velocity-based control from `action_detection.py`
- **P-Controller**: Smooth servo positioning based on face movement velocity
- **Auto-Scan Mode**: Servo scanning when no face is detected
- **Auto-Stop on Lock**: Immediate centering when target face is locked
- **Configurable Parameters**: Adjust servo gains, scan speed, and angles
- **Speed-Responsive Control**: Faster movements result in more responsive servo tracking

MQTT Topics:
- `vision/team351/movement` - Servo angle commands
- `robot/status` - System status messages
- `servo/status` - Servo feedback

---

## Database Administration

Use `admin_merged.py` for comprehensive database management:

### Features:
- **User Management**: List, search, and delete users
- **Cascade Deletion**: Remove users and all associated data (photos, enrollment folders, history)
- **Data Cleaning**: Clean orphaned photos, old history files, cache files
- **Storage Analysis**: Analyze disk usage and identify large files
- **Backup Management**: Automatic backups before modifications

### Usage:
```bash
python admin.py
```

### Key Options:
1. List all users with enrollment folder status
2. Delete user (database only)
3. Delete user (CASCADE - all data)
4. Data cleaning menu
5. Storage analysis

---

## Architecture

The system has been refactored into a modular architecture:

### Core Components:
- **FaceTracker**: Base face tracking framework with callback system
- **ServoController**: MQTT servo control with P-controller
- **FaceLockingApp**: Application layer with UI integration

### Benefits:
- **Modularity**: Each component has a single responsibility
- **Reusability**: Components can be used for different applications
- **Maintainability**: Easy to test and extend individual modules
- **Extensibility**: Callback system allows custom behaviors

---

## Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
