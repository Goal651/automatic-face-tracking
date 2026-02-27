# 🛡️ FaceLock: Intelligent Real-Time Face Locking System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Performance](https://img.shields.io/badge/CPU-Optimized-success.svg)](#performance-optimization)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance, CPU-optimized facial recognition and tracking system. Built with **ArcFace ONNX**, **Haar Cascades**, and **MediaPipe**, FaceLock provides stable face locking, action detection (blinks, smiles), and persistent behavior logging without requiring a GPU.

---

## ✨ Key Features

-   **🔒 Target Face Locking**: Manually select and lock onto a specific identity. The system maintains a persistent lock even if the face briefly leaves the frame.
-   **⚡ High Performance**: Optimized with spatial identity caching and skip-frame recognition, achieving 30+ FPS on standard CPUs.
-   **📊 Action Detection**: Real-time detection of eye blinks, smiles, and head movements with smooth velocity tracking.
-   **📝 Automated Logging**: Generates timestamped session histories (`.txt`) documenting every detected action for behavior analysis.
-   **🎯 Sub-pixel Alignment**: Uses 5-point landmark alignment (eyes, nose, mouth) to ensure maximum accuracy for the ArcFace embedder.

---

## 🚀 Quick Start

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
- **Run Locking System**: `python -m src.face_locking` (Main interface)
- **Run Recognition**: `python -m src.recognize` (Multi-face demo)

---

## 🛠️ Performance Optimization

We have implemented several "Brain-Saving" techniques to ensure zero lag:
-   **Identity Caching**: Recognizes a face once and "follows the box" spatially, avoiding expensive AI re-calculations on every frame.
-   **Recognition Skipping**: Heavy AI verification runs only once every 10 frames or upon significant movement.
-   **Noise Filtering**: Minimum face size thresholds (100x100) prevent the CPU from wasting cycles on background shadows.

---

## 🎹 Controls
| Key | Action |
| :--- | :--- |
| **L** | Toggle Lock on Target |
| **S** | Save Action History |
| **R** | Reload Face Database |
| **m** | Toggle Mirror Mode |
| **D** | Toggle Detailed UI |
| **Q** | Quit System |

---

## 📂 Project Structure

```text
├── src/
│   ├── face_locking.py     # Main system & UI logic
│   ├── action_detection.py # Blink/Smile/Movement algorithms
│   ├── recognize.py        # Core recognition & caching
│   └── enroll.py           # Face database management
├── data/
│   ├── db/                 # Face embeddings (.npz)
│   └── history/            # Generated action logs
└── models/                 # ONNX model files
```

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.# automatic-face-tracking
