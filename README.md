# 🌊 RipCatch-v2.0 - Rip Current Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

> **AI-powered rip current detection system to enhance beach safety and save lives.**

Rip currents are responsible for approximately 100 deaths annually in the United States alone. **RipCatch** leverages state-of-the-art computer vision and deep learning to automatically detect rip currents in beach imagery and video streams, providing real-time warnings to beachgoers and lifeguards.

---

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Performance Metrics](#-performance-metrics)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Versions](#-model-versions)
- [Training](#-training)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

### 🎯 Core Capabilities
- **Real-time Detection**: Process video streams at 10-15 FPS on GPU
- **High Accuracy**: 88.64% mAP@50 with balanced precision and recall
- **Multi-Modal Input**: Supports images, videos, and live camera feeds
- **Beach Scene Understanding**: Two-stage detection (beach classification + rip current detection)
- **Production Ready**: Export to ONNX, TensorFlow, TFLite for deployment

### 🧠 Technical Highlights
- **YOLOv8 Architecture**: State-of-the-art object detection
- **Advanced Training**: Gradient accumulation, mixed precision, robust augmentation
- **Optimized for Edge Devices**: Runs on NVIDIA Jetson, mobile devices, and cloud
- **Comprehensive Testing**: Validated on 16,907 images across diverse beach conditions

---

## 🎬 Demo

### Image Detection
![Rip Current Detection Demo](Demo.mp4)
*Example: RipCatch detecting rip currents in a beach scene*

### Video Detection
- **Input**: Beach surveillance footage
- **Output**: Real-time bounding boxes with confidence scores
- **Performance**: 10-15 FPS on NVIDIA RTX 3080

*Sample videos available in `Testing/videos/`*

---

## 📊 Performance Metrics

### RipCatch v2.0 (Latest Model)

| Metric | Value | Assessment |
|--------|-------|------------|
| **mAP@50** | **88.64%** | ✅ Strong performance |
| **mAP@50-95** | **61.45%** | ✅ Good localization |
| **Precision** | **89.03%** | ✅ Low false positives |
| **Recall** | **89.51%** | ✅ Few missed detections |
| **F1-Score** | **89.27%** | ✅ Well balanced |

### Training Configuration
- **Model**: YOLOv8m (Medium)
- **Dataset**: 16,907 images (14,436 train / 1,804 val / 667 test)
- **Training Time**: ~4-5 hours on NVIDIA RTX 3080
- **Image Resolution**: 640×640 pixels
- **Batch Size**: 16 (effective 64 with gradient accumulation)

### Inference Speed
```
GPU (RTX 3080):  10-15 FPS
GPU (Jetson Xavier): 5-8 FPS
CPU (Intel i7):  1-2 FPS
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM, 16GB recommended

### Installation (5 Minutes)

```bash
# 1. Clone the repository
git clone https://github.com/naga-narala/RipCatch.git
cd RipCatch

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pre-trained model weights
# Download from: [Releases](https://github.com/naga-narala/RipCatch/releases)
# Place in: RipCatch-v2.0/Model/weights/best.pt
```

### Run Inference (30 Seconds)

```python
from ultralytics import YOLO

# Load model
model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Run inference on image
results = model('Testing/Mixed/RIP1.webp')
results[0].show()

# Run inference on video
results = model('Testing/videos/video_test_1.mp4', save=True)
```

**See [QUICK_START.md](QUICK_START.md) for detailed setup guide.**

---

## 📦 Installation

### Method 1: pip (Recommended)

```bash
# Install PyTorch with CUDA 11.8 (for NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Method 2: Conda

```bash
# Create environment from file
conda env create -f environment.yml
conda activate ripcatch

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Method 3: Docker (Coming Soon)

```bash
docker pull naga-narala/ripcatch:latest
docker run -it --gpus all naga-narala/ripcatch:latest
```

---

## 🎯 Usage

### 1. Image Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Single image
results = model('path/to/beach_image.jpg')

# Batch inference
results = model(['image1.jpg', 'image2.jpg', 'image3.jpg'])

# Display results
for result in results:
    result.show()
    result.save('output.jpg')
```

### 2. Video Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Process video
results = model(
    'beach_video.mp4',
    save=True,              # Save annotated video
    conf=0.25,              # Confidence threshold
    iou=0.45,               # NMS IoU threshold
    device=0                # GPU device (0 for first GPU)
)

# Output saved to: runs/detect/predict/
```

### 3. Live Camera Feed

```python
from ultralytics import YOLO
import cv2

model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Open camera (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame, verbose=False)
    
    # Display
    annotated = results[0].plot()
    cv2.imshow('RipCatch - Live Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 4. Custom Configuration

```python
from ultralytics import YOLO

model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Advanced inference options
results = model.predict(
    source='beach_image.jpg',
    conf=0.35,              # Confidence threshold (0-1)
    iou=0.45,               # NMS IoU threshold
    imgsz=640,              # Image size
    device='cuda:0',        # Device (cuda:0, cpu)
    max_det=10,             # Max detections per image
    augment=True,           # Test-time augmentation
    visualize=False,        # Visualize features
    save=True,              # Save results
    save_txt=True,          # Save labels
    save_conf=True,         # Save confidence scores
    save_crop=True,         # Save cropped predictions
    line_width=2,           # Bounding box line width
    show_labels=True,       # Show labels
    show_conf=True,         # Show confidence scores
)
```

### 5. Extract Detection Results

```python
# Get bounding boxes, confidence, class
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()      # Bounding boxes
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    classes = result.boxes.cls.cpu().numpy()      # Class IDs
    
    for box, conf, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = box
        print(f"Rip current detected at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        print(f"Confidence: {conf:.2%}")
```

---

## 📁 Project Structure

```
RipCatch/
├── RipCatch-v1.0/           # Initial prototype
│   ├── Datasets/            # (Gitignored) Training data
│   ├── Documentation/       # Version-specific docs
│   └── models/              # Model checkpoints
│
├── RipCatch-v1.1/           # Two-stage detection
│   ├── Datasets/            # Beach + rip current datasets
│   ├── models/
│   │   ├── beach_classifier_best.pt
│   │   └── rip_detector_best.pt
│   ├── RipCatch-v1.1.ipynb
│   └── Documentation/
│       └── LOCAL_SETUP_GUIDE.md
│
├── RipCatch-v2.0/           # ⭐ Latest production model
│   ├── Datasets/
│   │   └── rip_dataset.zip  # (Download separately)
│   ├── Model/
│   │   ├── weights/
│   │   │   ├── best.pt      # 🔥 Best model checkpoint
│   │   │   └── last.pt      # Latest checkpoint
│   │   └── args.yaml        # Training configuration
│   ├── Results/
│   │   ├── evaluation_results.json
│   │   ├── results.csv
│   │   └── video_test_*_output.mp4
│   ├── RipCatch-v2.0.ipynb  # Training notebook
│   └── Documentation/
│       ├── NOTEBOOK_PLAN.md
│       └── TRAINING_SUMMARY_REPORT.md
│
├── Testing/                 # Test images and videos
│   ├── beach/               # Beach scenes (23 images)
│   ├── Mixed/               # Mixed beach + rip currents (34 images)
│   ├── real_time/           # Real-time test cases (4 images)
│   └── videos/              # Test videos (2 videos)
│
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment
├── setup.py                 # Package installation
├── pyproject.toml           # Modern Python config
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
├── README.md                # This file
├── QUICK_START.md           # Fast setup guide
├── CONTRIBUTING.md          # Contribution guidelines
├── CHANGELOG.md             # Version history
└── FOLDER_STRUCTURE.md      # Detailed structure docs
```

**See [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) for detailed explanation.**

---

## 🔄 Model Versions

### v2.0 - Production Model (Current) ⭐

**Status**: Production-ready  
**Released**: October 2025  
**Performance**: 88.64% mAP@50

**Features**:
- Single-stage YOLOv8m detection
- Advanced training with gradient accumulation
- Optimized hyperparameters (early stopping, regularization)
- Comprehensive evaluation and testing
- Export-ready for deployment

**Training Details**:
```yaml
Model: YOLOv8m
Epochs: 200 (early stopping at ~70)
Batch Size: 16 (effective 64)
Image Size: 640×640
Optimizer: AdamW
Learning Rate: 0.0007 → 0.005
Augmentation: Mosaic, MixUp, Copy-Paste, RandAugment
```

---

### v1.1 - Two-Stage Detection

**Status**: Deprecated  
**Released**: September 2025

**Features**:
- Stage 1: Beach scene classifier
- Stage 2: Rip current detector
- Separate models for each stage

**Performance**:
- Beach Classifier: ~95% accuracy
- Rip Detector: ~85% mAP@50

---

### v1.0 - Initial Prototype

**Status**: Deprecated  
**Released**: August 2025

**Features**:
- Basic YOLOv8n implementation
- Limited dataset
- Proof of concept

---

## 🏋️ Training

### Using Pre-configured Notebook (Recommended)

```bash
# 1. Download dataset (see below)
# 2. Open Jupyter notebook
jupyter notebook RipCatch-v2.0/RipCatch-v2.0.ipynb

# 3. Run cells sequentially
#    - Cell 1-2: Setup
#    - Cell 3-4: Dataset validation
#    - Cell 5-6: Training (4-5 hours)
#    - Cell 7-9: Evaluation and testing
```

### Using Python Script

```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolov8m.pt')

# Train on custom dataset
results = model.train(
    data='path/to/data.yaml',
    epochs=200,
    batch=16,
    imgsz=640,
    patience=25,
    optimizer='AdamW',
    lr0=0.0007,
    lrf=0.005,
    weight_decay=0.0015,
    dropout=0.15,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.3,
    device=0
)

# Evaluate
metrics = model.val()
print(f"mAP@50: {metrics.box.map50:.2%}")
```

### Dataset Preparation

**Required Structure**:
```
rip_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**data.yaml Format**:
```yaml
path: /path/to/rip_dataset
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['rip']
```

**Dataset Download**:
1. Download from [Google Drive](#) or [Roboflow Universe](#)
2. Extract to `RipCatch-v2.0/Datasets/rip_dataset/`
3. Verify structure with provided validation scripts

**See [TRAINING_SUMMARY_REPORT.md](RipCatch-v2.0/Documentation/TRAINING_SUMMARY_REPORT.md) for detailed training analysis.**

---

## 🚢 Deployment

### Export Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Export to ONNX (recommended for production)
model.export(
    format='onnx',
    imgsz=640,
    optimize=True,
    simplify=True
)

# Export to TensorFlow Lite (mobile)
model.export(format='tflite', int8=True)

# Export to TorchScript (C++)
model.export(format='torchscript')
```

### Deployment Options

| Platform | Format | Performance | Guide |
|----------|--------|-------------|-------|
| **Cloud (AWS/Azure)** | ONNX | ⚡ Fast | [Deploy Guide](#) |
| **Edge (Jetson)** | TensorRT | 🚀 Fastest | [Jetson Guide](#) |
| **Mobile (iOS/Android)** | TFLite | 🏃 Good | [Mobile Guide](#) |
| **Web Browser** | ONNX.js | 🐌 Moderate | [Web Guide](#) |
| **Raspberry Pi** | TFLite | 🐌 Slow | [RPi Guide](#) |

### API Integration (Example)

```python
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO('best.pt')

@app.post("/detect")
async def detect_rip_current(file: UploadFile):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run inference
    results = model(img)
    
    # Extract detections
    detections = []
    for box in results[0].boxes:
        detections.append({
            'bbox': box.xyxy[0].tolist(),
            'confidence': float(box.conf[0]),
            'class': 'rip_current'
        })
    
    return {'detections': detections, 'count': len(detections)}
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit changes**: `git commit -m 'Add AmazingFeature'`
4. **Push to branch**: `git push origin feature/AmazingFeature`
5. **Open Pull Request**

### Areas for Contribution

- 🐛 Bug fixes and error handling
- 📊 Dataset expansion and labeling
- 🔬 Model improvements and experiments
- 📱 Mobile app development
- 🌐 Web interface creation
- 📚 Documentation enhancements
- 🧪 Testing and validation

---

## 📖 Citation

If you use RipCatch in your research or project, please cite:

```bibtex
@software{ripcatch2025,
  title = {RipCatch: AI-Powered Rip Current Detection System},
  author = {Sravan Kumar},
  year = {2025},
  url = {https://github.com/naga-narala/RipCatch},
  version = {2.0}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Summary**:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ Liability limitation
- ⚠️ Warranty disclaimer

---

## 🙏 Acknowledgments

### Frameworks & Libraries
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** - Core detection framework
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[OpenCV](https://opencv.org/)** - Computer vision operations

### Datasets
- **Roboflow Universe** - Dataset hosting and annotation
- **Public beach imagery** - Various sources (properly attributed)

### Inspiration
- **Beach Safety Organizations** - Domain expertise and validation
- **Computer Vision Research Community** - Technical guidance

### Special Thanks
- NVIDIA for GPU support and optimization tools
- Open-source community for invaluable contributions
- Beach lifeguards and safety experts for domain knowledge

---

## 📞 Contact & Support

### Get Help
- 📧 **Email**: sravankumar.nnv@gmail.com
- 💬 **Discussions**: [GitHub Discussions](https://github.com/naga-narala/RipCatch/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/naga-narala/RipCatch/issues)

### Follow Development
- ⭐ **Star this repo** to show support
- 👀 **Watch** for updates
- 🔔 **Subscribe** to releases

---

## 🗺️ Roadmap

### v2.1 (In Progress)
- [ ] Enhanced temporal analysis for video streams
- [ ] Multi-camera tracking and fusion
- [ ] Automated alert system integration
- [ ] Mobile app (iOS/Android)

### v3.0 (Planned)
- [ ] Real-time depth estimation
- [ ] Weather condition integration
- [ ] Crowd density analysis
- [ ] Multi-language support

### Future
- [ ] Drone-based detection
- [ ] Satellite imagery integration
- [ ] Global beach monitoring network
- [ ] Public API for researchers

---

## ⚠️ Disclaimer

**RipCatch is a research tool and should NOT replace professional lifeguard supervision or official safety measures.** Always follow local beach safety guidelines and warnings. The system may produce false positives or miss detections. Use at your own risk.

---

<div align="center">

**Made with ❤️ for Beach Safety**

**[⬆ Back to Top](#-ripcatch---rip-current-detection-system)**

---

*Last Updated: October 2025*

</div>

