# 🚀 RipCatch - Quick Start Guide

Get RipCatch up and running in **5 minutes**!

---

## ⚡ Prerequisites

Before starting, ensure you have:

- ✅ **Python 3.8+** installed
- ✅ **8GB+ RAM** (16GB recommended)
- ✅ **10GB+ free disk space**
- ✅ **NVIDIA GPU** with CUDA support (optional but recommended)
- ✅ **Git** installed

---

## 📦 Installation

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/naga-narala/RipCatch.git
cd RipCatch
```

**Time**: ~30 seconds

---

### Step 2: Create Virtual Environment

#### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

#### Option B: Using Conda

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ripcatch
```

**Time**: ~1 minute

---

### Step 3: Install Dependencies

#### For GPU (NVIDIA with CUDA):

```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

#### For CPU Only:

```bash
# Install PyTorch CPU version
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

**Time**: ~2-3 minutes

---

### Step 4: Download Model Weights

**Option A: Direct Download**

1. Go to [Releases](https://github.com/naga-narala/RipCatch/releases/latest)
2. Download `ripcatch_v2.0_weights.zip`
3. Extract to `RipCatch-v2.0/Model/weights/`

**Option B: Command Line**

```bash
# Using wget (Linux/Mac)
wget https://github.com/naga-narala/RipCatch/releases/download/v2.0.0/best.pt -O RipCatch-v2.0/Model/weights/best.pt

# Using curl
curl -L https://github.com/naga-narala/RipCatch/releases/download/v2.0.0/best.pt -o RipCatch-v2.0/Model/weights/best.pt
```

**Time**: ~1 minute (depending on internet speed)

---

### Step 5: Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Test Ultralytics installation
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"

# Test model loading
python -c "from ultralytics import YOLO; model = YOLO('RipCatch-v2.0/Model/weights/best.pt'); print('Model loaded successfully!')"
```

**Expected Output:**
```
PyTorch: 2.0.1+cu118
CUDA Available: True
Ultralytics: OK
Model loaded successfully!
```

**Time**: ~10 seconds

---

## 🎯 First Run

### Test on Sample Image

```python
# Create test_detection.py
from ultralytics import YOLO

# Load model
model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Run inference on sample image
results = model('Testing/Mixed/RIP1.webp')

# Display results
results[0].show()

# Save results
results[0].save('output_detection.jpg')

print("✅ Detection complete! Check output_detection.jpg")
```

**Run it:**
```bash
python test_detection.py
```

**Time**: ~5 seconds

---

### Test on Sample Video

```python
# Create test_video.py
from ultralytics import YOLO

# Load model
model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Run inference on video
results = model(
    'Testing/videos/video_test_1.mp4',
    save=True,
    conf=0.25
)

print("✅ Video processing complete! Check runs/detect/predict/")
```

**Run it:**
```bash
python test_video.py
```

**Time**: ~30 seconds (depending on video length)

---

## 📊 Performance Benchmarks

### Expected Results on Test Images

| Test Case | Detection Rate | Avg Confidence | FPS (RTX 3080) |
|-----------|----------------|----------------|----------------|
| Beach scenes | 90%+ | 0.78 | 12-15 FPS |
| Clear rip currents | 95%+ | 0.85 | 12-15 FPS |
| Complex scenes | 75-85% | 0.65 | 12-15 FPS |
| Video streams | 85-90% | 0.75 | 10-12 FPS |

### Hardware Performance

| GPU | Inference Speed | Recommended Batch Size |
|-----|----------------|------------------------|
| RTX 4090 | 25-30 FPS | 64 |
| RTX 3080/3090 | 12-15 FPS | 32 |
| RTX 2080 Ti | 8-10 FPS | 16 |
| GTX 1080 Ti | 5-7 FPS | 8 |
| CPU (i7) | 1-2 FPS | 1 |

---

## 🔧 Common Issues & Solutions

### Issue 1: CUDA Not Available

**Symptoms:**
```
CUDA Available: False
```

**Solutions:**

1. **Check NVIDIA Driver:**
```bash
nvidia-smi
```
If this fails, install/update NVIDIA drivers: [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)

2. **Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Verify CUDA Version:**
```bash
nvcc --version
```
Match PyTorch CUDA version with system CUDA version.

---

### Issue 2: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce image size:**
```python
results = model('image.jpg', imgsz=320)  # Instead of 640
```

2. **Use CPU inference:**
```python
model = YOLO('best.pt')
results = model('image.jpg', device='cpu')
```

3. **Process smaller batches:**
```python
# Process one image at a time
for image_path in image_list:
    results = model(image_path)
```

---

### Issue 3: Model Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'RipCatch-v2.0/Model/weights/best.pt'
```

**Solutions:**

1. **Verify file path:**
```bash
# Check if file exists
ls -la RipCatch-v2.0/Model/weights/best.pt  # Linux/Mac
dir RipCatch-v2.0\Model\weights\best.pt     # Windows
```

2. **Download model weights again** from [Releases](#step-4-download-model-weights)

3. **Check current directory:**
```python
import os
print(f"Current directory: {os.getcwd()}")
```

---

### Issue 4: Slow Inference on CPU

**Symptoms:**
- Very slow inference (>5 seconds per image)
- High CPU usage

**Solutions:**

1. **Use smaller model** (if available):
```python
model = YOLO('yolov8n.pt')  # Nano version (faster but less accurate)
```

2. **Reduce image size:**
```python
results = model('image.jpg', imgsz=320)
```

3. **Disable verbose output:**
```python
results = model('image.jpg', verbose=False)
```

4. **Upgrade to GPU** for real-time performance

---

## 📖 Next Steps

### 1. **Explore Advanced Features**

```python
from ultralytics import YOLO

model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Adjust confidence threshold
results = model('image.jpg', conf=0.5)  # Higher threshold = fewer detections

# Enable Test-Time Augmentation (TTA) for better accuracy
results = model('image.jpg', augment=True)

# Save detection results as text
results = model('image.jpg', save_txt=True)

# Extract detection data
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    print(f"Found {len(boxes)} rip currents")
```

---

### 2. **Batch Processing**

```python
from pathlib import Path
from ultralytics import YOLO

model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Process all images in a folder
image_folder = Path('Testing/Mixed')
image_paths = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.webp'))

# Batch inference
results = model(image_paths, save=True)

print(f"✅ Processed {len(image_paths)} images")
```

---

### 3. **Video Stream Processing**

```python
from ultralytics import YOLO
import cv2

model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Open video file or camera
cap = cv2.VideoCapture('beach_video.mp4')  # Or use 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame, verbose=False)
    
    # Get annotated frame
    annotated_frame = results[0].plot()
    
    # Display
    cv2.imshow('RipCatch Detection', annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 4. **Export Model for Deployment**

```python
from ultralytics import YOLO

model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Export to ONNX (recommended for production)
model.export(format='onnx', imgsz=640, optimize=True, simplify=True)

# Export to TensorFlow Lite (for mobile)
model.export(format='tflite', int8=True)

# Export to TorchScript (for C++)
model.export(format='torchscript')

print("✅ Model exported successfully!")
```

---

## 📚 Additional Resources

### Documentation
- **[Full README](README.md)** - Complete project documentation
- **[Training Guide](RipCatch-v2.0/Documentation/TRAINING_SUMMARY_REPORT.md)** - Detailed training information
- **[Folder Structure](FOLDER_STRUCTURE.md)** - Repository organization
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute

### Tutorials
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### Community
- **GitHub Discussions**: [Ask questions](https://github.com/naga-narala/RipCatch/discussions)
- **Issues**: [Report bugs](https://github.com/naga-narala/RipCatch/issues)
- **Email**: sravankumar.nnv@gmail.com

---

## 🎓 Training Custom Model

Want to train on your own data?

### Quick Training

```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolov8m.pt')

# Train on custom dataset
results = model.train(
    data='path/to/your/data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    patience=20,
    device=0  # GPU 0
)

# Validate
metrics = model.val()

# Save
model.save('my_custom_model.pt')
```

**See [TRAINING_SUMMARY_REPORT.md](RipCatch-v2.0/Documentation/TRAINING_SUMMARY_REPORT.md) for advanced training guide.**

---

## 🐛 Troubleshooting Checklist

Before asking for help, verify:

- [ ] Python version is 3.8 or higher
- [ ] Virtual environment is activated
- [ ] All dependencies are installed correctly
- [ ] Model weights file exists and is not corrupted
- [ ] GPU drivers are up to date (if using GPU)
- [ ] CUDA version matches PyTorch CUDA version
- [ ] Sufficient RAM/VRAM available
- [ ] Test with sample images first before custom data

**Still having issues? [Open an issue](https://github.com/naga-narala/RipCatch/issues) with:**
- Error message (full traceback)
- System information (OS, Python version, GPU)
- Steps to reproduce
- Screenshots if applicable

---

## ✅ Quick Start Checklist

- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] PyTorch and CUDA working
- [ ] Model weights downloaded
- [ ] Sample image detection successful
- [ ] Sample video processing successful
- [ ] Understanding of basic usage

**Congratulations! You're ready to use RipCatch! 🎉**

---

<div align="center">

**🌊 Happy Detecting! 🌊**

**[⬆ Back to Top](#-ripcatch---quick-start-guide)**

*Questions? Check [README.md](README.md) or open an [issue](https://github.com/naga-narala/RipCatch/issues)*

</div>

