---
title: RipCatch v2.0 - Rip Current Detection
emoji: 🌊
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
tags:
  - yolov8
  - computer-vision
  - object-detection
  - pytorch
  - beach-safety
  - rip-current
  - oceanography
  - safety
  - deep-learning
---

# 🌊 RipCatch v2.0 - Advanced Rip Current Detection System

[![GitHub](https://img.shields.io/badge/GitHub-naga--narala/RipCatch-blue?logo=github)](https://github.com/naga-narala/RipCatch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

> **AI-powered single-stage rip current detection system to enhance beach safety and save lives.**

## 🎯 What is RipCatch?

RipCatch v2.0 is a state-of-the-art AI system that automatically detects **rip currents** in beach images and videos. Rip currents are responsible for approximately **100 deaths annually** in the United States alone. This tool helps identify these dangerous ocean currents to improve beach safety.

## 🚀 Key Features

- **High Accuracy**: 88.64% mAP@50 with 89.03% precision and 89.51% recall
- **Single-Stage Detection**: Unified YOLOv8m architecture - no separate beach classifier needed
- **Real-Time Processing**: 10-15 FPS on GPU, suitable for live surveillance
- **Multi-Modal Input**: Supports both images and videos
- **Production Ready**: Trained on 16,907 beach images with advanced ML techniques

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | **88.64%** |
| mAP@50-95 | **61.45%** |
| Precision | **89.03%** |
| Recall | **89.51%** |
| F1-Score | **89.27%** |

## 🛠️ Technical Details

- **Architecture**: YOLOv8m (Medium variant, 25M parameters)
- **Training Dataset**: 16,907 images (14,436 train / 1,804 val / 667 test)
- **Image Resolution**: 640×640 pixels
- **Training Hardware**: NVIDIA RTX 3080 (10GB VRAM)
- **Training Time**: 4-5 hours with early stopping
- **Framework**: PyTorch + Ultralytics YOLOv8

### Advanced Training Features

- ✅ Gradient accumulation (effective batch size 64)
- ✅ Early stopping (patience 25 epochs)
- ✅ Optimized learning rate schedule (0.0007 → 0.005)
- ✅ Strong regularization (weight decay, dropout, label smoothing)
- ✅ Advanced augmentation (mosaic, mixup, copy-paste, auto-augment)

## 🎨 How to Use

### Image Detection
1. Upload a beach image
2. Adjust confidence threshold (optional)
3. Click "Detect Rip Currents"
4. View results with bounding boxes and confidence scores

### Video Detection
1. Upload a beach video
2. Configure detection settings (optional)
3. Click "Process Video"
4. Review annotated video with detection statistics

## ⚠️ Safety Notice

This AI system is a **supplementary tool** and should be used alongside:
- Trained lifeguards
- Official beach safety warnings
- Local knowledge and experience

**Always exercise caution near ocean waters!**

## 🆕 What's New in v2.0

| Feature | v1.1 | v2.0 | Improvement |
|---------|------|------|-------------|
| **Architecture** | Two-stage (classifier + detector) | Single-stage YOLOv8m | 50% simpler |
| **Performance** | ~85% mAP@50 | **88.64% mAP@50** | +3.64% |
| **Inference Speed** | Sequential (2 passes) | Single pass | **2× faster** |
| **Deployment** | Complex (2 models) | Simple (1 model) | Easier to maintain |

## 📖 Citation

If you use RipCatch in your research or project, please cite:

```bibtex
@software{ripcatch2024,
  author = {Sravan Kumar},
  title = {RipCatch v2.0: Advanced Rip Current Detection System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/naga-narala/RipCatch}
}
```

## 🔗 Links

- **GitHub Repository**: [github.com/naga-narala/RipCatch](https://github.com/naga-narala/RipCatch)
- **Full Documentation**: See repository for detailed setup and training guides
- **License**: MIT License
- **Contact**: sravankumar.nnv@gmail.com

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 framework
- **PyTorch**: Deep learning framework
- **Gradio**: ML demo interface
- **Hugging Face**: Model hosting and deployment
- **Beach safety organizations** worldwide for raising awareness

---

<div align="center">

**Made with ❤️ for beach safety | © 2024-2026 Sravan Kumar**

</div>
