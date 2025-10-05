# Changelog

All notable changes to the RipCatch project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2025-10-05

### 🎉 Major Release - Production-Ready Model

**Status**: Production-ready  
**Performance**: 88.64% mAP@50, 89.03% Precision, 89.51% Recall

### Added
- ✨ **YOLOv8m-based single-stage detection** - Simplified from two-stage approach
- ✨ **Advanced training pipeline** with gradient accumulation and mixed precision
- ✨ **Comprehensive evaluation suite** with multiple metrics
- ✨ **Video inference support** with real-time processing capabilities
- ✨ **Model export functionality** supporting ONNX, TFLite, TorchScript
- ✨ **Jupyter notebook workflow** for reproducible training
- ✨ **Test dataset** with 60+ images and 2 videos
- ✨ **Detailed documentation** including training reports and setup guides
- ✨ **Results visualization** with inference examples

### Changed
- 🔄 **Model architecture**: Switched from two-stage to single-stage detection
- 🔄 **Training approach**: Advanced hyperparameter optimization
  - Learning rate: 0.0007 → 0.005 (gentler decay)
  - Batch size: 16 physical → 64 effective (gradient accumulation)
  - Dropout: 0.15 for better regularization
  - Early stopping: Patience 25 epochs
- 🔄 **Dataset**: Consolidated to 16,907 images (14,436 train / 1,804 val / 667 test)
- 🔄 **Image resolution**: 640×640 optimized for RTX 3080 (10GB VRAM)
- 🔄 **Augmentation strategy**: Enhanced with Mosaic, MixUp, Copy-Paste, RandAugment

### Improved
- ⚡ **Inference speed**: 10-15 FPS on RTX 3080 GPU
- ⚡ **Training efficiency**: 4-5 hours (65-70% faster with early stopping)
- ⚡ **Model generalization**: F1-score 89.27% (well-balanced)
- ⚡ **Localization accuracy**: mAP@50-95 61.45%

### Fixed
- 🐛 Fixed early convergence issue from v1.0 (peaked at epoch 29)
- 🐛 Resolved overfitting with stronger regularization (3x weight decay)
- 🐛 Addressed high loss variance with gradient accumulation
- 🐛 Improved IoU quality with optimized loss functions

### Performance Comparison

| Metric | v1.0 | v1.1 | v2.0 | Improvement |
|--------|------|------|------|-------------|
| mAP@50 | 79.0% | 85.0% | 88.64% | +9.64% |
| Precision | ~85% | ~88% | 89.03% | +4.03% |
| Recall | ~82% | ~84% | 89.51% | +7.51% |
| F1-Score | ~83% | ~86% | 89.27% | +6.27% |
| Training Time | 7.15h | 6h | 4-5h | -30% |
| Inference Speed | 8 FPS | 10 FPS | 12-15 FPS | +50% |

### Technical Details

**Training Configuration:**
```yaml
Model: YOLOv8m
Epochs: 200 (early stopping at ~70)
Batch Size: 16 (effective 64 via gradient accumulation)
Image Size: 640×640
Optimizer: AdamW
Learning Rate: 0.0007 → 0.005
Weight Decay: 0.0015
Dropout: 0.15
Augmentation: Mosaic 1.0, MixUp 0.2, Copy-Paste 0.3
```

### Documentation
- 📚 Complete training summary report with 15 performance insights
- 📚 Notebook plan with detailed cell-by-cell specifications
- 📚 Local setup guide for RTX 3080 optimization
- 📚 Model architecture and hyperparameter documentation

### Known Issues
- ⚠️ Performance 4.16% below target benchmark (92.8%)
- ⚠️ Limited to 640px resolution due to VRAM constraints
- ⚠️ Test set relatively small (3.9% of total dataset)

### Migration Guide

**From v1.1 to v2.0:**

```python
# v1.1 (Two-stage)
beach_model = YOLO('beach_classifier_best.pt')
rip_model = YOLO('rip_detector_best.pt')
beach_results = beach_model(image)
if beach_detected:
    rip_results = rip_model(image)

# v2.0 (Single-stage)
model = YOLO('RipCatch-v2.0/Model/weights/best.pt')
results = model(image)  # Direct detection
```

**Dataset Changes:**
- Old: Separate beach and rip current datasets
- New: Unified rip current dataset with 16,907 images
- Migration: Re-annotation not required, use provided unified dataset

---

## [1.1.0] - 2025-09-15

### Added
- ✨ **Two-stage detection pipeline**:
  - Stage 1: Beach scene classifier (95% accuracy)
  - Stage 2: Rip current detector (85% mAP@50)
- ✨ **Separate model training** for each stage
- ✨ **Beach classification** to filter non-beach images
- ✨ **RTX 3080 optimizations** with increased batch sizes
- ✨ **Local setup documentation** for Windows environment

### Changed
- 🔄 **Architecture**: Split into beach classifier + rip detector
- 🔄 **Batch sizes**: Increased for RTX 3080 (32 for classifier, 16 for detector)
- 🔄 **Paths**: Updated from Paperspace to local Windows paths
- 🔄 **Dataset structure**: Separate beach_data and rip-currents folders

### Improved
- ⚡ **Accuracy**: Beach classifier 95%, Rip detector 85% mAP@50
- ⚡ **Training speed**: 2-3x faster with RTX 3080 optimizations
- ⚡ **Memory efficiency**: Mixed precision training (AMP)

### Fixed
- 🐛 Removed Paperspace-specific code
- 🐛 Fixed Unix path issues on Windows
- 🐛 Resolved GPU memory constraints with optimized batch sizes

### Deprecated
- ⚠️ Two-stage approach (superseded by v2.0 single-stage)
- ⚠️ Separate models for beach classification and rip detection

---

## [1.0.0] - 2025-08-20

### 🎉 Initial Release - Proof of Concept

**Status**: Prototype  
**Performance**: ~79% mAP@50 (peak at epoch 29)

### Added
- ✨ **Basic YOLOv8n implementation** for rip current detection
- ✨ **Initial dataset** collection and annotation
- ✨ **Training pipeline** with standard hyperparameters
- ✨ **Simple inference scripts** for images
- ✨ **Project structure** and basic documentation

### Features
- Single-stage YOLOv8n detection
- Standard YOLOv8 training configuration
- Basic evaluation metrics
- Sample test images

### Limitations
- ⚠️ Model peaked early (epoch 29) then declined
- ⚠️ High overfitting (train-val gap 0.453)
- ⚠️ Limited dataset size
- ⚠️ No video inference support
- ⚠️ Suboptimal hyperparameters

### Performance
- mAP@50: 79.0% (peak) → 87.87% (final, but declining)
- Training time: 7.15 hours (150 epochs)
- Inference speed: ~8 FPS

### Lessons Learned
- Early stopping needed (wasted 102 epochs after convergence)
- Learning rate too aggressive
- Insufficient regularization
- Need for gradient accumulation

---

## [Unreleased] - Future Roadmap

### Planned for v2.1 (Q4 2025)
- [ ] **Real-time video stream optimization** with temporal analysis
- [ ] **Multi-camera tracking** and detection fusion
- [ ] **Alert system integration** for automated warnings
- [ ] **Mobile app** (iOS and Android)
- [ ] **Web interface** for easy deployment
- [ ] **API endpoints** for third-party integration
- [ ] **Enhanced documentation** with video tutorials

### Planned for v2.2
- [ ] **Test-Time Augmentation (TTA)** for +1.5-2.5% accuracy
- [ ] **Model ensemble** for improved robustness
- [ ] **Weather condition integration** (wind, tide data)
- [ ] **Confidence calibration** for better uncertainty estimation

### Planned for v3.0 (2026)
- [ ] **Depth estimation** from monocular images
- [ ] **3D water flow visualization**
- [ ] **Crowd density analysis** for risk assessment
- [ ] **Satellite imagery support** for wide-area monitoring
- [ ] **Multi-language support** (Spanish, Portuguese, French)
- [ ] **Explainable AI (XAI)** visualization of detection reasoning

### Long-term Vision
- [ ] **Drone-based detection** for aerial surveillance
- [ ] **Global beach monitoring network**
- [ ] **Integration with emergency services**
- [ ] **Public API** for researchers and organizations
- [ ] **Real-time hazard prediction** using historical data
- [ ] **Automated lifeguard assistance system**

---

## Version History Summary

| Version | Release Date | Key Feature | Performance | Status |
|---------|--------------|-------------|-------------|--------|
| **v2.0.0** | 2025-10-05 | Single-stage YOLOv8m | 88.64% mAP@50 | ✅ Current |
| v1.1.0 | 2025-09-15 | Two-stage detection | 85.0% mAP@50 | ⚠️ Deprecated |
| v1.0.0 | 2025-08-20 | Initial prototype | 79.0% mAP@50 | ⚠️ Deprecated |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute to this project.

---

## Support

For questions or issues:
- 🐛 [Report Bugs](https://github.com/naga-narala/RipCatch/issues)
- 💬 [Discussions](https://github.com/naga-narala/RipCatch/discussions)
- 📧 [Email Support](mailto:sravankumar.nnv@gmail.com)

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**🌊 RipCatch - Making Beaches Safer with AI 🌊**

[Homepage](https://github.com/naga-narala/RipCatch) • [Documentation](https://github.com/naga-narala/RipCatch/wiki) • [Report Bug](https://github.com/naga-narala/RipCatch/issues) • [Request Feature](https://github.com/naga-narala/RipCatch/issues)

</div>

