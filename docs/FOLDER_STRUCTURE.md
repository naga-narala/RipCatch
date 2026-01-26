# 📁 RipCatch - Folder Structure Documentation

Complete guide to the RipCatch repository organization and file purposes.

---

## 📊 Repository Overview

```
RipCatch/                        # Root directory
├── 📂 RipCatch-v1.0/            # Version 1.0 (Deprecated)
├── 📂 RipCatch-v1.1/            # Version 1.1 (Deprecated)
├── 📂 RipCatch-v2.0/            # ⭐ Version 2.0 (Current Production)
├── 📂 Testing/                  # Test images and videos
├── 📄 Documentation Files       # README, guides, etc.
└── 📄 Configuration Files       # requirements.txt, setup.py, etc.
```

**Total Size**: ~50 MB (excluding datasets and model weights)  
**Dataset Size**: ~7 GB (download separately)  
**Model Weights**: ~82 MB per version

---

## 🗂️ Detailed Structure

### Root Directory

```
RipCatch/
├── README.md                    # 📘 Main project documentation
├── QUICK_START.md               # 🚀 5-minute setup guide
├── CONTRIBUTING.md              # 🤝 Contribution guidelines
├── CHANGELOG.md                 # 📝 Version history
├── FOLDER_STRUCTURE.md          # 📁 This file
├── LICENSE                      # ⚖️ MIT License
├── requirements.txt             # 📦 Python dependencies
├── environment.yml              # 🐍 Conda environment spec
├── setup.py                     # 📦 Package installation script
├── pyproject.toml               # 🔧 Modern Python config
├── pyrightconfig.json           # 🔍 Type checking config
├── .gitignore                   # 🚫 Git ignore rules
├── .pre-commit-config.yaml      # 🪝 Pre-commit hooks (optional)
│
├── 📂 RipCatch-v1.0/            # Version 1.0 directory
├── 📂 RipCatch-v1.1/            # Version 1.1 directory
├── 📂 RipCatch-v2.0/            # ⭐ Version 2.0 directory
└── 📂 Testing/                  # Test data directory
```

---

## 📂 RipCatch-v2.0 (Current Production)

### Overview

**Status**: Production-ready  
**Performance**: 88.64% mAP@50  
**Model**: YOLOv8m single-stage detection  
**Last Updated**: October 2025

### Structure

```
RipCatch-v2.0/
│
├── 📓 RipCatch-v2.0.ipynb       # 🔥 Main training notebook
│   ├── Cell 1-2: Environment setup
│   ├── Cell 3-4: Dataset validation
│   ├── Cell 5-6: Training configuration and execution
│   ├── Cell 7-9: Evaluation and testing
│   └── Total: ~15 cells, ~4-5 hours runtime
│
├── 📂 Datasets/                 # Training data (gitignored)
│   ├── rip_dataset.zip          # Compressed dataset (7 GB)
│   └── rip_dataset/             # Extracted dataset
│       ├── data.yaml            # Dataset configuration
│       ├── train/               # Training set (14,436 images)
│       │   ├── images/
│       │   └── labels/
│       ├── valid/               # Validation set (1,804 images)
│       │   ├── images/
│       │   └── labels/
│       └── test/                # Test set (667 images)
│           ├── images/
│           └── labels/
│
├── 📂 Model/                    # Trained model files
│   ├── args.yaml                # Training hyperparameters
│   └── weights/                 # Model checkpoints
│       ├── best.pt              # 🔥 Best model (88.64% mAP@50)
│       ├── last.pt              # Last training checkpoint
│       ├── epoch0.pt            # Initial checkpoint
│       ├── epoch20.pt           # Checkpoint at epoch 20
│       ├── epoch40.pt           # Checkpoint at epoch 40
│       ├── epoch60.pt           # Checkpoint at epoch 60
│       ├── epoch80.pt           # Checkpoint at epoch 80
│       ├── epoch100.pt          # Checkpoint at epoch 100
│       ├── epoch120.pt          # Checkpoint at epoch 120
│       ├── epoch140.pt          # Checkpoint at epoch 140
│       ├── epoch160.pt          # Checkpoint at epoch 160
│       └── epoch180.pt          # Checkpoint at epoch 180
│
├── 📂 Results/                  # Training and evaluation results
│   ├── evaluation_results.json  # Validation metrics
│   ├── results.csv              # Per-epoch training metrics
│   ├── inference_results.png    # Sample detection visualization
│   ├── video_test_1_output.mp4  # Annotated test video 1
│   └── video_test_2_output.mp4  # Annotated test video 2
│
└── 📂 Documentation/            # Version-specific docs
    ├── NOTEBOOK_PLAN.md         # Notebook design and structure
    └── TRAINING_SUMMARY_REPORT.md  # 📊 Detailed training analysis
```

### File Purposes

#### Key Files

| File | Purpose | Size | Critical? |
|------|---------|------|-----------|
| `RipCatch-v2.0.ipynb` | Training and evaluation notebook | ~500 KB | ✅ Yes |
| `Model/weights/best.pt` | Best trained model | 82 MB | ✅ Yes |
| `Model/args.yaml` | Training configuration | 5 KB | ✅ Yes |
| `Results/evaluation_results.json` | Performance metrics | 1 KB | ⚠️ Important |
| `Documentation/TRAINING_SUMMARY_REPORT.md` | Training analysis | 50 KB | ⚠️ Important |

#### Dataset Files (Not in Git)

| File | Contents | Size | Download |
|------|----------|------|----------|
| `Datasets/rip_dataset.zip` | Compressed dataset | 7 GB | [Link](#) |
| `Datasets/rip_dataset/` | Extracted dataset | 7 GB | Extract zip |
| `Datasets/rip_dataset/data.yaml` | Dataset config | 1 KB | In dataset |

---

## 📂 RipCatch-v1.1 (Deprecated)

### Overview

**Status**: Deprecated (use v2.0 instead)  
**Approach**: Two-stage detection (beach classifier + rip detector)  
**Performance**: ~85% mAP@50  
**Last Updated**: September 2025

### Structure

```
RipCatch-v1.1/
│
├── 📓 RipCatch-v1.1.ipynb       # Training notebook (two-stage)
│
├── 📂 Datasets/                 # Training data
│   ├── beach_data.zip           # Beach classification dataset
│   └── rip-currents.zip         # Rip current detection dataset
│
├── 📂 models/                   # Trained models
│   ├── beach_classifier_best.pt # Beach scene classifier (95% accuracy)
│   └── rip_detector_best.pt     # Rip current detector (85% mAP@50)
│
└── 📂 Documentation/
    └── LOCAL_SETUP_GUIDE.md     # RTX 3080 setup instructions
```

### Why Deprecated?

- ❌ Two-stage approach is more complex
- ❌ Lower overall performance (85% vs 88.64%)
- ❌ Slower inference (two model passes)
- ✅ Superseded by v2.0 single-stage approach

---

## 📂 RipCatch-v1.0 (Initial Prototype)

### Overview

**Status**: Deprecated (historical reference only)  
**Approach**: Basic YOLOv8n implementation  
**Performance**: ~79% mAP@50 (peak)  
**Last Updated**: August 2025

### Structure

```
RipCatch-v1.0/
├── 📂 Datasets/                 # Initial dataset (small)
├── 📂 Documentation/            # Early documentation
└── 📂 models/                   # Initial model checkpoints
```

### Why Deprecated?

- ❌ Poor performance (79% mAP@50)
- ❌ Early convergence issues
- ❌ High overfitting
- ❌ Limited dataset
- ✅ Proof of concept only

---

## 📂 Testing Directory

### Overview

Test images and videos for validation and demonstration.

### Structure

```
Testing/
│
├── 📂 beach/                    # Beach scenes (23 images)
│   ├── test_1.jpg               # Sample beach image 1
│   ├── test_2.jpg               # Sample beach image 2
│   ├── ...                      # More test images
│   └── test_10.jpg              # Sample beach image 10
│
├── 📂 Mixed/                    # Beach + rip currents (34 images)
│   ├── RIP1.webp                # Rip current example 1
│   ├── RIP2.jpg                 # Rip current example 2
│   ├── rip-current.jpg          # Various rip current scenes
│   ├── Beach-With-Rip-Current-Small.jpg
│   └── ...                      # More mixed examples
│
├── 📂 real_time/                # Real-time test cases (4 images)
│   ├── IMG_4326.PNG             # Real-time test 1
│   ├── IMG_4327.PNG             # Real-time test 2
│   ├── IMG_4328.PNG             # Real-time test 3
│   └── IMG_4329.PNG             # Real-time test 4
│
└── 📂 videos/                   # Test videos (2 videos)
    ├── video_test_1.mp4         # Beach surveillance footage 1
    └── video_test_2.mp4         # Beach surveillance footage 2
```

### Usage

```python
from ultralytics import YOLO

model = YOLO('RipCatch-v2.0/Model/weights/best.pt')

# Test on beach scenes
results = model('Testing/beach/test_1.jpg')

# Test on rip current examples
results = model('Testing/Mixed/RIP1.webp')

# Test on videos
results = model('Testing/videos/video_test_1.mp4', save=True)
```

---

## 📄 Configuration Files

### Python Environment Files

#### requirements.txt

```
Purpose: Python package dependencies
Size: ~5 KB
Usage: pip install -r requirements.txt
```

**Key Dependencies:**
- `torch>=2.0.0` - PyTorch framework
- `ultralytics>=8.0.0` - YOLOv8 implementation
- `opencv-python>=4.8.0` - Image/video processing
- `numpy`, `pandas`, `matplotlib` - Data science tools

---

#### environment.yml

```
Purpose: Conda environment specification
Size: ~3 KB
Usage: conda env create -f environment.yml
```

**Includes:**
- Python 3.10
- PyTorch with CUDA 11.8
- All project dependencies

---

#### setup.py

```
Purpose: Package installation script
Size: ~2 KB
Usage: pip install -e .
```

**Enables:**
- Installation as Python package
- Development mode installation
- Dependency management

---

#### pyproject.toml

```
Purpose: Modern Python project configuration
Size: ~1 KB
Usage: Automatic (used by build tools)
```

**Defines:**
- Project metadata
- Build system requirements
- Tool configurations (black, isort, mypy)

---

#### pyrightconfig.json

```
Purpose: Python type checking configuration
Size: <1 KB
Usage: Automatic (used by Pyright/Pylance)
```

**Configures:**
- Type checking strictness
- Python version
- Exclude patterns

---

### Git Files

#### .gitignore

```
Purpose: Exclude files from version control
Size: ~8 KB
Key Exclusions:
  - Datasets/ (large files)
  - *.pt, *.pth (model weights)
  - *.mp4 (videos)
  - __pycache__/ (Python cache)
  - venv/, env/ (virtual environments)
```

---

#### .pre-commit-config.yaml (Optional)

```
Purpose: Automated code quality checks
Size: ~1 KB
Usage: pre-commit install
Checks:
  - Code formatting (black, isort)
  - Linting (flake8)
  - Trailing whitespace
  - File size limits
```

---

### Documentation Files

#### README.md

```
Purpose: Main project documentation
Size: ~30 KB
Sections:
  - Project overview
  - Installation guide
  - Usage examples
  - Performance metrics
  - Contributing info
```

---

#### QUICK_START.md

```
Purpose: 5-minute setup guide
Size: ~12 KB
Contents:
  - Fast installation steps
  - Quick testing examples
  - Troubleshooting
```

---

#### CONTRIBUTING.md

```
Purpose: Contribution guidelines
Size: ~15 KB
Contents:
  - Code of conduct
  - Development setup
  - Coding standards
  - PR process
```

---

#### CHANGELOG.md

```
Purpose: Version history
Size: ~10 KB
Contents:
  - Release notes
  - Feature additions
  - Bug fixes
  - Breaking changes
```

---

#### LICENSE

```
Purpose: MIT License
Size: ~1 KB
Permits:
  - Commercial use
  - Modification
  - Distribution
  - Private use
```

---

## 📊 File Size Summary

### Git-Tracked Files

| Category | Files | Total Size |
|----------|-------|------------|
| Documentation | 6 files | ~70 KB |
| Configuration | 7 files | ~20 KB |
| Notebooks | 3 files | ~1.5 MB |
| Test Images | 60+ files | ~45 MB |
| **Total (Git)** | **~80 files** | **~50 MB** |

### Git-Ignored Files (Download Separately)

| Category | Files | Total Size |
|----------|-------|------------|
| Datasets | 16,907 images | ~7 GB |
| Model Weights | 15 files | ~1.2 GB |
| Output Videos | 2-4 files | ~50-100 MB |
| **Total (Ignored)** | **~17,000 files** | **~8-9 GB** |

### Full Repository

| Type | Size |
|------|------|
| Git repository | ~50 MB |
| With datasets | ~7 GB |
| With models | ~1.2 GB |
| With outputs | ~100 MB |
| **Complete** | **~8-9 GB** |

---

## 🗺️ Access Patterns

### For End Users

**What you need:**
1. ✅ Git clone repository (~50 MB)
2. ✅ Download v2.0 model weights (~82 MB)
3. ❌ No need for datasets (unless training)
4. ❌ No need for old versions

**Total**: ~130 MB

---

### For Developers/Contributors

**What you need:**
1. ✅ Git clone repository
2. ✅ Download v2.0 model weights
3. ✅ Set up development environment
4. ⚠️ Download dataset only if training/testing
5. ❌ Old versions optional (reference only)

**Total**: ~130 MB (without dataset) or ~7.1 GB (with dataset)

---

### For Researchers

**What you need:**
1. ✅ Full repository
2. ✅ All model weights (v1.0, v1.1, v2.0)
3. ✅ Complete dataset
4. ✅ Training notebooks
5. ✅ Documentation

**Total**: ~8-9 GB

---

## 📦 Storage Optimization

### Reducing Repository Size

1. **Shallow Clone** (if you don't need history):
```bash
git clone --depth 1 https://github.com/naga-narala/RipCatch.git
```
Saves: ~20% of clone size

2. **Sparse Checkout** (specific folders only):
```bash
git clone --filter=blob:none --sparse https://github.com/naga-narala/RipCatch.git
cd RipCatch
git sparse-checkout set RipCatch-v2.0 Testing
```
Saves: ~50% for specific version

3. **Download Model Weights Separately**:
Instead of storing in Git, use:
- GitHub Releases
- External hosting (Google Drive, Dropbox)
- Git LFS (Large File Storage)

---

## 🔄 Migration Between Versions

### From v1.1 to v2.0

**Changes:**
- Two-stage → Single-stage detection
- Separate datasets → Unified dataset
- Two models → One model

**Migration Steps:**
1. Switch to v2.0 directory
2. Download unified dataset
3. Use single model file (best.pt)
4. Update inference code (remove two-stage logic)

---

### From v1.0 to v2.0

**Changes:**
- YOLOv8n → YOLOv8m
- Basic config → Advanced training
- Small dataset → Large dataset

**Migration Steps:**
1. Switch to v2.0 directory
2. Download new dataset (16,907 images)
3. Use new model (best.pt)
4. Update training scripts

---

## 🎯 Directory Purpose Quick Reference

| Directory | Purpose | Size | Git Tracked? |
|-----------|---------|------|--------------|
| `RipCatch-v2.0/` | **Current production model** | ~82 MB | Partial |
| `RipCatch-v1.1/` | Legacy two-stage model | ~160 MB | Partial |
| `RipCatch-v1.0/` | Initial prototype | ~50 MB | Partial |
| `Testing/` | Test images and videos | ~45 MB | ✅ Yes |
| `Datasets/` | Training data | 7 GB | ❌ No |
| `Model/weights/` | Model checkpoints | 1.2 GB | ❌ No |
| Documentation | Guides and READMEs | ~70 KB | ✅ Yes |
| Configuration | Setup files | ~20 KB | ✅ Yes |

---

## 📞 Questions?

**Need help navigating the repository?**

- 💬 [GitHub Discussions](https://github.com/naga-narala/RipCatch/discussions)
- 🐛 [Report Issues](https://github.com/naga-narala/RipCatch/issues)
- 📧 [Email](mailto:sravankumar.nnv@gmail.com)

---

<div align="center">

**📁 Repository Structure Documentation**

*Last Updated: October 2025*

**[⬆ Back to Top](#-ripcatch---folder-structure-documentation)**

</div>

