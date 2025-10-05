# Rip Current Detection - Local RTX 3080 Setup Guide

## 🔄 Changes Made for Local Environment

### 📂 Path Updates
- **Paperspace paths** (`/datasets`, `/notebooks`) → **Local Windows paths**
- **Beach data**: `A:\5_projects\rip_current_project\beach_data\beach_data`
- **Rip data**: `A:\5_projects\rip_current_project\rip-currents\rip-currents`
- **Working directory**: `A:\5_projects\rip_current_project`

### 🚀 RTX 3080 Optimizations
- **Beach classifier batch size**: 8 → 32 (4x increase for 12GB VRAM)
- **Rip detector batch size**: 8 → 16 (2x increase)
- **Image resolution**: Enhanced for better accuracy
- **Worker processes**: Increased to 4 for better CPU utilization
- **Mixed precision training**: Enabled AMP for RTX 3080

### 🧹 Removed Paperspace-Specific Code
- Removed dataset extraction from mounted storage
- Removed Paperspace environment references
- Updated all Unix paths to Windows paths
- Updated GPU memory references for RTX 3080

### ⚙️ Configuration Updates
- Added CUDA availability checks
- Added RTX 3080 detection and optimization
- Updated model save paths to local directories
- Enhanced error handling for local environment

## 🎯 Current Status

### ✅ Updated Sections
1. **Dataset setup and verification** - Now uses local paths
2. **Environment setup** - RTX 3080 optimized
3. **Beach classifier training** - Batch size optimized for RTX 3080
4. **Rip detector training** - Enhanced for local GPU
5. **Testing functions** - Updated paths and improved functionality

### 📋 Key Variables Set
```python
PROJECT_ROOT = r"A:\5_projects\rip_current_project"
BEACH_DATA_PATH = r"A:\5_projects\rip_current_project\beach_data\beach_data"
RIP_DATA_PATH = r"A:\5_projects\rip_current_project\rip-currents\rip-currents"
```

### 🔥 RTX 3080 Settings
- **Beach Classifier**: 32 batch size, 320px images, 50 epochs
- **Rip Detector**: 16 batch size, 832px images, 150 epochs
- **Mixed Precision**: Enabled for faster training
- **Workers**: 4 CPU workers for data loading

## 🚀 Next Steps

1. **Verify CUDA**: Run `python test_cuda.py` to check GPU setup
2. **Start Training**: Run the notebook cells in order
3. **Add Test Images**: Create test images in `PROJECT_ROOT/test_images/`
4. **Monitor Training**: Watch GPU utilization and adjust batch sizes if needed

## 💡 Batch Size Recommendations

### If you encounter memory issues:
- **Beach Classifier**: Reduce from 32 → 16 → 8
- **Rip Detector**: Reduce from 16 → 8 → 4
- **Image Size**: Reduce resolution if needed

### To maximize RTX 3080 usage:
- Monitor GPU memory with `nvidia-smi`
- Increase batch size until ~10-11GB memory usage
- Use mixed precision (AMP) for efficiency

## 🔧 Troubleshooting

### CUDA Issues:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Path Issues:
- Ensure datasets are in correct folders
- Check Windows path separators (`\` vs `/`)
- Verify folder permissions

### Memory Issues:
- Reduce batch sizes
- Close other GPU applications
- Monitor with Task Manager/nvidia-smi

## 📊 Expected Performance

With RTX 3080 optimizations:
- **Training Speed**: 2-3x faster than original settings
- **Memory Usage**: ~8-10GB VRAM during training
- **Batch Processing**: Significantly improved throughput
- **Model Accuracy**: Enhanced due to larger batch sizes and higher resolution