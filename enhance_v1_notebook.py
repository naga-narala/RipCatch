#!/usr/bin/env python3
"""
Script to enhance RipCatch v1.1 notebook with detailed descriptions.
Adds comprehensive markdown cells before each code cell to explain what they do.
"""

import json
from pathlib import Path

# Markdown descriptions for each code cell
MARKDOWN_DESCRIPTIONS = {
    0: """## 1️⃣ Environment Setup & GPU Detection

**What this cell does:**
- Detects available GPU (CUDA) and displays GPU specifications
- Verifies RTX 3080 setup with CUDA support
- Checks PyTorch installation and CUDA availability  
- Displays GPU memory (VRAM) information
- Sets up project paths and directories
- Configures matplotlib for inline plotting

**Expected execution time:** <10 seconds

**Expected output:**
- PyTorch version (should be 2.0+)
- CUDA availability: ✅ True
- GPU name: NVIDIA GeForce RTX 3080 (or your GPU)
- GPU memory: ~10-12 GB VRAM
- Python version and system information

**Troubleshooting:**
- If CUDA shows as unavailable, see the CUDA fix cells below
- Ensure NVIDIA drivers are up to date (version 520+)
- Verify PyTorch was installed with CUDA support

**Key improvements from Paperspace:**
- ✅ Updated paths for local Windows environment
- ✅ Optimized for RTX 3080 (12GB VRAM)
- ✅ Added comprehensive GPU detection
- ✅ Removed cloud-specific configurations
""",

    2: """## 2️⃣ Conda PyTorch CUDA Installation (Optional)

**What this cell does:**
- Provides automated script to install PyTorch with CUDA support via Conda
- Uninstalls existing PyTorch installation (if any)
- Installs PyTorch with CUDA 11.8 (recommended for RTX 3080)
- Verifies installation success
- **Only run this if you need to install/reinstall PyTorch**

**Expected execution time:** 5-15 minutes (depending on internet speed)

**When to run this cell:**
- If Cell 1 (GPU Detection) shows CUDA is unavailable
- If you're seeing CUDA DLL errors
- If you need to switch CUDA versions
- Fresh Python environment setup

**When to SKIP this cell:**
- If Cell 1 shows CUDA is already working ✅
- If PyTorch is already installed correctly

**After running:**
- **MUST restart the Jupyter kernel** for changes to take effect
- Re-run Cell 1 to verify CUDA is now available

**Alternative installation methods:**
- See Cell 4 (Markdown) for pip-based installation
- Check official PyTorch website for latest instructions
""",

    3: """## 3️⃣ Local Environment Configuration & Path Setup

**What this cell does:**
- Defines all project paths for local Windows environment
- Sets up beach classification dataset path
- Sets up rip current detection dataset path
- Verifies that dataset directories exist
- Creates output directories for trained models
- Displays dataset statistics (image counts, folder structure)

**Expected execution time:** <5 seconds

**Critical paths to verify:**
```python
PROJECT_ROOT       = 'A:\\5_projects\\rip_current_project'
BEACH_DATA_PATH    = PROJECT_ROOT / 'beach_data' / 'beach_data'
RIP_DATA_PATH      = PROJECT_ROOT / 'rip-currents' / 'rip-currents'
```

**Expected output:**
- ✅ Project root exists
- ✅ Beach dataset path exists (with train/valid/test folders)
- ✅ Rip current dataset path exists (with images and labels)
- Dataset statistics (number of images in each split)

**Customization:**
- **UPDATE `PROJECT_ROOT`** to match your local directory
- Ensure dataset folders follow YOLO format:
  - `train/images/` and `train/labels/`
  - `valid/images/` and `valid/labels/`
  - `test/images/` and `test/labels/`

**Troubleshooting:**
- If paths don't exist, download datasets or update paths
- Use raw strings `r"path"` for Windows paths
- Verify folder permissions (read/write access)
""",

    6: """## 4️⃣ Dataset Structure Verification

**What this cell does:**
- Walks through beach dataset directory structure
- Walks through rip current dataset directory structure  
- Displays folder hierarchy (train/valid/test splits)
- Lists sample files in each directory
- Verifies folder naming conventions

**Expected execution time:** <5 seconds

**Expected output:**
- Complete directory tree for both datasets
- Folder names: `train`, `valid`, `test`
- Sub-folders: `images`, `labels`
- Sample file names from each folder

**What to look for:**
- ✅ All required folders present (train/valid/test)
- ✅ Both images/ and labels/ subfolders exist
- ✅ File naming is consistent (`.jpg`, `.png`, `.txt`)
- ✅ No empty folders

**Common issues:**
- Missing `labels` folder → Need to annotate images
- Missing `test` split → Can use validation set for testing
- Different folder names → Update paths or rename folders
""",

    7: """## 5️⃣ Dataset Statistics & Quality Analysis

**What this cell does:**
- Counts images and labels in each dataset split
- Verifies image-label pairing (same count = ✅)
- Calculates dataset split percentages
- Checks for missing labels or images
- Displays comprehensive statistics table

**Expected execution time:** <10 seconds

**Expected output:**
- Beach dataset statistics:
  - Train: X images, Y labels
  - Valid: X images, Y labels  
  - Test: X images, Y labels
- Rip current dataset statistics:
  - Similar breakdown
- Split percentages (should be ~70% train, 20% valid, 10% test)

**Quality checks:**
- ✅ Image count == Label count (critical!)
- ✅ Reasonable split ratios (70-80% train is good)
- ✅ Sufficient validation data (min 15-20%)
- ✅ Test set exists (even if small)

**Troubleshooting:**
- Image-label mismatch → Check for missing `.txt` files
- Very small datasets → May need more data or augmentation
- Unbalanced splits → Consider re-splitting data
""",

    9: """## 6️⃣ Required Packages Installation

**What this cell does:**
- Installs ultralytics (YOLOv8 framework)
- Installs OpenCV for video processing
- Installs Pillow for image handling
- Installs PyYAML for configuration files
- Installs matplotlib for visualization
- Verifies all installations are successful

**Expected execution time:** 1-3 minutes

**When to run this cell:**
- First time setup in a new environment
- After creating a new virtual environment
- If you get "ModuleNotFoundError" errors

**When to SKIP this cell:**
- If packages are already installed
- If you've run this cell before in the same environment

**Packages installed:**
- `ultralytics` - YOLOv8 implementation (main framework)
- `opencv-python` - Video/image processing
- `pillow` - Image manipulation
- `pyyaml` - YAML configuration parsing
- `matplotlib` - Plotting and visualization

**Verification:**
- Each package should show "Successfully installed" or "Requirement already satisfied"
- No error messages should appear

**Alternative:**
Run in terminal: `pip install -r requirements.txt`
""",

    11: """## 7️⃣ Stage 1: Beach Classification Model Training

**What this cell does:**
- Trains YOLOv8 classification model to identify beach images
- Distinguishes beach scenes from non-beach scenes
- Uses transfer learning from pretrained YOLOv8 model
- Implements checkpoint resume (continues from last saved point if interrupted)
- Optimized for RTX 3080 with 32 batch size

**Model architecture:** YOLOv8n-cls (classification variant)

**Training configuration:**
- **Epochs:** 50
- **Batch size:** 32 (optimized for 10-12GB VRAM)
- **Image size:** 320×320 pixels
- **Learning rate:** Auto (default 0.01)
- **Workers:** 4 (parallel data loading)
- **Mixed precision:** Enabled (AMP)

**Expected training time:** 15-30 minutes on RTX 3080

**Progress indicators:**
- Epoch counter (1/50, 2/50, etc.)
- Loss values (should decrease over time)
- Accuracy metrics (should increase)
- Training speed (images/second)

**Expected final metrics:**
- **Accuracy:** >90% (goal: 95%)
- **Precision:** >90%
- **Recall:** >85%

**Outputs:**
- Model saved to: `beach_classifier_best.pt`
- Training plots in results folder
- Metrics logged to console

**Checkpoint resume:**
- If training is interrupted, re-run this cell
- It will automatically resume from `beach_classifier_last.pt`

**GPU monitoring:**
- Use `nvidia-smi` in terminal to monitor VRAM usage
- Should use ~8-10GB during training
- If OOM errors occur, reduce batch size to 16 or 8

**Troubleshooting:**
- Out of memory → Reduce batch_size to 16 or 8
- Low accuracy → Train for more epochs (100+)
- Slow training → Check GPU utilization with `nvidia-smi`
""",

    13: """## 8️⃣ Stage 2: Rip Current Detection Model Training

**What this cell does:**
- Trains YOLOv8 object detection model to detect rip currents
- Identifies and localizes rip currents with bounding boxes
- Uses transfer learning from COCO-pretrained YOLOv8 model
- Implements checkpoint resume for interrupted training
- Optimized for RTX 3080 with 16 batch size

**Model architecture:** YOLOv8n (nano variant for object detection)

**Training configuration:**
- **Epochs:** 150 (with early stopping)
- **Batch size:** 16 (optimized for 10-12GB VRAM)
- **Image size:** 832×832 pixels (high resolution for better accuracy)
- **Learning rate:** 0.001 (conservative for stability)
- **Patience:** 25 epochs (early stopping if no improvement)
- **Workers:** 4 (parallel data loading)
- **Mixed precision:** Enabled (AMP)
- **Augmentation:** Enabled (mosaic, flip, rotation, etc.)

**Expected training time:** 2-4 hours on RTX 3080

**Progress indicators:**
- Epoch counter and ETA
- Box loss (localization accuracy)
- Class loss (classification accuracy)
- DFL loss (distribution focal loss)
- mAP@50 (mean Average Precision at 50% IoU)
- mAP@50-95 (strict accuracy metric)

**Target metrics:**
- **mAP@50:** >80% (goal: 85%)
- **Precision:** >85%
- **Recall:** >80%
- **mAP@50-95:** >55%

**Outputs:**
- Best model saved to: `rip_detector_best.pt`
- Last checkpoint: `rip_detector_last.pt`
- Training plots and metrics in results folder
- Confusion matrix
- Precision-Recall curves

**Checkpoint resume:**
- If training is interrupted, re-run this cell
- Automatically resumes from `rip_detector_last.pt`
- Training history is preserved

**GPU monitoring:**
- VRAM usage: ~9-11GB during training
- If OOM errors, reduce batch_size to 8 or image size to 640

**Training tips:**
- Monitor mAP@50 - should steadily increase
- If plateaus early, increase epochs or adjust learning rate
- Watch for overfitting (validation loss >> training loss)

**Troubleshooting:**
- Out of memory → Reduce batch_size to 8 or imgsz to 640
- Low mAP → Train longer (200+ epochs) or get more data
- Model not learning → Check dataset labels are correct
- Slow training → Verify GPU is being used (check nvidia-smi)
""",

    15: """## 9️⃣ Beach Classifier Testing & Evaluation

**What this cell does:**
- Loads the trained beach classifier model (`beach_classifier_best.pt`)
- Tests on sample beach and non-beach images
- Displays predictions with confidence scores
- Visualizes results in a grid layout
- Evaluates classification accuracy

**Expected execution time:** 10-30 seconds (depending on number of test images)

**What to prepare:**
- Test images in: `PROJECT_ROOT/test_images/beach/`
- Include both beach and non-beach images
- Supported formats: `.jpg`, `.png`, `.jpeg`, `.webp`

**Expected output:**
- Grid of test images with predictions
- Each image labeled as "Beach" or "Not Beach"
- Confidence scores (0-100%)
- Overall accuracy on test set

**Interpreting results:**
- **High confidence (>90%):** Model is very certain
- **Medium confidence (70-90%):** Reasonably confident
- **Low confidence (<70%):** Model is uncertain
- **Misclassifications:** Images to review

**Quality checks:**
- ✅ All beach images correctly classified
- ✅ Non-beach images correctly rejected
- ✅ Confidence scores are high (>85%)
- ⚠️ Edge cases (beach at night, aerial views) may have lower confidence

**Performance benchmarks:**
- Good model: >90% accuracy, >85% avg confidence
- Excellent model: >95% accuracy, >90% avg confidence

**Troubleshooting:**
- Low confidence → May need more training
- Misclassifications → Review those images, might be edge cases
- Model not loading → Check path to `beach_classifier_best.pt`
""",

    16: """## 🔟 Rip Current Detector Testing & Evaluation

**What this cell does:**
- Loads the trained rip current detector (`rip_detector_best.pt`)
- Tests on sample beach images with rip currents
- Draws bounding boxes around detected rip currents
- Displays confidence scores for each detection
- Visualizes results with annotations

**Expected execution time:** 30-60 seconds (depending on number of test images)

**What to prepare:**
- Test images in: `PROJECT_ROOT/test_images/rip/`
- Images should contain rip currents
- Supported formats: `.jpg`, `.png`, `.jpeg`, `.webp`

**Expected output:**
- Images with bounding boxes around rip currents
- Confidence score for each detection (0-100%)
- Detection count per image
- Visual comparison of predictions

**Interpreting results:**
- **Green boxes:** Detected rip currents
- **Confidence >70%:** Likely rip current
- **Confidence 50-70%:** Possible rip current (review)
- **Confidence <50%:** Uncertain (may be false positive)

**Quality metrics to check:**
- ✅ True positives: Correctly detected rip currents
- ❌ False positives: Boxes on non-rip areas
- ❌ False negatives: Missed rip currents (no box)
- ✅ Localization: Boxes accurately cover rip current area

**Performance benchmarks:**
- Good model: >75% precision, >70% recall
- Excellent model: >85% precision, >80% recall

**Common detection challenges:**
- Low contrast rip currents
- Foam/whitewash vs actual rips
- Shadows and glare
- Small or distant rips

**Troubleshooting:**
- No detections → Try lowering confidence threshold (conf=0.25)
- Too many false positives → Increase conf threshold (conf=0.50)
- Model not loading → Check path to `rip_detector_best.pt`
- Bounding boxes inaccurate → May need more training data
""",

    17: """## 1️⃣1️⃣ Individual Image Testing with Custom Confidence

**What this cell does:**
- Tests rip detector on a SINGLE image with custom settings
- Allows adjustment of confidence threshold
- Provides detailed detection information
- Shows larger visualization for detailed inspection
- Exports annotated image with bounding boxes

**Expected execution time:** 5-10 seconds per image

**Customization options:**
- **IMAGE_PATH:** Path to your test image
- **CONFIDENCE_THRESHOLD:** 0.0 to 1.0 (default 0.35)
  - Lower (0.25): More detections (may include false positives)
  - Higher (0.50+): Fewer but more confident detections
- **SAVE_OUTPUT:** True/False - Save annotated image

**Expected output:**
- Large display of annotated image
- Detection details table:
  - Bounding box coordinates (x1, y1, x2, y2)
  - Confidence score
  - Box area (pixels)
- Saved image with annotations (if enabled)

**Use cases:**
- **Fine-tuning confidence threshold:** Test different values
- **Detailed inspection:** Examine specific challenging images
- **Creating presentations:** Export high-quality annotated images
- **Debugging:** Understand why certain images fail/succeed

**Interpreting output:**
- **Multiple detections:** Rip might be large or multiple rips present
- **No detections:** Try lowering confidence or check image quality
- **Overlapping boxes:** May indicate uncertainty in localization

**Best practices:**
- Start with conf=0.35 (balanced)
- Lower to 0.25 for conservative detection
- Raise to 0.50 for high-confidence only
- Save outputs for reporting and analysis

**Troubleshooting:**
- "No detections" → Lower confidence threshold
- "Too many boxes" → Raise confidence threshold  
- "Poor localization" → Model may need more training
- "Image not found" → Check IMAGE_PATH is correct
""",

    18: """## 1️⃣2️⃣ Complete Two-Stage Pipeline: Beach + Rip Detection

**What this cell does:**
- Implements the FULL two-stage detection pipeline
- **Stage 1:** Beach classifier filters out non-beach images
- **Stage 2:** Rip detector analyzes beach images for rip currents
- Processes multiple images sequentially
- Displays results for each stage
- Provides comprehensive detection report

**Pipeline workflow:**
```
Input Image
    ↓
[Beach Classifier]
    ↓
Is Beach? → NO → Skip (show "Not a beach")
    ↓ YES
[Rip Current Detector]
    ↓
Detections → Show with bounding boxes + confidence
```

**Expected execution time:** 1-3 minutes (depending on number of images)

**What to prepare:**
- Mixed test set in: `PROJECT_ROOT/test_images/mixed/`
- Include: beach with rips, beach without rips, non-beach images
- This tests the complete system

**Expected output:**
For each image:
- **Non-beach images:** "Not a beach scene" message
- **Beach without rips:** "Beach detected, no rip currents"
- **Beach with rips:** Annotated image with bounding boxes

**Performance insights:**
- Precision improvement: Beach filter reduces false positives
- Processing efficiency: Skip non-beach images early
- Real-world simulation: Handles mixed input types

**Quality metrics:**
- ✅ Non-beach correctly filtered out
- ✅ Beach images passed to rip detector
- ✅ Rip currents detected in beach images
- ✅ No false positives on normal beach scenes

**Advantages of two-stage approach:**
1. **Reduced false positives:** Non-beach images don't reach rip detector
2. **Faster processing:** Skip non-relevant images early
3. **Better accuracy:** Each model specialized for its task
4. **Realistic workflow:** Mimics real-world deployment

**Use cases:**
- **Production deployment:** Filter camera feeds
- **Batch processing:** Analyze large image collections
- **System validation:** Test end-to-end pipeline
- **Performance benchmarking:** Compare vs single-stage

**Troubleshooting:**
- Beach classifier failing → Re-train or adjust confidence
- Rip detector missing currents → Check second stage threshold
- Slow processing → Optimize batch processing
- Models not loading → Verify both model paths
""",

    19: """## 1️⃣3️⃣ Real-Time Video Inference

**What this cell does:**
- Processes video files frame-by-frame
- Detects rip currents in each frame
- Displays real-time annotations
- Saves output video with bounding boxes
- Provides frame-by-frame statistics

**Expected execution time:** Varies by video length
- Short video (30s): 2-5 minutes
- Medium video (2min): 10-15 minutes
- Long video (5min+): 30+ minutes

**What to prepare:**
- Video file in: `PROJECT_ROOT/videos/`
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`
- Recommended: 720p or 1080p resolution

**Processing steps:**
1. Load video file
2. Extract frames
3. Run rip detector on each frame
4. Annotate detections
5. Compile annotated frames to output video
6. Save result

**Expected output:**
- Annotated video file: `output_video_annotated.mp4`
- Statistics:
  - Total frames processed
  - Frames with detections
  - Average confidence
  - Processing speed (FPS)
- Progress bar during processing

**Performance benchmarks (RTX 3080):**
- Processing speed: 10-20 FPS
- 30-second video: ~2-3 minutes processing time
- 2-minute video: ~10-15 minutes processing time

**Configuration options:**
- **conf:** Confidence threshold (default 0.35)
- **iou:** IoU threshold for NMS (default 0.45)
- **max_det:** Max detections per frame (default 300)
- **save_frames:** Save individual annotated frames (optional)

**Interpreting results:**
- **Consistent detections:** Stable rip current across frames
- **Flickering boxes:** May indicate low confidence or detection uncertainty
- **Detection appears/disappears:** Rip current may be transient or threshold is borderline

**Real-world applications:**
- **Beach surveillance:** Analyze recorded footage
- **Safety monitoring:** Detect dangerous conditions
- **Research:** Study rip current behavior over time
- **Alert systems:** Trigger warnings when detected

**Optimization tips:**
- Lower resolution videos process faster (but may reduce accuracy)
- Skip frames (process every 2nd or 3rd frame) for speed
- Use GPU for inference (automatic with CUDA)
- Close other GPU applications for maximum speed

**Troubleshooting:**
- Slow processing → Reduce video resolution or skip frames
- Out of memory → Process in smaller batches or reduce batch size
- No detections → Adjust confidence threshold
- Output quality poor → Increase output video bitrate
- Video not loading → Check file path and format support
"""
}


def enhance_notebook(input_path, output_path):
    """
    Enhance v1.1 notebook with detailed markdown descriptions.
    
    Args:
        input_path: Path to original notebook
        output_path: Path to save enhanced notebook
    """
    print(f"Loading notebook from: {input_path}")
    
    # Read with error handling for bad unicode
    with open(input_path, 'r', encoding='utf-8', errors='surrogateescape') as f:
        content = f.read()
    
    # Replace surrogate characters before parsing JSON
    content = content.encode('utf-8', errors='replace').decode('utf-8')
    notebook = json.loads(content)
    
    print(f"Original notebook has {len(notebook['cells'])} cells")
    
    # Create new cells list with inserted markdown
    new_cells = []
    cell_index = 0
    
    for i, cell in enumerate(notebook['cells']):
        # If there's a markdown description for this code cell, add it BEFORE
        if cell['cell_type'] == 'code' and i in MARKDOWN_DESCRIPTIONS:
            markdown_cell = {
                "cell_type": "markdown",
                "id": f"markdown_desc_{cell_index}",
                "metadata": {},
                "source": [MARKDOWN_DESCRIPTIONS[i]]
            }
            new_cells.append(markdown_cell)
            print(f"Added markdown before cell {i}")
            cell_index += 1
        
        # Add the original cell
        new_cells.append(cell)
        cell_index += 1
    
    # Update notebook with new cells
    notebook['cells'] = new_cells
    
    print(f"Enhanced notebook has {len(new_cells)} cells")
    print(f"Added {len(new_cells) - len(notebook['cells'])} new markdown cells")
    
    # Save enhanced notebook with error handling for bad unicode
    print(f"Saving enhanced notebook to: {output_path}")
    
    # First, clean any problematic unicode characters
    def clean_cell(cell):
        """Remove problematic unicode surrogates from cell source"""
        if 'source' in cell and isinstance(cell['source'], list):
            cleaned_source = []
            for line in cell['source']:
                # Replace surrogate characters with ?
                cleaned_line = line.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                cleaned_source.append(cleaned_line)
            cell['source'] = cleaned_source
        return cell
    
    notebook['cells'] = [clean_cell(cell) for cell in notebook['cells']]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Enhancement complete!")
    
    # Print summary
    markdown_count = sum(1 for c in new_cells if c['cell_type'] == 'markdown')
    code_count = sum(1 for c in new_cells if c['cell_type'] == 'code')
    print(f"\nSummary:")
    print(f"  Markdown cells: {markdown_count}")
    print(f"  Code cells: {code_count}")
    print(f"  Documentation ratio: {markdown_count}/{code_count} = {markdown_count/code_count*100:.1f}%")


if __name__ == "__main__":
    input_nb = Path("RipCatch-v1.1/RipCatch-v1.1.ipynb")
    output_nb = Path("RipCatch-v1.1/RipCatch-v1.1.ipynb")  # Overwrite original
    
    enhance_notebook(input_nb, output_nb)
