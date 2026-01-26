"""
🌊 RipCatch v2.0 - Hugging Face Gradio Interface
AI-powered Rip Current Detection System

Author: Sravan Kumar
GitHub: https://github.com/naga-narala/RipCatch
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import os
from pathlib import Path
import tempfile

# Configuration
MODEL_PATH = "RipCatch-v2.0/Model/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Load model
@torch.no_grad()
def load_model():
    """Load the YOLOv8 model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    model.conf = CONFIDENCE_THRESHOLD
    model.iou = IOU_THRESHOLD
    return model

# Initialize model
print("Loading RipCatch v2.0 model...")
model = load_model()
print("✅ Model loaded successfully!")

def detect_rip_currents_image(image, confidence_threshold, iou_threshold):
    """
    Detect rip currents in an image
    
    Args:
        image: PIL Image or numpy array
        confidence_threshold: Minimum confidence for detection
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Annotated image with detections
    """
    if image is None:
        return None
    
    # Update model thresholds
    model.conf = confidence_threshold
    model.iou = iou_threshold
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Run inference
    results = model(image, verbose=False)
    
    # Get annotated image
    annotated_image = results[0].plot()
    
    # Convert BGR to RGB for display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Get detection info
    num_detections = len(results[0].boxes)
    detection_info = f"🌊 **Detections Found: {num_detections}**\n\n"
    
    if num_detections > 0:
        detection_info += "⚠️ **WARNING: Rip Current Detected!**\n\n"
        detection_info += "**Detection Details:**\n"
        for i, box in enumerate(results[0].boxes):
            conf = float(box.conf[0])
            detection_info += f"- Detection #{i+1}: Confidence = {conf:.2%}\n"
        detection_info += "\n**Safety Advisory:**\n"
        detection_info += "- Do not enter the water in detected areas\n"
        detection_info += "- Swim near a lifeguard\n"
        detection_info += "- If caught, swim parallel to shore\n"
    else:
        detection_info += "✅ **No rip currents detected in this image**\n"
        detection_info += "\n*Note: Always exercise caution near ocean waters*"
    
    return annotated_image, detection_info

def detect_rip_currents_video(video, confidence_threshold, iou_threshold, progress=gr.Progress()):
    """
    Detect rip currents in a video
    
    Args:
        video: Video file path
        confidence_threshold: Minimum confidence for detection
        iou_threshold: IoU threshold for NMS
        progress: Gradio progress bar
    
    Returns:
        Annotated video with detections
    """
    if video is None:
        return None, "No video uploaded"
    
    # Update model thresholds
    model.conf = confidence_threshold
    model.iou = iou_threshold
    
    # Open video
    cap = cv2.VideoCapture(video)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    progress(0, desc="Starting video processing...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, verbose=False)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Write frame
        out.write(annotated_frame)
        
        # Count detections
        total_detections += len(results[0].boxes)
        frame_count += 1
        
        # Update progress
        if frame_count % 10 == 0:
            progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    # Generate summary
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    summary = f"🎥 **Video Processing Complete**\n\n"
    summary += f"- Total Frames: {frame_count}\n"
    summary += f"- Total Detections: {total_detections}\n"
    summary += f"- Average Detections/Frame: {avg_detections:.2f}\n\n"
    
    if total_detections > 0:
        summary += "⚠️ **WARNING: Rip currents detected in this video!**\n"
    else:
        summary += "✅ **No rip currents detected**\n"
    
    return output_path, summary

# Example images for demo
examples_images = [
    ["Testing/beach/beach_1.jpg", 0.25, 0.45],
    ["Testing/beach/beach_2.jpg", 0.25, 0.45],
]

examples_videos = [
    ["Testing/videos/rip_current_1.mp4", 0.25, 0.45],
]

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.gr-button-primary {
    background: linear-gradient(90deg, #0066cc 0%, #00aaff 100%) !important;
    border: none !important;
}
.gr-button-secondary {
    color: #0066cc !important;
}
footer {
    display: none !important;
}
"""

# Create Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="RipCatch v2.0") as demo:
    
    gr.Markdown("""
    # 🌊 RipCatch v2.0 - Advanced Rip Current Detection
    
    **AI-powered system to detect rip currents and enhance beach safety**
    
    [![GitHub](https://img.shields.io/badge/GitHub-naga--narala/RipCatch-blue?logo=github)](https://github.com/naga-narala/RipCatch)
    [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
    [![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
    
    **Model Performance**: 88.64% mAP@50 | 89.03% Precision | 89.51% Recall
    
    ---
    """)
    
    with gr.Tab("📸 Image Detection"):
        gr.Markdown("""
        ### Upload a beach image to detect rip currents
        
        The model will analyze the image and highlight any detected rip currents with bounding boxes.
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Beach Image")
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    conf_slider_img = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Minimum confidence for detection"
                    )
                    iou_slider_img = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.45,
                        step=0.05,
                        label="IoU Threshold",
                        info="Overlap threshold for filtering duplicate detections"
                    )
                
                detect_btn_img = gr.Button("🔍 Detect Rip Currents", variant="primary", size="lg")
            
            with gr.Column():
                image_output = gr.Image(type="numpy", label="Detection Results")
                detection_info = gr.Markdown(label="Detection Information")
        
        gr.Examples(
            examples=examples_images,
            inputs=[image_input, conf_slider_img, iou_slider_img],
            outputs=[image_output, detection_info],
            fn=detect_rip_currents_image,
            cache_examples=False,
            label="Example Images"
        )
    
    with gr.Tab("🎥 Video Detection"):
        gr.Markdown("""
        ### Upload a beach video to detect rip currents
        
        The model will process each frame and highlight detected rip currents throughout the video.
        
        **Note**: Video processing may take several minutes depending on length.
        """)
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Beach Video")
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    conf_slider_vid = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    iou_slider_vid = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.45,
                        step=0.05,
                        label="IoU Threshold"
                    )
                
                detect_btn_vid = gr.Button("🔍 Process Video", variant="primary", size="lg")
            
            with gr.Column():
                video_output = gr.Video(label="Detection Results")
                video_info = gr.Markdown(label="Processing Summary")
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About RipCatch v2.0
        
        ### 🎯 What are Rip Currents?
        
        Rip currents are powerful, narrow channels of fast-moving water that flow from the shore out to sea. They are responsible for approximately **100 deaths annually** in the United States alone.
        
        ### 🚀 How RipCatch Works
        
        RipCatch v2.0 uses a **YOLOv8m** (medium variant) deep learning model trained on 16,907 beach images to automatically detect rip currents in real-time:
        
        - **Single-Stage Detection**: Unified architecture (no separate beach classifier needed)
        - **High Accuracy**: 88.64% mAP@50 with 89.03% precision
        - **Real-Time Processing**: 10-15 FPS on GPU
        - **Advanced Training**: Gradient accumulation, early stopping, optimized hyperparameters
        
        ### 📊 Model Performance
        
        | Metric | Value |
        |--------|-------|
        | mAP@50 | 88.64% |
        | mAP@50-95 | 61.45% |
        | Precision | 89.03% |
        | Recall | 89.51% |
        | F1-Score | 89.27% |
        
        ### 🏗️ Technical Details
        
        - **Architecture**: YOLOv8m (25M parameters)
        - **Training Data**: 14,436 training images
        - **Validation Data**: 1,804 images
        - **Test Data**: 667 images
        - **Image Resolution**: 640×640
        - **Training Time**: 4-5 hours on RTX 3080
        
        ### ⚠️ Safety Notice
        
        This is an AI-powered detection system and should be used as a **supplementary tool** alongside:
        - Trained lifeguards
        - Official beach safety warnings
        - Local knowledge and experience
        
        **Always exercise caution near ocean waters!**
        
        ### 🔗 Links
        
        - **GitHub Repository**: [github.com/naga-narala/RipCatch](https://github.com/naga-narala/RipCatch)
        - **Documentation**: See repository README for full details
        - **License**: MIT License
        - **Contact**: sravankumar.nnv@gmail.com
        
        ### 📖 Citation
        
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
        
        ---
        
        ### 🙏 Acknowledgments
        
        - **Ultralytics**: YOLOv8 framework
        - **PyTorch**: Deep learning framework
        - **Gradio**: ML demo interface
        - **Beach safety organizations** worldwide for raising awareness
        
        ---
        
        <div align="center">
        
        **Made with ❤️ for beach safety | © 2024-2026 Sravan Kumar**
        
        </div>
        """)
    
    # Event handlers
    detect_btn_img.click(
        fn=detect_rip_currents_image,
        inputs=[image_input, conf_slider_img, iou_slider_img],
        outputs=[image_output, detection_info]
    )
    
    detect_btn_vid.click(
        fn=detect_rip_currents_video,
        inputs=[video_input, conf_slider_vid, iou_slider_vid],
        outputs=[video_output, video_info]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
