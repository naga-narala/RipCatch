"""
RipCatch v2.0 - MLflow Model Serving
Serve models via MLflow for inference

Author: Sravan Kumar
GitHub: https://github.com/naga-narala/RipCatch
"""

import mlflow
import mlflow.pyfunc
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import io
import json
from pathlib import Path

class RipCatchMLflowModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow model wrapper for YOLOv8"""
    
    def load_context(self, context):
        """Load model from MLflow artifacts"""
        model_path = context.artifacts["model_weights"]
        self.model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
    
    def predict(self, context, model_input):
        """
        Predict rip currents from images
        
        Args:
            model_input: Can be:
                - numpy array of image
                - PIL Image
                - path to image file
                - dict with 'image' key
        
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        # Handle different input types
        if isinstance(model_input, dict):
            image_input = model_input.get('image')
        else:
            image_input = model_input
        
        # Run prediction
        if isinstance(image_input, str):
            # File path
            results = self.model(image_input)
        elif isinstance(image_input, np.ndarray):
            # Numpy array
            results = self.model(image_input)
        elif isinstance(image_input, Image.Image):
            # PIL Image
            results = self.model(image_input)
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")
        
        # Parse results
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class_id": int(box.cls),
                    "class_name": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    },
                    "bbox_normalized": {
                        "x_center": float(box.xywhn[0][0]),
                        "y_center": float(box.xywhn[0][1]),
                        "width": float(box.xywhn[0][2]),
                        "height": float(box.xywhn[0][3])
                    }
                })
        
        return detections


def register_model_for_serving(model_weights_path, model_name="RipCatch-Serving", version="2.0"):
    """Register model for MLflow serving"""
    
    artifacts = {
        "model_weights": model_weights_path
    }
    
    conda_env = {
        "name": "ripcatch-serve",
        "channels": ["conda-forge", "pytorch"],
        "dependencies": [
            "python=3.10",
            "pip",
            {
                "pip": [
                    "mlflow>=2.9.0",
                    "ultralytics>=8.0.0",
                    "opencv-python>=4.8.0",
                    "pillow>=10.0.0",
                    "numpy>=1.24.0"
                ]
            }
        ]
    }
    
    signature = mlflow.models.infer_signature(
        model_input={"image": "path/to/image.jpg"},
        model_output=[
            {
                "class_id": 0,
                "class_name": "rip_current",
                "confidence": 0.95,
                "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 400}
            }
        ]
    )
    
    with mlflow.start_run(run_name=f"register_{model_name}_{version}"):
        model_info = mlflow.pyfunc.log_model(
            artifact_path="ripcatch_model",
            python_model=RipCatchMLflowModel(),
            artifacts=artifacts,
            conda_env=conda_env,
            signature=signature,
            registered_model_name=model_name
        )
        
        # Add tags to model version
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
        
        client.set_model_version_tag(
            name=model_name,
            version=latest_version.version,
            key="version",
            value=version
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=latest_version.version,
            key="task",
            value="rip-current-detection"
        )
        
        print(f"✅ Model registered for serving: {model_info.model_uri}")
        print(f"📦 Model version: {latest_version.version}")
        
        return model_info, latest_version.version


def promote_model_to_stage(model_name, version, stage="Production"):
    """Promote model version to a specific stage"""
    client = mlflow.tracking.MlflowClient()
    
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    print(f"✅ Model {model_name} v{version} promoted to {stage}")


def serve_model_instructions(model_name="RipCatch-Serving", stage="Production", port=5001):
    """Print instructions for serving model"""
    
    print("\n" + "="*70)
    print("🚀 MLflow Model Serving Instructions")
    print("="*70)
    
    print(f"\n1️⃣  Serve model via MLflow CLI:")
    print(f"   mlflow models serve -m \"models:/{model_name}/{stage}\" -p {port}")
    
    print(f"\n2️⃣  Test via REST API:")
    print(f"""
   # Using curl:
   curl -X POST http://localhost:{port}/invocations \\
       -H 'Content-Type: application/json' \\
       -d '{{"inputs": {{"image": "path/to/beach_image.jpg"}}}}'
   
   # Using Python:
   import requests
   import json
   
   response = requests.post(
       'http://localhost:{port}/invocations',
       headers={{'Content-Type': 'application/json'}},
       data=json.dumps({{"inputs": {{"image": "test_image.jpg"}}}})
   )
   
   detections = response.json()
   print(detections)
""")
    
    print(f"\n3️⃣  Deploy to production:")
    print(f"   # Docker deployment")
    print(f"   mlflow models build-docker -m \"models:/{model_name}/{stage}\" -n ripcatch-mlflow")
    print(f"   docker run -p {port}:8080 ripcatch-mlflow")
    
    print("\n" + "="*70 + "\n")


def load_and_test_model(model_uri, test_image_path):
    """Load model from MLflow and test inference"""
    
    print(f"\n🔍 Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    
    print(f"🖼️  Testing with image: {test_image_path}")
    
    # Make prediction
    predictions = model.predict({"image": test_image_path})
    
    print(f"\n📊 Detections found: {len(predictions)}")
    for i, det in enumerate(predictions, 1):
        print(f"   {i}. {det['class_name']}: {det['confidence']:.2%} at {det['bbox']}")
    
    return predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MLflow Model Serving for RipCatch')
    parser.add_argument('--register', action='store_true', help='Register model for serving')
    parser.add_argument('--model-path', type=str, help='Path to model weights (for registration)')
    parser.add_argument('--model-name', type=str, default='RipCatch-Serving', help='Model name in registry')
    parser.add_argument('--promote', type=int, help='Promote version to Production')
    parser.add_argument('--serve-info', action='store_true', help='Show serving instructions')
    parser.add_argument('--test', type=str, help='Test model with image path')
    parser.add_argument('--model-uri', type=str, help='Model URI for testing')
    
    args = parser.parse_args()
    
    if args.register:
        if not args.model_path:
            print("❌ Error: --model-path required for registration")
            exit(1)
        
        model_info, version = register_model_for_serving(
            model_weights_path=args.model_path,
            model_name=args.model_name
        )
        print(f"\n✅ Registration complete!")
        print(f"📦 Model URI: {model_info.model_uri}")
        print(f"🔢 Version: {version}")
    
    if args.promote:
        promote_model_to_stage(
            model_name=args.model_name,
            version=args.promote,
            stage="Production"
        )
    
    if args.serve_info:
        serve_model_instructions(model_name=args.model_name)
    
    if args.test:
        if not args.model_uri:
            args.model_uri = f"models:/{args.model_name}/Production"
        
        load_and_test_model(
            model_uri=args.model_uri,
            test_image_path=args.test
        )
