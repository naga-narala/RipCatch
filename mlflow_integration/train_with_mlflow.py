"""
RipCatch v2.0 - YOLOv8 Training with MLflow Tracking
=====================================================
This script trains a YOLOv8 model for rip current detection while automatically
tracking all experiments with MLflow.

WHAT THIS SCRIPT DOES:
---------------------
1. Trains a YOLOv8 model on your rip current dataset
2. Automatically logs all training parameters to MLflow
3. Records performance metrics during training
4. Saves the trained model and training artifacts
5. Makes it easy to compare different training runs

BEGINNER'S GUIDE:
----------------
Think of this as an automated training assistant that:
- Keeps detailed notes of every training session
- Saves all important information automatically
- Helps you compare different experiments later
- Never forgets which settings produced which results

HOW TO USE:
----------
Basic usage:
    python train_with_mlflow.py --data data.yaml --epochs 100 --batch 16

With custom settings:
    python train_with_mlflow.py --data data.yaml --epochs 200 --batch 32 --lr0 0.001

IMPORTANT FOR BEGINNERS:
-----------------------
- Make sure MLflow UI is running: mlflow ui --port 5000
- Your training progress will be visible in real-time at http://localhost:5000
- All results are saved automatically - no manual work needed!
- You can stop and resume training - MLflow tracks everything

Author: Sravan Kumar
GitHub: https://github.com/naga-narala/RipCatch
"""

import mlflow  # MLflow for experiment tracking
import mlflow.pytorch  # MLflow integration with PyTorch
from ultralytics import YOLO  # YOLOv8 from Ultralytics
import yaml  # For reading YAML configuration files
from pathlib import Path  # For handling file paths
import json  # For handling JSON data
import pandas as pd  # For data analysis and logging
from mlflow_config import MLflowConfig  # Our custom MLflow configuration

class RipCatchMLflowTrainer:
    """
    YOLOv8 Trainer with Automatic MLflow Tracking
    
    WHAT THIS CLASS DOES:
    --------------------
    This class wraps YOLOv8 training and automatically logs everything to MLflow.
    Think of it as a "smart trainer" that remembers everything about your experiments.
    
    SIMPLE EXPLANATION:
    ------------------
    Normal training: You run training, get results, but might forget the settings.
    With this class: Everything is automatically recorded - settings, results, models!
    
    EXAMPLE USAGE:
    -------------
    # Create a trainer
    trainer = RipCatchMLflowTrainer(
        model_variant="yolov8m.pt",  # Which YOLO model to use
        data_yaml="data.yaml"  # Where's your dataset config?
    )
    
    # Train with MLflow tracking
    trainer.train(
        epochs=100,  # How many times to go through the dataset
        batch=16,  # How many images to process at once
        imgsz=640  # Resize images to 640x640 pixels
    )
    
    That's it! MLflow handles all the logging automatically.
    """
    
    def __init__(self, model_variant="yolov8m.pt", data_yaml="data.yaml"):
        """
        Initialize the trainer
        
        PARAMETERS FOR BEGINNERS:
        ------------------------
        model_variant: str (default: "yolov8m.pt")
            Which YOLOv8 model size to use:
            - "yolov8n.pt" = Nano (fastest, less accurate)
            - "yolov8s.pt" = Small (balanced)
            - "yolov8m.pt" = Medium (good balance) ← RECOMMENDED
            - "yolov8l.pt" = Large (slower, more accurate)
            - "yolov8x.pt" = Extra large (slowest, most accurate)
        
        data_yaml: str (default: "data.yaml")
            Path to your dataset configuration file
            This file tells YOLO where your images and labels are
        
        WHAT HAPPENS WHEN YOU CREATE THIS:
        ----------------------------------
        1. Stores which model you want to use
        2. Remembers where your dataset is
        3. Sets up MLflow configuration
        4. Gets ready to track experiments
        """
        # Store the model variant (which YOLO size)
        self.model_variant = model_variant
        
        # Store the dataset configuration path
        self.data_yaml = data_yaml
        
        # Initialize MLflow configuration
        # This sets up experiment tracking
        self.mlflow_config = MLflowConfig()
        
        print(f"✅ Trainer initialized!")
        print(f"   Model: {model_variant}")
        print(f"   Dataset: {data_yaml}")
        print(f"   MLflow tracking: {self.mlflow_config.tracking_uri}")
    
    def log_dataset_info(self, data_yaml_path):
        """
        Log dataset configuration and statistics to MLflow
        
        WHAT THIS FUNCTION DOES:
        -----------------------
        Reads your dataset configuration file (data.yaml) and logs all the
        important information to MLflow so you remember what data you used.
        
        WHY IS THIS IMPORTANT?
        ---------------------
        If you train multiple models on different datasets, MLflow helps you
        remember which dataset was used for each model. This is crucial for
        reproducing results!
        
        WHAT GETS LOGGED:
        ----------------
        - Dataset path (where images are stored)
        - Number of classes (how many different things to detect)
        - Class names (what objects the model detects)
        - Train/validation/test split paths
        - The actual data.yaml file itself
        
        BEGINNER TIP:
        ------------
        This happens automatically when you call train() - you don't need
        to call this function yourself!
        """
        # Read the dataset configuration file
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Extract and log dataset parameters
        dataset_params = {
            "dataset_path": data_config.get('path', 'N/A'),  # Root dataset folder
            "num_classes": data_config.get('nc', 0),  # Number of classes
            "class_names": str(data_config.get('names', [])),  # List of class names
            "train_path": data_config.get('train', 'N/A'),  # Training images folder
            "val_path": data_config.get('val', 'N/A'),  # Validation images folder
            "test_path": data_config.get('test', 'N/A'),  # Test images folder (if exists)
        }
        
        # Log all dataset parameters to MLflow
        mlflow.log_params(dataset_params)
        
        # Save the actual data.yaml file as an artifact
        # This way you can download it later if needed
        mlflow.log_artifact(data_yaml_path, "dataset_config")
        
        print("📊 Dataset info logged to MLflow")
        print(f"   Classes: {data_config.get('nc', 0)}")
        print(f"   Names: {data_config.get('names', [])}")
    
    def log_hyperparameters(self, **kwargs):
        """
        Log training hyperparameters to MLflow
        
        WHAT ARE HYPERPARAMETERS?
        ------------------------
        Hyperparameters are settings that control how the model learns.
        Think of them like recipe settings when baking:
        - Oven temperature = Learning rate
        - Baking time = Number of epochs
        - Batch size = How many cookies you bake at once
        
        WHY LOG THESE?
        -------------
        If your model performs well, you want to remember exactly what settings
        you used so you can reproduce the results or improve on them!
        
        KEY HYPERPARAMETERS EXPLAINED:
        -----------------------------
        - epochs: How many times to go through the entire dataset
        - batch_size: How many images to process together
        - img_size: Resize all images to this size (e.g., 640x640)
        - lr0: Learning rate (how big each learning step is)
        - optimizer: Which optimization algorithm to use (SGD, Adam, etc.)
        - momentum: Helps optimization converge faster
        - weight_decay: Prevents overfitting
        
        BEGINNER TIP:
        ------------
        You don't need to understand all of these! The default values work well.
        As you gain experience, you can experiment with different values.
        """
        # Compile all hyperparameters with default values
        default_params = {
            "model_variant": self.model_variant,  # Which YOLO model (n/s/m/l/x)
            "epochs": kwargs.get('epochs', 100),  # Training iterations
            "batch_size": kwargs.get('batch', 16),  # Images per batch
            "img_size": kwargs.get('imgsz', 640),  # Image dimensions
            "optimizer": kwargs.get('optimizer', 'SGD'),  # Optimization algorithm
            "lr0": kwargs.get('lr0', 0.01),  # Initial learning rate
            "lrf": kwargs.get('lrf', 0.01),  # Final learning rate
            "momentum": kwargs.get('momentum', 0.937),  # SGD momentum
            "weight_decay": kwargs.get('weight_decay', 0.0005),  # Regularization
            "warmup_epochs": kwargs.get('warmup_epochs', 3.0),  # Warmup period
            "patience": kwargs.get('patience', 50),  # Early stopping patience
            "dropout": kwargs.get('dropout', 0.0),  # Dropout rate
        }
        
        # Log all parameters to MLflow
        mlflow.log_params(default_params)
        
        print("⚙️  Hyperparameters logged to MLflow")
        print(f"   Epochs: {default_params['epochs']}")
        print(f"   Batch size: {default_params['batch_size']}")
        print(f"   Learning rate: {default_params['lr0']}")
    
    def train(self, epochs=100, batch=16, imgsz=640, **kwargs):
        """Train YOLOv8 with MLflow tracking"""
        
        run_name = MLflowConfig.get_run_name(
            model_variant=self.model_variant.replace('.pt', ''),
            dataset_version="v2.0"
        )
        
        with mlflow.start_run(run_name=run_name) as run:
            print("\n" + "="*60)
            print(f"🚀 Starting MLflow Run: {run_name}")
            print(f"📊 Run ID: {run.info.run_id}")
            print(f"🔗 View at: {self.mlflow_config.tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
            print("="*60 + "\n")
            
            # Log dataset info
            self.log_dataset_info(self.data_yaml)
            
            # Log hyperparameters
            self.log_hyperparameters(
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                **kwargs
            )
            
            # Initialize YOLO model
            model = YOLO(self.model_variant)
            
            # Train model
            print("\n🏋️ Training started...\n")
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                project="runs/detect",
                name=run_name,
                save=True,
                plots=True,
                verbose=True,
                **kwargs
            )
            
            # Log metrics from training
            self.log_training_metrics(results)
            
            # Log model artifacts
            self.log_model_artifacts(model, run_name)
            
            # Register model in MLflow Model Registry
            self.register_model(model, run_name, run.info.run_id)
            
            print("\n" + "="*60)
            print(f"✅ Training complete!")
            print(f"📊 Run ID: {run.info.run_id}")
            print(f"🔗 View results: {self.mlflow_config.tracking_uri}")
            print("="*60 + "\n")
            
            return results
    
    def log_training_metrics(self, results):
        """Log training metrics to MLflow"""
        try:
            # Final metrics
            metrics_to_log = {
                "mAP50": results.results_dict.get('metrics/mAP50(B)', 0),
                "mAP50-95": results.results_dict.get('metrics/mAP50-95(B)', 0),
                "precision": results.results_dict.get('metrics/precision(B)', 0),
                "recall": results.results_dict.get('metrics/recall(B)', 0),
                "box_loss": results.results_dict.get('train/box_loss', 0),
                "cls_loss": results.results_dict.get('train/cls_loss', 0),
                "dfl_loss": results.results_dict.get('train/dfl_loss', 0),
            }
            
            for metric_name, value in metrics_to_log.items():
                if value is not None:
                    mlflow.log_metric(metric_name, float(value))
            
            # Calculate F1 score
            precision = metrics_to_log.get('precision', 0)
            recall = metrics_to_log.get('recall', 0)
            if precision and recall:
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
                mlflow.log_metric("f1_score", float(f1_score))
            
            print("📈 Training metrics logged to MLflow")
            
        except Exception as e:
            print(f"⚠️ Error logging metrics: {e}")
    
    def log_model_artifacts(self, model, run_name):
        """Log model files and training artifacts"""
        try:
            run_dir = Path(f"runs/detect/{run_name}")
            
            # Log best weights
            weights_path = run_dir / "weights" / "best.pt"
            if weights_path.exists():
                mlflow.log_artifact(str(weights_path), "model_weights")
                print(f"📦 Logged best weights: {weights_path}")
            
            # Log last weights
            last_weights_path = run_dir / "weights" / "last.pt"
            if last_weights_path.exists():
                mlflow.log_artifact(str(last_weights_path), "model_weights")
            
            # Log training results CSV
            results_path = run_dir / "results.csv"
            if results_path.exists():
                mlflow.log_artifact(str(results_path), "training_results")
            
            # Log confusion matrix
            confusion_matrix_path = run_dir / "confusion_matrix.png"
            if confusion_matrix_path.exists():
                mlflow.log_artifact(str(confusion_matrix_path), "visualizations")
            
            # Log training curves
            results_png = run_dir / "results.png"
            if results_png.exists():
                mlflow.log_artifact(str(results_png), "visualizations")
            
            # Log PR curve
            pr_curve_path = run_dir / "PR_curve.png"
            if pr_curve_path.exists():
                mlflow.log_artifact(str(pr_curve_path), "visualizations")
            
            # Log F1 curve
            f1_curve_path = run_dir / "F1_curve.png"
            if f1_curve_path.exists():
                mlflow.log_artifact(str(f1_curve_path), "visualizations")
            
            print("📦 Model artifacts logged to MLflow")
            
        except Exception as e:
            print(f"⚠️ Error logging artifacts: {e}")
    
    def register_model(self, model, run_name, run_id):
        """Register model in MLflow Model Registry"""
        try:
            # Create model metadata
            model_metadata = {
                "run_name": run_name,
                "run_id": run_id,
                "model_variant": self.model_variant,
                "framework": "YOLOv8",
                "task": "object-detection",
                "dataset": "RipCatch-v2.0"
            }
            
            # Log model metadata
            mlflow.log_dict(model_metadata, "model_metadata.json")
            
            print(f"🎯 Model registered with metadata")
            
        except Exception as e:
            print(f"⚠️ Model registration warning: {e}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RipCatch YOLOv8 with MLflow')
    parser.add_argument('--model', type=str, default='yolov8m.pt', help='Model variant')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RipCatchMLflowTrainer(
        model_variant=args.model,
        data_yaml=args.data
    )
    
    # Train with MLflow tracking
    results = trainer.train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        save=True,
        plots=True
    )
    
    print("\n✅ Training complete! Check MLflow UI for results.")
