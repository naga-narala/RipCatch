"""
RipCatch v2.0 - MLflow Configuration
====================================
This file handles all MLflow setup and configuration for the RipCatch project.

WHAT IS MLFLOW?
--------------
MLflow is like a "notebook" for your machine learning experiments. It helps you:
1. Track experiments (what settings you tried)
2. Save results (how well each model performed)
3. Store models (save your trained models)
4. Compare different versions (which model is best?)

BEGINNER'S GUIDE:
----------------
Think of MLflow as your ML project manager that automatically:
- Records every training run you do
- Saves all the settings you used
- Tracks how well your model performed
- Stores your trained models
- Lets you compare different experiments

Author: Sravan Kumar
GitHub: https://github.com/naga-narala/RipCatch
"""

import mlflow  # The main MLflow library for tracking experiments
import os  # For reading environment variables and file paths
from pathlib import Path  # For handling file paths in a cross-platform way
from datetime import datetime  # For creating timestamps in run names

class MLflowConfig:
    """
    MLflow configuration for RipCatch project
    
    WHAT THIS CLASS DOES:
    --------------------
    This class sets up MLflow for your RipCatch experiments. It's like
    creating a lab notebook where you'll record all your ML experiments.
    
    SIMPLE EXPLANATION:
    ------------------
    Imagine you're baking cookies and want to track different recipes:
    - MLflowConfig creates your recipe notebook
    - Each experiment is a new recipe you try
    - MLflow records ingredients (parameters), baking time, and results
    - Later, you can compare which recipe made the best cookies!
    """
    
    def __init__(self, experiment_name="RipCatch-v2.0", tracking_uri="./mlruns", artifact_location="./mlartifacts"):
        """
        Initialize MLflow configuration
        
        PARAMETERS EXPLAINED FOR BEGINNERS:
        ----------------------------------
        experiment_name: str (default: "RipCatch-v2.0")
            - The name of your experiment (like a project folder name)
            - Example: "RipCatch-v2.0" or "RipCatch-Testing"
            - Think of it as the title of your experiment notebook
        
        tracking_uri: str (default: "./mlruns")
            - Where MLflow saves experiment data
            - "./mlruns" = saves in a local folder called "mlruns"
            - "http://localhost:5000" = connects to MLflow server
            - Like choosing where to save your experiment notebook
        
        artifact_location: str (default: "./mlartifacts")
            - Where MLflow saves files (models, plots, etc.)
            - Think of it as a filing cabinet for your experiment outputs
        """
        # Store the experiment name (like naming your project)
        self.experiment_name = experiment_name
        
        # Set tracking URI - where MLflow stores experiment data
        # Priority: 1) Environment variable, 2) Parameter, 3) Default local folder
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", tracking_uri)
        
        # Set artifact location - where MLflow stores files (models, images, etc.)
        self.artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION", artifact_location)
        
        # Setup MLflow when this class is created
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """
        Initialize MLflow tracking
        
        WHAT THIS FUNCTION DOES:
        -----------------------
        This function sets up MLflow to start tracking your experiments.
        It's like opening your experiment notebook and getting ready to write.
        
        STEP-BY-STEP PROCESS:
        --------------------
        1. Tells MLflow where to save data (tracking_uri)
        2. Creates a new experiment if it doesn't exist
        3. Sets up tags to organize your experiments
        4. Prints confirmation that everything is ready
        
        BEGINNER TIP:
        ------------
        You only need to run this once at the start. After that, MLflow
        will automatically track everything you do!
        """
        # Step 1: Tell MLflow where to save experiment data
        # This is like choosing which notebook to write in
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Step 2: Create or get the experiment
        # Think of this as creating a new section in your notebook
        try:
            # Check if experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            if experiment is None:
                # Experiment doesn't exist, so create a new one
                print(f"📝 Creating new experiment: {self.experiment_name}")
                
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,  # Name of your experiment
                    artifact_location=self.artifact_location,  # Where to save files
                    tags={
                        # Tags help organize and find experiments later
                        # It's like adding labels to your notebook sections
                        "project": "RipCatch",  # Project name
                        "version": "2.0",  # Model version
                        "model": "YOLOv8",  # Model type
                        "task": "rip-current-detection",  # What the model does
                        "framework": "ultralytics"  # Library used
                    }
                )
            else:
                # Experiment already exists, so use it
                experiment_id = experiment.experiment_id
                print(f"📂 Using existing experiment: {self.experiment_name}")
            
            # Step 3: Set this as the active experiment
            # All future tracking will go into this experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Step 4: Print success message
            print(f"✅ MLflow experiment set: {self.experiment_name} (ID: {experiment_id})")
            print(f"📊 Tracking URI: {self.tracking_uri}")
            print(f"📁 Artifacts will be saved to: {self.artifact_location}")
            
            # Return experiment information (useful for other functions)
            return {
                "experiment_name": self.experiment_name,
                "experiment_id": experiment_id,
                "tracking_uri": self.tracking_uri,
                "artifact_location": self.artifact_location
            }
            
        except Exception as e:
            # If something goes wrong, show a helpful error message
            print(f"⚠️ MLflow setup warning: {e}")
            print(f"💡 TROUBLESHOOTING:")
            print(f"   1. Make sure MLflow is installed: pip install mlflow")
            print(f"   2. If using a server, make sure it's running:")
            print(f"      mlflow ui --backend-store-uri ./mlruns --port 5000")
            print(f"   3. Check if the tracking URI is correct: {self.tracking_uri}")
            return None
    
    @staticmethod
    def get_run_name(model_variant="yolov8m", dataset_version="v2.0", suffix=""):
        """
        Generate a unique run name for an experiment
        
        WHAT IS A "RUN"?
        ---------------
        A "run" is a single training session. If you train your model 10 times
        with different settings, that's 10 runs. Each run needs a unique name.
        
        WHY UNIQUE NAMES?
        ----------------
        Unique names help you identify each experiment later. It's like labeling
        each batch of cookies you bake with the date and recipe variation.
        
        PARAMETERS:
        ----------
        model_variant: str (default: "yolov8m")
            - Which YOLO model size you're using
            - Options: yolov8n (nano), yolov8s (small), yolov8m (medium), etc.
            - Example: "yolov8m" = medium-sized model
        
        dataset_version: str (default: "v2.0")
            - Which version of your dataset
            - Example: "v2.0", "v1.1", "test"
        
        suffix: str (default: "")
            - Optional: add extra description
            - Example: "augmented", "lr_0.01", "batch_32"
        
        RETURNS:
        -------
        A unique name like: "yolov8m_v2.0_20260128_143052"
        Or with suffix: "yolov8m_v2.0_augmented_20260128_143052"
        
        BEGINNER EXAMPLE:
        ----------------
        run_name = get_run_name("yolov8s", "v2.0", "test")
        # Result: "yolov8s_v2.0_test_20260128_143052"
        """
        # Get current date and time (like adding a timestamp to your experiment)
        # Format: YYYYMMDD_HHMMSS (e.g., 20260128_143052 = Jan 28, 2026, 2:30:52 PM)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build the run name
        if suffix:
            # If you provided a suffix, include it in the name
            return f"{model_variant}_{dataset_version}_{suffix}_{timestamp}"
        else:
            # No suffix, just use model and dataset version
            return f"{model_variant}_{dataset_version}_{timestamp}"
    
    @staticmethod
    def get_model_tags(stage="development", version="2.0"):
        """
        Get standard model tags for organizing experiments
        
        WHAT ARE TAGS?
        -------------
        Tags are like labels or stickers you put on experiments to organize them.
        They help you filter and find experiments later.
        
        THINK OF IT LIKE:
        ----------------
        - Organizing photos with hashtags (#vacation, #family, #2024)
        - Filing documents in labeled folders
        - Adding post-it notes to experiments
        
        PARAMETERS:
        ----------
        stage: str (default: "development")
            - What phase is this model in?
            - Options: "development", "testing", "staging", "production"
            - "development" = still experimenting
            - "production" = ready for real use
        
        version: str (default: "2.0")
            - Version number of your model
            - Example: "1.0", "2.0", "2.1"
        
        RETURNS:
        -------
        A dictionary of tags like:
        {
            "stage": "development",
            "version": "2.0",
            "project": "RipCatch",
            "model_type": "object-detection",
            "framework": "YOLOv8"
        }
        
        BEGINNER EXAMPLE:
        ----------------
        tags = get_model_tags(stage="production", version="2.0")
        # This marks your model as production-ready version 2.0
        """
        return {
            "stage": stage,  # development, testing, staging, or production
            "version": version,  # model version number
            "project": "RipCatch",  # project name
            "model_type": "object-detection",  # what type of AI task
            "framework": "YOLOv8"  # which AI framework you're using
        }

# Global config instance
# This creates one MLflow configuration that can be used throughout your project
# Think of it as setting up your lab notebook once, then using it everywhere
mlflow_config = MLflowConfig()

if __name__ == "__main__":
    """
    This section runs only when you execute this file directly
    It's a test to make sure everything is working correctly
    
    TO RUN THIS TEST:
    ----------------
    python mlflow_config.py
    
    WHAT IT DOES:
    ------------
    1. Creates an MLflow configuration
    2. Prints the settings
    3. Generates a sample run name
    4. Shows you that everything is working
    """
    # Test configuration
    print("\n" + "="*60)
    print("🔍 Testing MLflow Configuration...")
    print("="*60)
    print(f"📁 Experiment Name: {mlflow_config.experiment_name}")
    print(f"📊 Tracking URI: {mlflow_config.tracking_uri}")
    print(f"💾 Artifact Location: {mlflow_config.artifact_location}")
    print(f"🏷️  Sample Run Name: {MLflowConfig.get_run_name()}")
    print(f"🏷️  Sample Tags: {MLflowConfig.get_model_tags()}")
    print("="*60)
    print("✅ Configuration test complete!")
    print("\n💡 TIP: You can now use this config in your training scripts!")
    print("   Example: from mlflow_integration.mlflow_config import MLflowConfig")
    print("="*60 + "\n")
