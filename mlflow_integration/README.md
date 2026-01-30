# 🚀 MLflow Integration for RipCatch

Complete MLflow integration for experiment tracking, model versioning, and deployment management in the RipCatch project.

## 📁 Structure

```
mlflow_integration/
├── 📄 Python Files (Production)
│   ├── mlflow_config.py          # Core MLflow setup and configuration
│   ├── train_with_mlflow.py      # Training pipeline with experiment tracking
│   ├── model_evaluation.py       # Model evaluation and comparison
│   ├── mlflow_server.py          # Model serving via MLflow
│   └── experiment_tracking.py    # Experiment management utilities
│
├── 📓 Jupyter Notebooks (Research & Experimentation)
│   ├── 01_mlflow_quickstart.ipynb        # Getting started with MLflow
│   ├── 02_experiment_tracking.ipynb      # Advanced experiment tracking
│   ├── 03_model_comparison.ipynb         # Compare model versions
│   ├── 04_hyperparameter_tuning.ipynb    # HPO with Optuna/MLflow
│   └── 05_production_deployment.ipynb    # Deploy models to production
│
├── 📋 Configuration Files
│   ├── requirements-mlflow.txt           # MLflow dependencies
│   ├── docker-compose-mlflow.yml         # MLflow server + PostgreSQL + MinIO
│   └── README.md                          # This file
│
└── 📊 Generated Artifacts (created at runtime)
    ├── mlruns/                            # Local experiment tracking
    └── mlartifacts/                       # Model artifacts and files
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-mlflow.txt
```

### 2. Start MLflow Tracking Server (Option A: Local)

```bash
# Simple local tracking
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open http://localhost:5000 to view the MLflow UI.

### 3. Start MLflow with Docker (Option B: Production)

```bash
# Start MLflow server with PostgreSQL backend and MinIO storage
docker-compose -f docker-compose-mlflow.yml up -d

# View logs
docker-compose -f docker-compose-mlflow.yml logs -f mlflow

# Access services:
# - MLflow UI: http://localhost:5000
# - MinIO Console: http://localhost:9001 (admin/minioadmin)
# - PostgreSQL: localhost:5432
```

### 4. Run Your First Experiment

#### Option A: Using Python Scripts

```bash
# Train with MLflow tracking
python mlflow_integration/train_with_mlflow.py \
    --data ../RipCatch-v2.0/Datasets/data.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640

# Evaluate model
python mlflow_integration/model_evaluation.py \
    --model_path ../RipCatch-v2.0/Model/weights/best.pt \
    --data ../RipCatch-v2.0/Datasets/data.yaml

# Compare models
python mlflow_integration/experiment_tracking.py --compare
```

#### Option B: Using Jupyter Notebooks

```bash
jupyter notebook mlflow_integration/01_mlflow_quickstart.ipynb
```

## 📚 Python Modules

### `mlflow_config.py`

Core configuration and setup for MLflow integration.

```python
from mlflow_integration.mlflow_config import MLflowConfig

config = MLflowConfig(
    tracking_uri="./mlruns",
    experiment_name="RipCatch-Training",
    artifact_location="./mlartifacts"
)
config.setup_mlflow()
```

**Features:**
- Centralized MLflow configuration
- Experiment creation and management
- Run naming conventions
- Artifact organization

### `train_with_mlflow.py`

Training pipeline with comprehensive MLflow tracking.

```python
from mlflow_integration.train_with_mlflow import RipCatchMLflowTrainer

trainer = RipCatchMLflowTrainer(
    model_name="yolov8m.pt",
    data_yaml="../RipCatch-v2.0/Datasets/data.yaml",
    experiment_name="RipCatch-v2.0-Training"
)

trainer.train(
    epochs=100,
    batch=16,
    imgsz=640,
    device="cuda"
)
```

**Automatically Logs:**
- Hyperparameters (epochs, batch size, learning rate, etc.)
- Training metrics (loss, mAP, precision, recall)
- Validation metrics per epoch
- Model checkpoints
- Training curves and plots
- System info (GPU, CUDA version, etc.)

### `model_evaluation.py`

Comprehensive model evaluation and comparison.

```python
from mlflow_integration.model_evaluation import RipCatchEvaluator

evaluator = RipCatchEvaluator(
    model_path="../RipCatch-v2.0/Model/weights/best.pt",
    data_yaml="../RipCatch-v2.0/Datasets/data.yaml"
)

metrics = evaluator.evaluate()
evaluator.compare_models(["run_id_1", "run_id_2"])
```

**Features:**
- Per-class performance metrics
- Confusion matrices
- Precision-recall curves
- Model comparison reports
- Statistical analysis

### `mlflow_server.py`

Serve models via MLflow REST API.

```python
from mlflow_integration.mlflow_server import RipCatchMLflowModel

# Register model for serving
model = RipCatchMLflowModel()
model.register_model_for_serving(
    model_path="../RipCatch-v2.0/Model/weights/best.pt",
    model_name="RipCatch-v2.0-Production"
)

# Serve model
# mlflow models serve -m "models:/RipCatch-v2.0-Production/1" -p 5001
```

**Features:**
- Custom PyFunc model wrapper
- REST API endpoint
- Batch prediction support
- Multiple input formats (numpy, PIL, file paths)

### `experiment_tracking.py`

Utilities for experiment management.

```python
from mlflow_integration.experiment_tracking import ExperimentManager

manager = ExperimentManager(experiment_name="RipCatch-Training")

# Find best model
best_run = manager.get_best_model(metric="mAP50", ascending=False)

# Compare runs
manager.compare_runs(["run_id_1", "run_id_2"])

# Visualize metrics
manager.visualize_metrics(metric_names=["mAP50", "precision", "recall"])
```

**Features:**
- Run comparison
- Best model selection
- Metric visualization
- Export to CSV/JSON

## 📓 Jupyter Notebooks

### 01_mlflow_quickstart.ipynb

Introduction to MLflow with RipCatch.

**Topics:**
- MLflow installation and setup
- Basic experiment tracking
- Logging parameters, metrics, and artifacts
- Viewing results in MLflow UI

### 02_experiment_tracking.ipynb

Deep dive into experiment tracking.

**Topics:**
- Multiple training runs with different configs
- Advanced logging techniques
- Organizing experiments with tags
- Querying and filtering runs

### 03_model_comparison.ipynb

Compare RipCatch model versions.

**Topics:**
- Load and compare v1.0, v1.1, v2.0
- Performance metrics comparison
- Visual comparison charts
- Statistical significance testing

### 04_hyperparameter_tuning.ipynb

Hyperparameter optimization with MLflow.

**Topics:**
- Grid search with MLflow tracking
- Optuna integration
- Parallel experiment runs
- Best hyperparameter selection

### 05_production_deployment.ipynb

Deploy models to production.

**Topics:**
- Model registry and versioning
- Model staging (dev, staging, production)
- REST API serving
- Docker containerization
- A/B testing setup

## 🔧 Configuration

### Environment Variables

```bash
# MLflow Tracking Server
export MLFLOW_TRACKING_URI=http://localhost:5000

# S3/MinIO for artifact storage (optional)
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# Database backend (optional)
export MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:password@localhost:5432/mlflow_db
```

### MLflow Configuration File

Create `.mlflowrc` in your home directory:

```ini
[mlflow]
tracking_uri = http://localhost:5000
default_artifact_root = s3://mlflow/artifacts
experiment_name = RipCatch-Default
```

## 📊 MLflow UI Features

Access the MLflow UI at http://localhost:5000:

- **Experiments**: View and organize all experiments
- **Runs**: Compare runs side-by-side
- **Parameters**: Track all hyperparameters
- **Metrics**: Visualize training progress
- **Artifacts**: Download models and plots
- **Models**: Manage model registry
- **Notes**: Add experiment documentation

## 🐳 Docker Deployment

### Services

The `docker-compose-mlflow.yml` includes:

1. **PostgreSQL**: Backend database for experiment metadata
2. **MinIO**: S3-compatible object storage for artifacts
3. **MLflow Server**: Tracking server with UI
4. **RipCatch Worker** (optional): GPU-enabled training container

### Start All Services

```bash
docker-compose -f docker-compose-mlflow.yml up -d
```

### Start with GPU Training Worker

```bash
docker-compose -f docker-compose-mlflow.yml --profile gpu-training up -d
```

### Stop Services

```bash
docker-compose -f docker-compose-mlflow.yml down
```

### View Logs

```bash
docker-compose -f docker-compose-mlflow.yml logs -f
```

## 🎯 Use Cases

### 1. Track Training Experiments

```python
from mlflow_integration.train_with_mlflow import RipCatchMLflowTrainer

trainer = RipCatchMLflowTrainer(
    model_name="yolov8m.pt",
    data_yaml="data.yaml",
    experiment_name="RipCatch-Ablation-Study"
)

# Run multiple experiments with different configurations
for lr in [0.001, 0.01, 0.1]:
    trainer.train(epochs=50, lr0=lr, name=f"lr_{lr}")
```

### 2. Compare Model Versions

```python
from mlflow_integration.model_evaluation import RipCatchEvaluator

evaluator = RipCatchEvaluator(
    model_path="best.pt",
    data_yaml="data.yaml"
)

# Compare with baseline
evaluator.compare_models(
    run_ids=["baseline_run_id", "current_run_id"],
    output_path="comparison_report.html"
)
```

### 3. Deploy Best Model

```python
from mlflow_integration.experiment_tracking import ExperimentManager

manager = ExperimentManager(experiment_name="RipCatch-Production")

# Get best performing model
best_run = manager.get_best_model(metric="mAP50", ascending=False)

# Register for production
mlflow.register_model(
    f"runs:/{best_run.info.run_id}/model",
    "RipCatch-Production"
)
```

### 4. Serve Model via REST API

```bash
# Serve latest production model
mlflow models serve -m "models:/RipCatch-Production/latest" -p 5001

# Make predictions
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"instances": [{"image": "base64_encoded_image"}]}'
```

## 📈 Best Practices

### Experiment Naming

Use descriptive, hierarchical names:
```
RipCatch-v2.0/YOLOv8m/baseline
RipCatch-v2.0/YOLOv8m/augmentation-study
RipCatch-v2.0/YOLOv8l/learning-rate-tuning
```

### Tagging

Add meaningful tags to runs:
```python
mlflow.set_tags({
    "model_version": "v2.0",
    "dataset": "RipCatch-2024",
    "purpose": "production",
    "team": "research"
})
```

### Logging Frequency

- Log training metrics every epoch
- Log validation metrics every N epochs
- Save checkpoints periodically
- Log final model and artifacts

### Model Registry

Use semantic versioning:
- Development: `models:/RipCatch-Dev/latest`
- Staging: `models:/RipCatch-Staging/latest`
- Production: `models:/RipCatch-Production/1`

## 🐛 Troubleshooting

### Issue: MLflow UI not accessible

```bash
# Check if server is running
curl http://localhost:5000

# Restart server
mlflow ui --backend-store-uri ./mlruns --port 5000
```

### Issue: PostgreSQL connection error

```bash
# Check PostgreSQL status
docker-compose -f docker-compose-mlflow.yml ps postgres

# Restart PostgreSQL
docker-compose -f docker-compose-mlflow.yml restart postgres
```

### Issue: Artifacts not saving

```bash
# Check permissions
ls -la mlartifacts/

# Set correct permissions
chmod -R 755 mlartifacts/
```

### Issue: GPU not detected in Docker

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Update Docker Compose to use nvidia runtime
docker-compose -f docker-compose-mlflow.yml --profile gpu-training up -d
```

## 📚 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

## 🤝 Contributing

Contributions to improve the MLflow integration are welcome! Please follow the guidelines in `CONTRIBUTING.md`.

## 📝 License

This MLflow integration is part of the RipCatch project. See `LICENSE` for details.

---

**Happy Experimenting! 🚀**
