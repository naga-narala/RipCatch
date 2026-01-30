"""
RipCatch v2.0 - Model Evaluation with MLflow
Comprehensive model evaluation and comparison

Author: Sravan Kumar
GitHub: https://github.com/naga-narala/RipCatch
"""

import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow_config import MLflowConfig

class RipCatchEvaluator:
    """Model evaluation with MLflow tracking"""
    
    def __init__(self, model_path, test_data_yaml):
        self.model_path = model_path
        self.test_data_yaml = test_data_yaml
        self.model = YOLO(model_path)
        self.mlflow_config = MLflowConfig(experiment_name="RipCatch-Evaluation")
    
    def evaluate(self, run_name=None, split='test'):
        """Evaluate model and log to MLflow"""
        
        if run_name is None:
            run_name = f"eval_{Path(self.model_path).stem}"
        
        with mlflow.start_run(run_name=run_name):
            print(f"\n🔍 Starting evaluation: {run_name}\n")
            
            # Log model info
            mlflow.log_param("model_path", self.model_path)
            mlflow.log_param("test_dataset", self.test_data_yaml)
            mlflow.log_param("split", split)
            
            # Run validation
            results = self.model.val(
                data=self.test_data_yaml,
                split=split,
                save_json=True,
                save_hybrid=True,
                plots=True
            )
            
            # Log evaluation metrics
            self.log_evaluation_metrics(results)
            
            # Generate and log comparison report
            self.generate_evaluation_report(results)
            
            # Log confusion matrix and other plots
            self.log_evaluation_plots(results)
            
            print(f"\n✅ Evaluation complete!")
            print(f"📊 View results in MLflow UI")
            
            return results
    
    def log_evaluation_metrics(self, results):
        """Log all evaluation metrics"""
        metrics = {
            "test_mAP50": float(results.box.map50),
            "test_mAP50-95": float(results.box.map),
            "test_precision": float(results.box.mp),
            "test_recall": float(results.box.mr),
        }
        
        # Calculate F1 score
        precision = metrics["test_precision"]
        recall = metrics["test_recall"]
        if precision and recall:
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            metrics["test_f1_score"] = float(f1)
        
        # Log all metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log per-class metrics if available
        if hasattr(results.box, 'ap_class_index'):
            for idx, ap50 in enumerate(results.box.ap50):
                mlflow.log_metric(f"test_mAP50_class_{idx}", float(ap50))
        
        print("📈 Evaluation metrics logged to MLflow")
    
    def generate_evaluation_report(self, results):
        """Generate comprehensive evaluation report"""
        report = {
            "model_path": str(self.model_path),
            "test_dataset": str(self.test_data_yaml),
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": {
                "mAP50": float(results.box.map50),
                "mAP50-95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
            },
            "performance": {
                "inference_time_ms": results.speed.get('inference', 0),
                "preprocessing_time_ms": results.speed.get('preprocess', 0),
                "postprocessing_time_ms": results.speed.get('postprocess', 0),
            }
        }
        
        # Save report
        report_path = Path("evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        mlflow.log_artifact(str(report_path), "evaluation_reports")
        report_path.unlink()  # Clean up
        
        # Also create a readable text summary
        summary = f"""
# RipCatch v2.0 - Evaluation Report

## Model Information
- Model Path: {self.model_path}
- Test Dataset: {self.test_data_yaml}
- Evaluation Time: {report['timestamp']}

## Performance Metrics
- mAP@50: {report['metrics']['mAP50']:.4f}
- mAP@50-95: {report['metrics']['mAP50-95']:.4f}
- Precision: {report['metrics']['precision']:.4f}
- Recall: {report['metrics']['recall']:.4f}

## Inference Speed
- Preprocessing: {report['performance']['preprocessing_time_ms']:.2f} ms
- Inference: {report['performance']['inference_time_ms']:.2f} ms
- Postprocessing: {report['performance']['postprocessing_time_ms']:.2f} ms
"""
        
        summary_path = Path("evaluation_summary.txt")
        summary_path.write_text(summary)
        mlflow.log_artifact(str(summary_path), "evaluation_reports")
        summary_path.unlink()
        
        print("📊 Evaluation report saved and logged")
    
    def log_evaluation_plots(self, results):
        """Log evaluation visualizations"""
        # Confusion matrix, PR curve, etc. are saved by YOLO
        # We just need to log them
        save_dir = Path(results.save_dir)
        
        plot_files = [
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "PR_curve.png",
            "F1_curve.png",
            "P_curve.png",
            "R_curve.png"
        ]
        
        for plot_file in plot_files:
            plot_path = save_dir / plot_file
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), "evaluation_plots")
        
        print("📊 Evaluation plots logged to MLflow")


def compare_models(model_paths, test_data_yaml, model_names=None):
    """Compare multiple models using MLflow"""
    
    mlflow_config = MLflowConfig(experiment_name="RipCatch-Model-Comparison")
    
    if model_names is None:
        model_names = [Path(p).stem for p in model_paths]
    
    with mlflow.start_run(run_name="model_comparison"):
        results_dict = {}
        
        print("\n" + "="*60)
        print("🔬 Starting Model Comparison")
        print("="*60 + "\n")
        
        for model_path, model_name in zip(model_paths, model_names):
            print(f"\n📊 Evaluating: {model_name}")
            evaluator = RipCatchEvaluator(model_path, test_data_yaml)
            results = evaluator.evaluate(run_name=f"eval_{model_name}")
            results_dict[model_name] = results
        
        # Create comparison dataframe
        comparison_data = {
            model_name: {
                "mAP@50": float(results.box.map50),
                "mAP@50-95": float(results.box.map),
                "Precision": float(results.box.mp),
                "Recall": float(results.box.mr),
                "Inference (ms)": float(results.speed.get('inference', 0))
            }
            for model_name, results in results_dict.items()
        }
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        # Log comparison table
        comparison_path = Path("model_comparison.csv")
        comparison_df.to_csv(comparison_path)
        mlflow.log_artifact(str(comparison_path), "comparisons")
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RipCatch Model Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall']
        for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            comparison_df[metric].plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(comparison_df[metric]):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        viz_path = Path("model_comparison_viz.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(str(viz_path), "comparisons")
        plt.close()
        
        # Clean up
        comparison_path.unlink()
        viz_path.unlink()
        
        print("\n" + "="*60)
        print("📊 Model Comparison Results:")
        print("="*60)
        print(comparison_df.to_string())
        print("\n✅ Comparison complete! Check MLflow UI for visualizations.")
        
        return comparison_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RipCatch models with MLflow')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate')
    parser.add_argument('--compare', nargs='+', help='Multiple models to compare')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        compare_models(
            model_paths=args.compare,
            test_data_yaml=args.data
        )
    else:
        # Evaluate single model
        evaluator = RipCatchEvaluator(
            model_path=args.model,
            test_data_yaml=args.data
        )
        evaluator.evaluate(split=args.split)
