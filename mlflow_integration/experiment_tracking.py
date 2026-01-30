"""
RipCatch v2.0 - MLflow Experiment Tracking Utilities
Helper functions for experiment management and analysis

Author: Sravan Kumar
GitHub: https://github.com/naga-narala/RipCatch
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ExperimentTracker:
    """Utility class for MLflow experiment tracking"""
    
    def __init__(self, tracking_uri="http://localhost:5000"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def list_experiments(self):
        """List all experiments"""
        experiments = self.client.search_experiments()
        
        exp_data = []
        for exp in experiments:
            exp_data.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            })
        
        df = pd.DataFrame(exp_data)
        print("\n📊 Available Experiments:")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        return df
    
    def get_runs(self, experiment_name="RipCatch-v2.0", max_results=100):
        """Get all runs for an experiment"""
        experiment = self.client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found")
            return None
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        
        run_data = []
        for run in runs:
            run_data.append({
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(run.info.start_time/1000),
                "duration_min": (run.info.end_time - run.info.start_time) / 60000 if run.info.end_time else None,
                **run.data.metrics,
                **run.data.params
            })
        
        df = pd.DataFrame(run_data)
        print(f"\n📊 Runs for experiment '{experiment_name}':")
        print("="*80)
        print(df.head(10).to_string(index=False))
        print(f"\n... showing 10 of {len(df)} total runs")
        print("="*80 + "\n")
        
        return df
    
    def compare_runs(self, run_ids, metrics=None):
        """Compare specific runs"""
        if metrics is None:
            metrics = ["mAP50", "mAP50-95", "precision", "recall", "f1_score"]
        
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            run_info = {
                "run_id": run_id[:8],  # Short ID
                "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                "status": run.info.status
            }
            
            # Add metrics
            for metric in metrics:
                run_info[metric] = run.data.metrics.get(metric, None)
            
            # Add key params
            run_info["epochs"] = run.data.params.get("epochs", None)
            run_info["batch_size"] = run.data.params.get("batch_size", None)
            run_info["lr0"] = run.data.params.get("lr0", None)
            
            comparison_data.append(run_info)
        
        df = pd.DataFrame(comparison_data)
        
        print("\n📊 Run Comparison:")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        # Create visualization
        self.visualize_comparison(df, metrics)
        
        return df
    
    def visualize_comparison(self, comparison_df, metrics):
        """Visualize run comparison"""
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            data = comparison_df[[metric, 'run_name']].dropna()
            if not data.empty:
                data.plot(x='run_name', y=metric, kind='bar', ax=ax, legend=False)
                ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_xlabel('')
                ax.grid(axis='y', alpha=0.3)
                
                # Rotate x labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("run_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved: {plot_path}")
        plt.show()
    
    def get_best_run(self, experiment_name="RipCatch-v2.0", metric="mAP50", ascending=False):
        """Get the best run based on a metric"""
        runs_df = self.get_runs(experiment_name)
        
        if runs_df is None or metric not in runs_df.columns:
            print(f"❌ Metric '{metric}' not found")
            return None
        
        best_run = runs_df.sort_values(by=metric, ascending=ascending).iloc[0]
        
        print(f"\n🏆 Best run based on {metric}:")
        print("="*80)
        print(f"Run Name: {best_run['run_name']}")
        print(f"Run ID: {best_run['run_id']}")
        print(f"{metric}: {best_run[metric]:.4f}")
        print("="*80 + "\n")
        
        return best_run
    
    def delete_runs(self, run_ids):
        """Delete specific runs"""
        for run_id in run_ids:
            self.client.delete_run(run_id)
            print(f"🗑️  Deleted run: {run_id}")
        
        print(f"✅ Deleted {len(run_ids)} runs")
    
    def export_runs_to_csv(self, experiment_name="RipCatch-v2.0", output_file="mlflow_runs.csv"):
        """Export runs to CSV"""
        runs_df = self.get_runs(experiment_name)
        
        if runs_df is not None:
            runs_df.to_csv(output_file, index=False)
            print(f"✅ Exported {len(runs_df)} runs to {output_file}")
            return output_file
        
        return None
    
    def get_model_versions(self, model_name="RipCatch-YOLOv8"):
        """Get all versions of a registered model"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            version_data = []
            for v in versions:
                version_data.append({
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "created": datetime.fromtimestamp(v.creation_timestamp/1000),
                    "status": v.status
                })
            
            df = pd.DataFrame(version_data)
            print(f"\n📦 Model versions for '{model_name}':")
            print("="*80)
            print(df.to_string(index=False))
            print("="*80 + "\n")
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting model versions: {e}")
            return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MLflow Experiment Tracking Utilities')
    parser.add_argument('--list-experiments', action='store_true', help='List all experiments')
    parser.add_argument('--get-runs', type=str, help='Get runs for experiment')
    parser.add_argument('--compare-runs', nargs='+', help='Compare specific run IDs')
    parser.add_argument('--best-run', type=str, help='Get best run for experiment')
    parser.add_argument('--metric', type=str, default='mAP50', help='Metric for best run')
    parser.add_argument('--export', type=str, help='Export runs to CSV for experiment')
    parser.add_argument('--model-versions', type=str, help='Get versions for model name')
    
    args = parser.parse_args()
    
    tracker = ExperimentTracker()
    
    if args.list_experiments:
        tracker.list_experiments()
    
    if args.get_runs:
        tracker.get_runs(experiment_name=args.get_runs)
    
    if args.compare_runs:
        tracker.compare_runs(run_ids=args.compare_runs)
    
    if args.best_run:
        tracker.get_best_run(experiment_name=args.best_run, metric=args.metric)
    
    if args.export:
        tracker.export_runs_to_csv(experiment_name=args.export)
    
    if args.model_versions:
        tracker.get_model_versions(model_name=args.model_versions)
