import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_curve, 
    auc,
    classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ModelEvaluation")

# Import the PyTorch model class to re-create it
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.autoencoder_model import PyTorchAutoencoder

class ModelEvaluator:
    """
    Evaluates the performance of the anomaly detection models 
    and saves metrics and visualizations.
    """
    def __init__(self, test_data_path: Path, models_dir: Path, output_dir: Path) -> None:
        self.test_data_path = Path(test_data_path)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette for charts
        self.colors = {
            "Isolation Forest": "#1F77B4",
            "Local Outlier Factor": "#FF7F0E",
            "One-Class SVM": "#2CA02C",
            "Autoencoder": "#D62728"
        }
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads features and labels from the processed test set."""
        logger.info(f"Loading preprocessed test data from {self.test_data_path}...")
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"Processed test data not found at {self.test_data_path}")
            
        df = pd.read_csv(self.test_data_path)
        X = df.drop(columns=['label', 'attack_cat'], errors='ignore')
        y = df['label']
        return X, y

    def load_sklearn_model(self, filename: str) -> Any:
        """Loads a scikit-learn model from a pickle file."""
        path = self.models_dir / filename
        logger.info(f"Loading scikit-learn model from {path}...")
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        return joblib.load(path)

    def load_autoencoder(self, filename: str) -> Tuple[PyTorchAutoencoder, float]:
        """Loads the PyTorch Autoencoder and its threshold."""
        path = self.models_dir / filename
        logger.info(f"Loading PyTorch Autoencoder from {path}...")
        if not path.exists():
            raise FileNotFoundError(f"Autoencoder file not found at {path}")
            
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        input_dim = checkpoint["input_dim"]
        threshold = checkpoint["threshold"]
        
        model = PyTorchAutoencoder(input_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        return model, threshold

    def evaluate_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Runs evaluation for all four models and stores metrics, 
        predictions, and scores.
        """
        X_test, y_test = self.load_data()
        X_test_arr = X_test.values.astype(np.float32)
        
        results = {}
        
        # 1. Isolation Forest
        try:
            iforest = self.load_sklearn_model("isolation_forest.pkl")
            # predict returns 1 (normal) or -1 (anomaly)
            preds_raw = iforest.predict(X_test)
            y_pred = np.where(preds_raw == -1, 1, 0)
            scores = -iforest.decision_function(X_test)
            
            results["Isolation Forest"] = {
                "y_pred": y_pred,
                "scores": scores
            }
        except Exception as e:
            logger.error(f"Error evaluating Isolation Forest: {e}")
            
        # 2. Local Outlier Factor
        try:
            lof = self.load_sklearn_model("lof.pkl")
            preds_raw = lof.predict(X_test)
            y_pred = np.where(preds_raw == -1, 1, 0)
            scores = -lof.decision_function(X_test)
            
            results["Local Outlier Factor"] = {
                "y_pred": y_pred,
                "scores": scores
            }
        except Exception as e:
            logger.error(f"Error evaluating Local Outlier Factor: {e}")

        # 3. One-Class SVM
        try:
            oc_svm = self.load_sklearn_model("one_class_svm.pkl")
            preds_raw = oc_svm.predict(X_test)
            y_pred = np.where(preds_raw == -1, 1, 0)
            scores = -oc_svm.decision_function(X_test)
            
            results["One-Class SVM"] = {
                "y_pred": y_pred,
                "scores": scores
            }
        except Exception as e:
            logger.error(f"Error evaluating One-Class SVM: {e}")

        # 4. Autoencoder
        try:
            ae, threshold = self.load_autoencoder("autoencoder.keras")
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test_arr)
                reconstructed = ae(X_test_tensor)
                # Compute row-wise MSE
                mse = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).numpy()
                
            y_pred = np.where(mse > threshold, 1, 0)
            
            results["Autoencoder"] = {
                "y_pred": y_pred,
                "scores": mse
            }
        except Exception as e:
            logger.error(f"Error evaluating Autoencoder: {e}")

        # Calculate metrics for each model
        evaluation_report = {}
        for name, data in results.items():
            y_pred = data["y_pred"]
            scores = data["scores"]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            logger.info(f"=== {name} Metrics ===")
            logger.info(f"Accuracy : {acc:.4f}")
            logger.info(f"Precision: {prec:.4f}")
            logger.info(f"Recall   : {rec:.4f}")
            logger.info(f"F1-Score : {f1:.4f}")
            
            # Print text classification report
            print(f"\nClassification Report for {name}:")
            print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
            
            evaluation_report[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "scores": scores,
                "y_pred": y_pred
            }
            
        return evaluation_report

    def plot_confusion_matrices(self, evaluation_report: Dict[str, Dict[str, Any]], y_test: pd.Series) -> None:
        """Generates a side-by-side subplot panel of confusion matrices."""
        logger.info("Generating confusion matrices plot...")
        models_to_plot = list(evaluation_report.keys())
        n_models = len(models_to_plot)
        
        if n_models == 0:
            logger.warning("No model reports found to plot confusion matrices.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, name in enumerate(models_to_plot):
            cm = evaluation_report[name]["confusion_matrix"]
            
            # Normalised version or raw counts?
            # We display raw counts and percentages
            group_names = ['True Normal', 'False Anomaly', 'False Normal', 'True Anomaly']
            group_counts = [f"{value:d}" for value in cm.flatten()]
            group_percentages = [f"{value:.2%}" for value in cm.flatten()/np.sum(cm)]
            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(2,2)
            
            sns.heatmap(
                cm, 
                annot=labels, 
                fmt="", 
                cmap="Blues", 
                ax=axes[idx],
                cbar=False,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                annot_kws={"size": 11}
            )
            axes[idx].set_title(f"{name} Confusion Matrix", fontsize=13, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=11)
            axes[idx].set_ylabel('True Label', fontsize=11)
            
        # Hide any unused axes
        for i in range(n_models, 4):
            fig.delaxes(axes[i])
            
        plt.suptitle("Confusion Matrices Comparison", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
        plt.close()

    def plot_roc_curves(self, evaluation_report: Dict[str, Dict[str, Any]], y_test: pd.Series) -> None:
        """Generates overlaid ROC curves for all models."""
        logger.info("Generating ROC curves plot...")
        plt.figure(figsize=(10, 8))
        
        for name, report in evaluation_report.items():
            scores = report["scores"]
            
            # Calculate ROC
            fpr, tpr, _ = roc_curve(y_test, scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, 
                tpr, 
                color=self.colors.get(name, "#333333"), 
                lw=2, 
                label=f'{name} (AUC = {roc_auc:.4f})'
            )
            
        plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14, fontweight='bold', pad=15)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curve.png", dpi=150)
        plt.close()

    def save_comparison_table(self, evaluation_report: Dict[str, Dict[str, Any]]) -> None:
        """Saves models comparison metrics to CSV file."""
        csv_path = self.output_dir / "metrics_comparison.csv"
        logger.info(f"Saving metrics comparison table to {csv_path}...")
        
        comparison_data = []
        for name, report in evaluation_report.items():
            comparison_data.append({
                "Model": name,
                "Accuracy": report["accuracy"],
                "Precision": report["precision"],
                "Recall": report["recall"],
                "F1-Score": report["f1_score"]
            })
            
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv(csv_path, index=False)
        logger.info("Metrics comparison table saved.")

    def run_evaluation(self) -> None:
        """Executes full evaluation pipeline."""
        X_test, y_test = self.load_data()
        
        # Run predictions and metrics calculation
        report = self.evaluate_models()
        
        # Save plots
        self.plot_confusion_matrices(report, y_test)
        self.plot_roc_curves(report, y_test)
        
        # Save comparison table
        self.save_comparison_table(report)
        logger.info("Milestone 5 execution completed successfully.")

def main() -> None:
    test_data_path = Path("data/processed/test_processed.csv")
    models_dir = Path("models")
    output_dir = Path("reports/metrics")
    
    evaluator = ModelEvaluator(test_data_path, models_dir, output_dir)
    try:
        evaluator.run_evaluation()
        print("Milestone 5 execution complete! Check reports/metrics/ for results.")
    except Exception as e:
        print(f"Milestone 5 execution failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
