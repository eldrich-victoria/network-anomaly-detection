import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Union, List
import pandas as pd
import numpy as np
import joblib
import torch
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PredictPipeline")

import sys
# Add workspace root to sys.path to resolve 'src' package imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.preprocessing.encoders import RobustLabelEncoder
from src.models.autoencoder_model import PyTorchAutoencoder
from src.utils.alert_system import AlertSystem

class AnomalyPredictor:
    """
    Handles end-to-end anomaly prediction on raw input data
    using saved preprocessors and trained models.
    """
    def __init__(self, models_dir: Path = Path("models")) -> None:
        self.models_dir = Path(models_dir)
        
        # Load preprocessing configurations
        self.scaler_path = self.models_dir / "scaler.pkl"
        self.encoders_path = self.models_dir / "encoders.pkl"
        self.config_path = self.models_dir / "preprocessing_config.pkl"
        
        if not (self.scaler_path.exists() and self.encoders_path.exists() and self.config_path.exists()):
            raise FileNotFoundError("Preprocessing artifacts not found in models directory. Run preprocessing first.")
            
        self.scaler = joblib.load(self.scaler_path)
        self.encoders = joblib.load(self.encoders_path)
        self.config = joblib.load(self.config_path)
        
        self.categorical_cols = self.config["categorical_cols"]
        self.numerical_cols = self.config["numerical_cols"]
        self.exclude_cols = self.config["exclude_cols"]
        
        self.alert_system = AlertSystem(Path("alerts.csv"))

    def preprocess_input(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cleans and transforms raw input DataFrame.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (raw_df_cleaned, transformed_df)
        """
        raw_df = df.copy()
        
        # Clean missing indicators in categoricals
        for col in self.categorical_cols:
            if col in raw_df.columns:
                raw_df[col] = raw_df[col].replace({"-": "unknown", "": "unknown", np.nan: "unknown"})
            else:
                raw_df[col] = "unknown"
                
        # Fill missing numeric values with medians
        # We try to use training medians or fallback to column median
        for col in self.numerical_cols:
            if col not in raw_df.columns:
                # If feature is completely missing, fill with 0
                raw_df[col] = 0.0
            else:
                raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(0.0)
                
        # Keep copy of cleaned raw data for display purposes
        raw_display = raw_df.copy()
        
        # Encode categoricals using fitted encoders
        transformed_df = raw_df.copy()
        for col in self.categorical_cols:
            transformed_df[col] = self.encoders[col].transform(raw_df[col])
            
        # Scale numerical columns using fitted scaler
        transformed_df[self.numerical_cols] = self.scaler.transform(raw_df[self.numerical_cols])
        
        # Order columns correctly
        final_cols = self.numerical_cols + self.categorical_cols
        return raw_display, transformed_df[final_cols]

    def predict(self, df_input: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Preprocesses and predicts anomalies for the input DataFrame.
        
        Args:
            df_input (pd.DataFrame): Raw network traffic dataframe.
            model_name (str): Chosen model ('Isolation Forest', 'Local Outlier Factor', 
                              'One-Class SVM', or 'Autoencoder').
                              
        Returns:
            pd.DataFrame: Original dataframe appended with Prediction, Anomaly_Score, and Severity.
        """
        logger.info(f"Running predictions using model: {model_name}...")
        
        # 1. Preprocess input
        raw_display, X_processed = self.preprocess_input(df_input)
        X_values = X_processed.values.astype(np.float32)
        
        predictions = []
        scores = []
        severities = []
        
        # Load and run predictions based on the chosen model
        if model_name == "Isolation Forest":
            model = joblib.load(self.models_dir / "isolation_forest.pkl")
            preds_raw = model.predict(X_processed)
            y_pred = np.where(preds_raw == -1, 1, 0)
            y_score = -model.decision_function(X_processed)
            
            # Map threshold
            threshold = 0.0 # Standard threshold for decision_function anomalies
            for pred, score in zip(y_pred, y_score):
                predictions.append(int(pred))
                scores.append(float(score))
                severities.append(self.alert_system.calculate_severity("Isolation Forest", score, threshold))
                
        elif model_name == "Local Outlier Factor":
            model = joblib.load(self.models_dir / "lof.pkl")
            preds_raw = model.predict(X_processed)
            y_pred = np.where(preds_raw == -1, 1, 0)
            y_score = -model.decision_function(X_processed)
            
            threshold = 0.0
            for pred, score in zip(y_pred, y_score):
                predictions.append(int(pred))
                scores.append(float(score))
                severities.append(self.alert_system.calculate_severity("Local Outlier Factor", score, threshold))

        elif model_name == "One-Class SVM":
            model = joblib.load(self.models_dir / "one_class_svm.pkl")
            preds_raw = model.predict(X_processed)
            y_pred = np.where(preds_raw == -1, 1, 0)
            y_score = -model.decision_function(X_processed)
            
            threshold = 0.0
            for pred, score in zip(y_pred, y_score):
                predictions.append(int(pred))
                scores.append(float(score))
                severities.append(self.alert_system.calculate_severity("One-Class SVM", score, threshold))

        elif model_name == "Autoencoder":
            # Load PyTorch checkpoint
            checkpoint = torch.load(self.models_dir / "autoencoder.keras", map_location=torch.device('cpu'))
            input_dim = checkpoint["input_dim"]
            threshold = checkpoint["threshold"]
            std_error = checkpoint.get("std_error", 0.05)
            
            ae = PyTorchAutoencoder(input_dim)
            ae.load_state_dict(checkpoint["model_state_dict"])
            ae.eval()
            
            with torch.no_grad():
                X_tensor = torch.tensor(X_values)
                reconstructed = ae(X_tensor)
                mse_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
                
            for error in mse_errors:
                pred = 1 if error > threshold else 0
                predictions.append(pred)
                scores.append(float(error))
                severities.append(self.alert_system.calculate_severity("Autoencoder", error, threshold, std_error))
                
        else:
            raise ValueError(f"Unknown model name: {model_name}")
            
        # Append results to raw display
        result_df = raw_display.copy()
        result_df["Prediction"] = predictions
        result_df["Anomaly_Score"] = scores
        result_df["Severity"] = severities
        
        # Log detected anomalies (Prediction == 1) to alerts log
        anomalies_df = result_df[result_df["Prediction"] == 1].copy()
        if not anomalies_df.empty:
            anomalies_log_df = pd.DataFrame()
            anomalies_log_df["Timestamp"] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(anomalies_df)
            anomalies_log_df["Model"] = [model_name] * len(anomalies_df)
            anomalies_log_df["Anomaly_Score"] = anomalies_df["Anomaly_Score"]
            anomalies_log_df["Severity"] = anomalies_df["Severity"]
            
            # Add network context columns
            for col in ["dur", "proto", "service", "state", "sbytes", "dbytes", "rate", "attack_cat"]:
                if col in anomalies_df.columns:
                    anomalies_log_df[col] = anomalies_df[col]
                    
            self.alert_system.log_alerts(anomalies_log_df)
            
        return result_df

def main() -> None:
    # Test prediction pipeline with a small sample from testing set
    test_csv = Path("data/raw/UNSW_NB15_testing-set.csv")
    if not test_csv.exists():
        print("Testing file not found. Preprocess raw data first.")
        return
        
    df_sample = pd.read_csv(test_csv).head(5)
    
    try:
        predictor = AnomalyPredictor()
        
        # Predict using Autoencoder
        results = predictor.predict(df_sample, "Autoencoder")
        print("\n=== Prediction Results Sample (Autoencoder) ===")
        print(results[["proto", "service", "state", "Prediction", "Anomaly_Score", "Severity"]])
        
    except Exception as e:
        print(f"Prediction test failed: {e}")

if __name__ == "__main__":
    main()
