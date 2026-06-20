import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AlertSystem")

class AlertSystem:
    """
    Manages threat severity classification and records security incidents 
    to the alerts.csv log file.
    """
    def __init__(self, alerts_csv_path: Path = Path("alerts.csv")) -> None:
        self.alerts_csv_path = Path(alerts_csv_path)
        
    def calculate_severity(
        self, 
        model_name: str, 
        score: float, 
        threshold: float, 
        std_val: float = 0.05
    ) -> str:
        """
        Determines the severity level (Low, Medium, High, Critical) of an anomaly
        based on the model type and distance from the decision boundary.
        
        Args:
            model_name (str): Name of the model (e.g., 'Autoencoder').
            score (float): Anomaly score/value.
            threshold (float): Anomaly decision threshold.
            std_val (float): Standard deviation of normal scores (for Autoencoder/novelty models).
            
        Returns:
            str: Severity Level ('Low', 'Medium', 'High', 'Critical')
        """
        # If score is below threshold, it's normal traffic (Low severity / Informational)
        if score <= threshold:
            return "Low"
            
        excess = score - threshold
        
        if model_name == "Autoencoder":
            # For Autoencoder, reconstruction error is compared to the threshold.
            # std_val is the standard deviation of normal training reconstruction error (approx 0.062).
            if excess <= std_val:
                return "Medium"
            elif excess <= 3 * std_val:
                return "High"
            else:
                return "Critical"
                
        elif model_name == "Isolation Forest":
            # IF scores = -decision_function. Anomaly if > 0.
            # Max possible excess is usually around 0.5.
            if excess <= 0.05:
                return "Medium"
            elif excess <= 0.15:
                return "High"
            else:
                return "Critical"
                
        elif model_name == "Local Outlier Factor":
            # LOF scores = -decision_function. Anomaly if > 0.
            if excess <= 0.3:
                return "Medium"
            elif excess <= 1.0:
                return "High"
            else:
                return "Critical"
                
        elif model_name == "One-Class SVM":
            # OC-SVM scores = -decision_function. Anomaly if > 0.
            if excess <= 1.0:
                return "Medium"
            elif excess <= 3.0:
                return "High"
            else:
                return "Critical"
                
        else:
            # Fallback
            return "Medium"

    def log_alerts(self, df_anomalies: pd.DataFrame) -> None:
        """
        Appends new anomaly records to the alerts.csv log file.
        
        Args:
            df_anomalies (pd.DataFrame): Dataframe containing anomalous traffic flows
                                         with columns like: Timestamp, Model, Score, Severity, 
                                         and optional network characteristics (e.g. dur, proto, service, sbytes, dbytes).
        """
        if df_anomalies.empty:
            return
            
        logger.info(f"Logging {len(df_anomalies)} anomalies to alert log...")
        
        # Ensure timestamp column is present
        if "Timestamp" not in df_anomalies.columns:
            df_anomalies = df_anomalies.copy()
            df_anomalies["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        # Select key columns to keep in alerts log
        columns_to_save = ["Timestamp", "Model", "Anomaly_Score", "Severity"]
        # Include basic network flow columns if present for context
        network_cols = ["dur", "proto", "service", "state", "sbytes", "dbytes", "rate", "attack_cat"]
        for col in network_cols:
            if col in df_anomalies.columns:
                columns_to_save.append(col)
                
        df_to_log = df_anomalies[columns_to_save]
        
        try:
            if self.alerts_csv_path.exists():
                df_to_log.to_csv(self.alerts_csv_path, mode='a', header=False, index=False)
            else:
                df_to_log.to_csv(self.alerts_csv_path, mode='w', header=True, index=False)
            logger.info(f"Successfully wrote alerts to {self.alerts_csv_path}")
        except Exception as e:
            logger.error(f"Error logging alerts: {e}", exc_info=True)
            raise

def main() -> None:
    # Test alert system
    alert_sys = AlertSystem(Path("alerts.csv"))
    score = 0.15
    thresh = 0.05
    std = 0.03
    severity = alert_sys.calculate_severity("Autoencoder", score, thresh, std)
    print(f"Calculated test severity for Autoencoder (Score: {score}, Thresh: {thresh}, Std: {std}): {severity}")

if __name__ == "__main__":
    main()
