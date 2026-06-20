import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("IsolationForestModel")

class IsolationForestTrainer:
    """
    Trains and saves an Isolation Forest model for anomaly detection.
    """
    def __init__(self, processed_train_path: Path, models_dir: Path) -> None:
        self.processed_train_path = Path(processed_train_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "isolation_forest.pkl"

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads features and labels from the processed train set."""
        logger.info(f"Loading preprocessed train data from {self.processed_train_path}...")
        if not self.processed_train_path.exists():
            raise FileNotFoundError(f"Processed train data not found at {self.processed_train_path}")
            
        df = pd.read_csv(self.processed_train_path)
        X = df.drop(columns=['label', 'attack_cat'], errors='ignore')
        y = df['label']
        return X, y

    def train(self) -> None:
        """Trains the Isolation Forest model and saves it."""
        try:
            X, y = self.load_data()
            
            # Calculate contamination rate and cap at 0.49 to satisfy scikit-learn constraint (0.0, 0.5]
            raw_contamination = float(np.mean(y))
            contamination = min(0.49, raw_contamination)
            logger.info(f"Training dataset contains {raw_contamination:.4f} anomaly ratio. Using capped contamination: {contamination:.4f}")
            
            logger.info("Initializing Isolation Forest...")
            model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            
            logger.info("Fitting Isolation Forest model...")
            model.fit(X)
            
            logger.info(f"Saving model to {self.model_path}...")
            joblib.dump(model, self.model_path)
            logger.info("Isolation Forest training completed successfully.")
            
        except Exception as e:
            logger.error(f"Error training Isolation Forest: {e}", exc_info=True)
            raise

# Import Tuple here to avoid type hint problems in python
from typing import Tuple

def main() -> None:
    train_path = Path("data/processed/train_processed.csv")
    models_dir = Path("models")
    
    trainer = IsolationForestTrainer(train_path, models_dir)
    try:
        trainer.train()
    except Exception as e:
        print(f"Isolation Forest training failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
