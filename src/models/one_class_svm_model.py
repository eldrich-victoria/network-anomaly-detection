import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import joblib
from sklearn.svm import OneClassSVM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("OneClassSVMModel")

class OneClassSVMTrainer:
    """
    Trains and saves a One-Class SVM model using a subsample 
    of normal network traffic.
    """
    def __init__(self, processed_train_path: Path, models_dir: Path, sample_size: int = 5000) -> None:
        self.processed_train_path = Path(processed_train_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "one_class_svm.pkl"
        self.sample_size = sample_size

    def load_normal_data(self) -> pd.DataFrame:
        """Loads a subsample of normal traffic flows."""
        logger.info(f"Loading preprocessed train data from {self.processed_train_path}...")
        if not self.processed_train_path.exists():
            raise FileNotFoundError(f"Processed train data not found at {self.processed_train_path}")
            
        df = pd.read_csv(self.processed_train_path)
        
        # Filter for normal traffic only (label == 0)
        normal_df = df[df['label'] == 0]
        logger.info(f"Total normal traffic samples available: {len(normal_df)}")
        
        # Subsample to avoid long execution times
        if len(normal_df) > self.sample_size:
            normal_df = normal_df.sample(n=self.sample_size, random_state=42)
            logger.info(f"Subsampled normal traffic to: {self.sample_size} samples")
            
        X_normal = normal_df.drop(columns=['label', 'attack_cat'], errors='ignore')
        return X_normal

    def train(self) -> None:
        """Trains the One-Class SVM model and saves it."""
        try:
            X_normal = self.load_normal_data()
            
            logger.info("Initializing One-Class SVM...")
            model = OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=0.05 # upper bound on fraction of training errors, lower bound on fraction of support vectors
            )
            
            logger.info("Fitting One-Class SVM model...")
            model.fit(X_normal)
            
            logger.info(f"Saving model to {self.model_path}...")
            joblib.dump(model, self.model_path)
            logger.info("One-Class SVM training completed successfully.")
            
        except Exception as e:
            logger.error(f"Error training One-Class SVM: {e}", exc_info=True)
            raise

def main() -> None:
    train_path = Path("data/processed/train_processed.csv")
    models_dir = Path("models")
    
    trainer = OneClassSVMTrainer(train_path, models_dir)
    try:
        trainer.train()
    except Exception as e:
        print(f"One-Class SVM training failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
