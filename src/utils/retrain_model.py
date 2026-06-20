import logging
import sys
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RetrainModel")

# Add workspace root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import existing pipeline components
from src.preprocessing.preprocessing import Preprocessor
from src.models.isolation_forest_model import IsolationForestTrainer
from src.models.lof_model import LOFTrainer
from src.models.one_class_svm_model import OneClassSVMTrainer
from src.models.autoencoder_model import AutoencoderTrainer
from src.evaluation.evaluate_models import ModelEvaluator

class RetrainingOrchestrator:
    """
    Orchestrates the entire data preparation, retraining, and 
    evaluation pipeline when a new dataset is supplied.
    """
    def __init__(
        self, 
        raw_train_path: Path, 
        raw_test_path: Path,
        models_dir: Path = Path("models"),
        processed_dir: Path = Path("data/processed"),
        reports_dir: Path = Path("reports")
    ) -> None:
        self.raw_train_path = Path(raw_train_path)
        self.raw_test_path = Path(raw_test_path)
        self.models_dir = Path(models_dir)
        self.processed_dir = Path(processed_dir)
        self.reports_dir = Path(reports_dir)
        
        self.processed_train_file = self.processed_dir / "train_processed.csv"
        self.processed_test_file = self.processed_dir / "test_processed.csv"

    def run_preprocessing(self) -> None:
        """Executes preprocessing on new raw files and updates scalers/encoders."""
        logger.info("--- Step 1: Preprocessing New Datasets ---")
        try:
            # Instantiate Preprocessor.
            # Note: Preprocessor loads from raw_train_path.parent inside.
            # We override the loading paths in Preprocessor or pass the correct raw directory.
            # The Preprocessor class takes (raw_dir, processed_dir, models_dir).
            raw_dir = self.raw_train_path.parent
            preprocessor = Preprocessor(raw_dir, self.processed_dir, self.models_dir)
            preprocessor.process_and_save()
            logger.info("Preprocessing complete. Scaler and encoders updated.")
        except Exception as e:
            logger.error(f"Preprocessing step failed during retraining: {e}", exc_info=True)
            raise

    def retrain_models(self) -> None:
        """Retrains all four anomaly detection models on the newly processed training data."""
        logger.info("--- Step 2: Retraining All 4 Anomaly Models ---")
        try:
            # 1. Isolation Forest
            logger.info("Retraining Isolation Forest...")
            iforest_trainer = IsolationForestTrainer(self.processed_train_file, self.models_dir)
            iforest_trainer.train()
            
            # 2. Local Outlier Factor
            logger.info("Retraining Local Outlier Factor...")
            lof_trainer = LOFTrainer(self.processed_train_file, self.models_dir)
            lof_trainer.train()
            
            # 3. One-Class SVM
            logger.info("Retraining One-Class SVM...")
            ocsvm_trainer = OneClassSVMTrainer(self.processed_train_file, self.models_dir)
            ocsvm_trainer.train()
            
            # 4. Autoencoder
            logger.info("Retraining PyTorch Autoencoder...")
            ae_trainer = AutoencoderTrainer(
                self.processed_train_file, 
                self.models_dir,
                epochs=15, # Use 15 epochs during retraining for speed/responsiveness
                batch_size=256
            )
            ae_trainer.train()
            
            logger.info("All models retrained and overwritten successfully.")
        except Exception as e:
            logger.error(f"Model retraining failed: {e}", exc_info=True)
            raise

    def evaluate_new_models(self) -> None:
        """Runs the evaluation script to update all comparison figures and metrics tables."""
        logger.info("--- Step 3: Re-evaluating Updated Models ---")
        try:
            metrics_dir = self.reports_dir / "metrics"
            evaluator = ModelEvaluator(self.processed_test_file, self.models_dir, metrics_dir)
            evaluator.run_evaluation()
            logger.info("Model evaluation updated successfully.")
        except Exception as e:
            logger.error(f"Re-evaluation failed during retraining: {e}", exc_info=True)
            raise

    def run_update_pipeline(self) -> None:
        """Orchestrates the entire update pipeline sequentially."""
        logger.info("Starting System Model Update Pipeline...")
        self.run_preprocessing()
        self.retrain_models()
        self.evaluate_new_models()
        logger.info("System Model Update Pipeline finished successfully. All components are up-to-date!")

def main() -> None:
    # Set default paths
    raw_train = Path("data/raw/UNSW_NB15_training-set.csv")
    raw_test = Path("data/raw/UNSW_NB15_testing-set.csv")
    
    orchestrator = RetrainingOrchestrator(raw_train, raw_test)
    try:
        # We can test retraining. It will rerun training of all models and update evaluations.
        # Let's run the update pipeline to make sure it works!
        orchestrator.run_update_pipeline()
        print("Milestone 9 execution complete! All models updated and re-evaluated.")
    except Exception as e:
        print(f"Retraining pipeline failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
