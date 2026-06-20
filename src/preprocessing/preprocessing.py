import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Preprocessing")

import sys
from pathlib import Path
# Add workspace root to sys.path to resolve 'src' package imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.encoders import RobustLabelEncoder

class Preprocessor:
    """
    Clean, encode, and scale the UNSW-NB15 dataset.
    """
    def __init__(self, raw_dir: Path, processed_dir: Path, models_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.encoders: Dict[str, RobustLabelEncoder] = {}
        
        # Nominal columns to encode (from features.csv)
        self.categorical_cols = ["proto", "service", "state"]
        
        # Columns to exclude from features (like ID and targets)
        self.exclude_cols = ["id", "label", "attack_cat"]
        self.numerical_cols: List[str] = []

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads train and test CSVs from data/raw."""
        train_path = self.raw_dir / "UNSW_NB15_training-set.csv"
        test_path = self.raw_dir / "UNSW_NB15_testing-set.csv"
        
        logger.info(f"Loading raw datasets from {self.raw_dir}...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        return train_df, test_df

    def clean_missing_and_duplicates(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Cleans data: removes duplicates (train only) and replaces placeholder '-' values.
        """
        df_cleaned = df.copy()
        
        # Replace service '-' indicator with 'unknown'
        if "service" in df_cleaned.columns:
            # We replace any '-' or empty spaces in service/proto/state
            for col in self.categorical_cols:
                if col in df_cleaned.columns:
                    df_cleaned[col] = df_cleaned[col].replace({"-": "unknown", "": "unknown", np.nan: "unknown"})
                    
        # Fill standard NaNs in other columns
        # Categoricals with 'unknown' and numericals with median
        for col in df_cleaned.columns:
            if col in self.exclude_cols:
                continue
            if col in self.categorical_cols:
                df_cleaned[col] = df_cleaned[col].fillna("unknown")
            else:
                # Numerical columns
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
                
        # Drop duplicates for train only to prevent overfitting
        if is_train:
            dups = df_cleaned.duplicated().sum()
            if dups > 0:
                logger.info(f"Removing {dups} duplicate rows from training set.")
                df_cleaned = df_cleaned.drop_duplicates()
                
        return df_cleaned

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fits encoders and scaler on the training set.
        """
        logger.info("Fitting Preprocessor on training data...")
        df_cleaned = self.clean_missing_and_duplicates(train_df, is_train=True)
        
        # Identify numerical columns (all columns that are not categorical and not excluded)
        self.numerical_cols = [
            col for col in df_cleaned.columns 
            if col not in self.categorical_cols and col not in self.exclude_cols
        ]
        
        logger.info(f"Categorical features: {self.categorical_cols}")
        logger.info(f"Numerical features: {len(self.numerical_cols)} columns")
        
        # Fit encoders
        for col in self.categorical_cols:
            logger.info(f"Fitting RobustLabelEncoder for: {col}")
            encoder = RobustLabelEncoder()
            encoder.fit(df_cleaned[col])
            self.encoders[col] = encoder
            
        # Fit scaler on pre-encoded categoricals and scaled numericals
        # Let's scale only numerical columns
        logger.info("Fitting StandardScaler on numerical columns...")
        self.scaler.fit(df_cleaned[self.numerical_cols])

    def transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Transforms the dataset (cleans, encodes, scales).
        """
        logger.info(f"Transforming dataset (is_train={is_train})...")
        df_cleaned = self.clean_missing_and_duplicates(df, is_train=is_train)
        df_transformed = df_cleaned.copy()
        
        # Transform categorical columns
        for col in self.categorical_cols:
            if col in df_transformed.columns:
                df_transformed[col] = self.encoders[col].transform(df_cleaned[col])
                
        # Scale numerical columns
        if len(self.numerical_cols) == 0:
            # Fallback if fit wasn't called on this instance, identify them
            self.numerical_cols = [
                c for c in df_transformed.columns 
                if c not in self.categorical_cols and c not in self.exclude_cols
            ]
            
        df_transformed[self.numerical_cols] = self.scaler.transform(df_cleaned[self.numerical_cols])
        
        # Ensure column order is consistent: numerical first, then categorical, then target columns
        final_cols = self.numerical_cols + self.categorical_cols
        for target in ["attack_cat", "label"]:
            if target in df_transformed.columns:
                final_cols.append(target)
                
        return df_transformed[final_cols]

    def save_transforms(self) -> None:
        """Saves scaler and encoders to the models directory."""
        scaler_path = self.models_dir / "scaler.pkl"
        encoders_path = self.models_dir / "encoders.pkl"
        
        logger.info(f"Saving StandardScaler to {scaler_path}")
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Saving RobustLabelEncoders to {encoders_path}")
        joblib.dump(self.encoders, encoders_path)
        
        # Save categorical/numerical column configuration for predictions
        config_path = self.models_dir / "preprocessing_config.pkl"
        joblib.dump({
            "categorical_cols": self.categorical_cols,
            "numerical_cols": self.numerical_cols,
            "exclude_cols": self.exclude_cols
        }, config_path)

    def process_and_save(self) -> None:
        """Executes the full pipeline and saves the resulting CSV files."""
        train_df, test_df = self.load_raw_data()
        
        self.fit(train_df)
        
        train_processed = self.transform(train_df, is_train=True)
        test_processed = self.transform(test_df, is_train=False)
        
        # Save processed CSVs
        train_out_path = self.processed_dir / "train_processed.csv"
        test_out_path = self.processed_dir / "test_processed.csv"
        
        logger.info(f"Saving processed train to {train_out_path}...")
        train_processed.to_csv(train_out_path, index=False)
        
        logger.info(f"Saving processed test to {test_out_path}...")
        test_processed.to_csv(test_out_path, index=False)
        
        self.save_transforms()
        logger.info("Preprocessing step successfully complete.")

def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    
    preprocessor = Preprocessor(raw_dir, processed_dir, models_dir)
    try:
        preprocessor.process_and_save()
        print("Milestone 2 execution complete! Processed files and scaler/encoders are saved.")
    except Exception as e:
        print(f"Milestone 2 execution failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
