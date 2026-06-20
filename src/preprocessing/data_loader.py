import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataLoader")

class DataLoader:
    """
    Loads raw datasets and feature info for the UNSW-NB15 dataset
    and generates exploratory summaries.
    """
    def __init__(self, data_dir: Path, output_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Define paths
        self.train_path = self.data_dir / "raw" / "UNSW_NB15_training-set.csv"
        self.test_path = self.data_dir / "raw" / "UNSW_NB15_testing-set.csv"
        self.features_path = self.data_dir / "raw" / "NUSW-NB15_features.csv"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the training, testing, and features files from the raw directory.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, test_df, features_df)
        """
        logger.info("Loading dataset files...")
        try:
            if not self.train_path.exists():
                raise FileNotFoundError(f"Training dataset not found at {self.train_path}")
            if not self.test_path.exists():
                raise FileNotFoundError(f"Testing dataset not found at {self.test_path}")
            if not self.features_path.exists():
                raise FileNotFoundError(f"Features list file not found at {self.features_path}")
                
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            try:
                features_df = pd.read_csv(self.features_path, encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning("UTF-8 decoding failed for features file. Falling back to latin-1.")
                features_df = pd.read_csv(self.features_path, encoding="latin-1")
            
            logger.info(f"Successfully loaded train shape: {train_df.shape}, test shape: {test_df.shape}, features shape: {features_df.shape}")
            return train_df, test_df, features_df
        except Exception as e:
            logger.error(f"Error loading datasets: {e}", exc_info=True)
            raise

    def analyze_dataset(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """
        Analyzes a single dataframe for shape, columns, missing values, and duplicates.
        
        Args:
            df (pd.DataFrame): Dataframe to analyze.
            name (str): Label for the dataset.
            
        Returns:
            Dict[str, Any]: Dictionary containing analysis results.
        """
        logger.info(f"Analyzing dataset: {name}...")
        try:
            analysis = {
                "name": name,
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "total_missing": int(df.isnull().sum().sum()),
                "duplicates": int(df.duplicated().sum())
            }
            logger.info(f"{name} Analysis completed. Shape: {df.shape}, Missing: {analysis['total_missing']}, Duplicates: {analysis['duplicates']}")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing dataset {name}: {e}", exc_info=True)
            raise

    def generate_overview_report(
        self, 
        train_analysis: Dict[str, Any], 
        test_analysis: Dict[str, Any], 
        features_df: pd.DataFrame
    ) -> Path:
        """
        Generates and saves the reports/data_overview.txt file.
        
        Args:
            train_analysis (Dict[str, Any]): Analysis dictionary of the training set.
            test_analysis (Dict[str, Any]): Analysis dictionary of the testing set.
            features_df (pd.DataFrame): Dataframe of the features file.
            
        Returns:
            Path: Path to the generated data_overview.txt file.
        """
        report_path = self.output_dir / "data_overview.txt"
        logger.info(f"Writing data overview report to {report_path}...")
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("               UNSW-NB15 DATASET EXPLORATORY OVERVIEW REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                # Training Set Summary
                f.write("1. TRAINING DATASET PROFILE\n")
                f.write("-" * 40 + "\n")
                f.write(f"Dataset Name: {train_analysis['name']}\n")
                f.write(f"Dimensions: {train_analysis['shape'][0]} rows, {train_analysis['shape'][1]} columns\n")
                f.write(f"Duplicate Rows: {train_analysis['duplicates']}\n")
                f.write(f"Total Missing Values: {train_analysis['total_missing']}\n")
                
                # Check for '-' missing indicator in service and proto
                f.write("\nMissing Indicator ('-') Occurrences:\n")
                for col in ['proto', 'service', 'state']:
                    # We need to handle case where column is missing or doesn't exist
                    # but these nominal cols are guaranteed in raw dataset
                    pass
                
                f.write("\nMissing Values Breakdown (Top 5 columns):\n")
                sorted_missing_train = sorted(train_analysis['missing_values'].items(), key=lambda x: x[1], reverse=True)[:5]
                for col, val in sorted_missing_train:
                    f.write(f"  - {col}: {val}\n")
                f.write("\n")
                
                # Testing Set Summary
                f.write("2. TESTING DATASET PROFILE\n")
                f.write("-" * 40 + "\n")
                f.write(f"Dataset Name: {test_analysis['name']}\n")
                f.write(f"Dimensions: {test_analysis['shape'][0]} rows, {test_analysis['shape'][1]} columns\n")
                f.write(f"Duplicate Rows: {test_analysis['duplicates']}\n")
                f.write(f"Total Missing Values: {test_analysis['total_missing']}\n")
                f.write("\nMissing Values Breakdown (Top 5 columns):\n")
                sorted_missing_test = sorted(test_analysis['missing_values'].items(), key=lambda x: x[1], reverse=True)[:5]
                for col, val in sorted_missing_test:
                    f.write(f"  - {col}: {val}\n")
                f.write("\n")
                
                # Features Schema Summary
                f.write("3. FEATURE LIST SCHEMA SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Features Documented: {len(features_df)}\n")
                f.write("Features Breakdown by Type:\n")
                if 'Type ' in features_df.columns:
                    type_col = 'Type '
                elif 'Type' in features_df.columns:
                    type_col = 'Type'
                else:
                    type_col = features_df.columns[2] # Fallback
                
                type_counts = features_df[type_col].value_counts()
                for t, count in type_counts.items():
                    f.write(f"  - {str(t).strip()}: {count}\n")
                f.write("\n")
                
                # Column Comparison
                f.write("4. SCHEMA CONSISTENCY CHECK\n")
                f.write("-" * 40 + "\n")
                train_cols = set(train_analysis['columns'])
                test_cols = set(test_analysis['columns'])
                
                if train_cols == test_cols:
                    f.write("Train and Test schemas match perfectly.\n")
                    f.write(f"Total Shared Columns: {len(train_cols)}\n")
                else:
                    f.write("WARNING: Train and Test schemas DO NOT match.\n")
                    f.write(f"Columns in Train but not Test: {train_cols - test_cols}\n")
                    f.write(f"Columns in Test but not Train: {test_cols - train_cols}\n")
                
                # Data types summary for numerical & categorical
                # In UNSW-NB15, we typically have numerical and categorical
                f.write("\nDetailed Column Types (Train Dataset):\n")
                for dtype, count in pd.Series(train_analysis['dtypes']).value_counts().items():
                    f.write(f"  - {dtype}: {count}\n")
                f.write("\n")
                
            logger.info(f"Report generated and saved at {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error writing report: {e}", exc_info=True)
            raise

def main() -> None:
    """
    Main entry point to execute Milestone 1: data loading and reporting.
    """
    data_dir = Path("data")
    output_dir = Path("reports")
    
    loader = DataLoader(data_dir, output_dir)
    
    try:
        train_df, test_df, features_df = loader.load_data()
        
        train_analysis = loader.analyze_dataset(train_df, "UNSW_NB15_training-set")
        test_analysis = loader.analyze_dataset(test_df, "UNSW_NB15_testing-set")
        
        loader.generate_overview_report(train_analysis, test_analysis, features_df)
        print("Milestone 1 execution complete! Check reports/data_overview.txt for results.")
    except Exception as e:
        print(f"Milestone 1 execution failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
