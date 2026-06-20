import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AutoencoderModel")

class PyTorchAutoencoder(nn.Module):
    """
    PyTorch implementation of the requested Autoencoder architecture:
    Input -> Dense(128, relu) -> Dense(64, relu) -> Dense(32, relu)
          -> Dense(64, relu) -> Dense(128, relu) -> Output(Linear)
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderTrainer:
    """
    Trains the PyTorch Autoencoder on normal network traffic 
    and determines the reconstruction error threshold.
    """
    def __init__(
        self, 
        processed_train_path: Path, 
        models_dir: Path,
        epochs: int = 50,
        batch_size: int = 256,
        patience: int = 5
    ) -> None:
        self.processed_train_path = Path(processed_train_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "autoencoder.keras" # Saved using torch.save
        self.meta_path = self.models_dir / "autoencoder_meta.json"
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        
        # Check GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_normal_data(self) -> np.ndarray:
        """Loads training data and filters for normal traffic only (label == 0)."""
        logger.info(f"Loading preprocessed train data from {self.processed_train_path}...")
        if not self.processed_train_path.exists():
            raise FileNotFoundError(f"Processed train data not found at {self.processed_train_path}")
            
        df = pd.read_csv(self.processed_train_path)
        
        # Filter for normal traffic
        normal_df = df[df['label'] == 0]
        logger.info(f"Loaded normal traffic data: {normal_df.shape}")
        
        X_normal = normal_df.drop(columns=['label', 'attack_cat'], errors='ignore').values.astype(np.float32)
        return X_normal

    def train(self) -> None:
        """Trains the Autoencoder and calculates anomaly threshold."""
        try:
            X_normal = self.load_normal_data()
            input_dim = X_normal.shape[1]
            logger.info(f"Autoencoder Input Dimension: {input_dim}")
            
            # Split normal data into train (80%) and validation (20%)
            np.random.seed(42)
            shuffled_indices = np.random.permutation(len(X_normal))
            split_idx = int(len(X_normal) * 0.8)
            
            train_indices = shuffled_indices[:split_idx]
            val_indices = shuffled_indices[split_idx:]
            
            X_train = X_normal[train_indices]
            X_val = X_normal[val_indices]
            
            # Create DataLoaders
            train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(X_train))
            val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(X_val))
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Initialize model
            model = PyTorchAutoencoder(input_dim).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Early stopping variables
            best_val_loss = float("inf")
            best_model_state = None
            epochs_no_improve = 0
            
            logger.info("Starting Autoencoder training...")
            for epoch in range(1, self.epochs + 1):
                model.train()
                train_loss = 0.0
                for batch_x, _ in train_loader:
                    batch_x = batch_x.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_x)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * batch_x.size(0)
                
                train_loss /= len(train_dataset)
                
                # Validation step
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x_val, _ in val_loader:
                        batch_x_val = batch_x_val.to(self.device)
                        outputs_val = model(batch_x_val)
                        loss_val = criterion(outputs_val, batch_x_val)
                        val_loss += loss_val.item() * batch_x_val.size(0)
                        
                val_loss /= len(val_dataset)
                
                if epoch % 5 == 0 or epoch == 1 or epoch == self.epochs:
                    logger.info(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
                
                # Early Stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                        break
            
            # Load best model state
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # Compute reconstruction errors on the normal training set to find anomaly threshold
            model.eval()
            with torch.no_grad():
                X_normal_tensor = torch.tensor(X_normal).to(self.device)
                reconstructed = model(X_normal_tensor)
                
                # Calculate row-wise MSE
                reconstruction_errors = torch.mean((X_normal_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
                
            # Determine threshold as the 95th percentile
            threshold = float(np.percentile(reconstruction_errors, 95))
            mean_error = float(np.mean(reconstruction_errors))
            std_error = float(np.std(reconstruction_errors))
            max_error = float(np.max(reconstruction_errors))
            
            logger.info(f"Reconstruction Error Stats on Normal Train:")
            logger.info(f"  - Mean: {mean_error:.6f}")
            logger.info(f"  - Std: {std_error:.6f}")
            logger.info(f"  - Max: {max_error:.6f}")
            logger.info(f"  - 95th Percentile (Threshold): {threshold:.6f}")
            
            # Save weights, input_dim, threshold, and error stats
            logger.info(f"Saving Autoencoder state dict and metadata to {self.model_path}...")
            save_dict = {
                "input_dim": input_dim,
                "model_state_dict": model.state_dict(),
                "threshold": threshold,
                "mean_error": mean_error,
                "std_error": std_error
            }
            torch.save(save_dict, self.model_path)
            
            # Also save JSON metadata for easy reading by other scripts without torch
            meta_dict = {
                "input_dim": input_dim,
                "threshold": threshold,
                "mean_error": mean_error,
                "std_error": std_error
            }
            with open(self.meta_path, "w") as f:
                json.dump(meta_dict, f, indent=4)
                
            logger.info("Autoencoder training completed successfully.")
            
        except Exception as e:
            logger.error(f"Error training Autoencoder: {e}", exc_info=True)
            raise

def main() -> None:
    train_path = Path("data/processed/train_processed.csv")
    models_dir = Path("models")
    
    trainer = AutoencoderTrainer(train_path, models_dir)
    try:
        trainer.train()
    except Exception as e:
        print(f"Autoencoder training failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
