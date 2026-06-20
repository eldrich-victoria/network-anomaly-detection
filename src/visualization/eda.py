import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EDA")

class EDAVisualizer:
    """
    Performs exploratory data analysis and generates visualizations 
    for the UNSW-NB15 dataset.
    """
    def __init__(self, raw_data_path: Path, output_dir: Path) -> None:
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        sns.set_theme(style="whitegrid", palette="muted")
        
        # Color palettes matching rich aesthetics
        self.primary_color = "#4C6EF5"   # Sleek Indigo
        self.secondary_color = "#FA5252" # Crimson/Coral for anomalies
        self.palette = [self.primary_color, self.secondary_color]
        
    def load_data(self) -> pd.DataFrame:
        """Loads and returns training data for visualization."""
        logger.info(f"Loading training data from {self.raw_data_path}...")
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Training dataset not found at {self.raw_data_path}")
        return pd.read_csv(self.raw_data_path)

    def plot_label_distribution(self, df: pd.DataFrame) -> None:
        """Generates count plot of normal vs attack traffic."""
        logger.info("Plotting label distribution...")
        plt.figure(figsize=(7, 5))
        
        # Map label to readable strings
        df_copy = df.copy()
        df_copy['Traffic Type'] = df_copy['label'].map({0: 'Normal', 1: 'Anomaly'})
        
        ax = sns.countplot(
            x='Traffic Type', 
            data=df_copy, 
            hue='Traffic Type',
            palette=self.palette,
            legend=False
        )
        
        plt.title('Traffic Category Count (Normal vs Anomaly)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Traffic Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add values on top of bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height):,}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 5),
                        textcoords='offset points')
                        
        plt.tight_layout()
        plt.savefig(self.output_dir / "label_distribution.png", dpi=150)
        plt.close()

    def plot_attack_distribution(self, df: pd.DataFrame) -> None:
        """Generates distribution of specific attack categories (excluding Normal)."""
        logger.info("Plotting attack category distribution...")
        plt.figure(figsize=(10, 6))
        
        # Filter out Normal traffic to focus on attack types
        attacks_df = df[df['attack_cat'] != 'Normal']
        
        # Get count order
        order = attacks_df['attack_cat'].value_counts().index
        
        sns.countplot(
            y='attack_cat', 
            data=attacks_df, 
            order=order, 
            hue='attack_cat',
            palette="Reds_r", 
            legend=False
        )
        
        plt.title('Distribution of Attack Categories (Anomalies Only)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Attack Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / "attack_distribution.png", dpi=150)
        plt.close()

    def plot_correlation_heatmap(self, df: pd.DataFrame, top_n: int = 12) -> None:
        """Generates correlation heatmap of top N numerical features correlated with the label."""
        logger.info("Plotting correlation heatmap...")
        
        # Get numerical columns only
        numerical_df = df.select_dtypes(include=[np.number]).drop(columns=['id'], errors='ignore')
        
        # Calculate correlation with label
        corr_with_label = numerical_df.corr()['label'].abs().sort_values(ascending=False)
        top_corr_features = corr_with_label.index[:top_n+1] # Include label itself
        
        # Compute subset correlation matrix
        corr_matrix = numerical_df[top_corr_features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f", 
            linewidths=0.5, 
            vmin=-1, 
            vmax=1,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        plt.title(f'Correlation Heatmap (Top {top_n} Features with Label)', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_heatmap.png", dpi=150)
        plt.close()

    def plot_protocol_distribution(self, df: pd.DataFrame, top_k: int = 8) -> None:
        """Generates distribution of the top protocols."""
        logger.info("Plotting protocol distribution...")
        plt.figure(figsize=(10, 5))
        
        top_protos = df['proto'].value_counts().head(top_k).index
        proto_df = df[df['proto'].isin(top_protos)]
        
        sns.countplot(
            x='proto', 
            data=proto_df, 
            order=top_protos, 
            hue='label', 
            palette=self.palette
        )
        
        plt.title(f'Traffic Count by Protocol (Top {top_k})', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Protocol', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Traffic Class', labels=['Normal', 'Anomaly'])
        plt.tight_layout()
        plt.savefig(self.output_dir / "protocol_distribution.png", dpi=150)
        plt.close()

    def plot_service_distribution(self, df: pd.DataFrame) -> None:
        """Generates distribution of network services."""
        logger.info("Plotting service distribution...")
        plt.figure(figsize=(10, 5))
        
        # Clean '-' placeholders to 'unknown' or 'other'
        df_copy = df.copy()
        df_copy['service'] = df_copy['service'].replace({'-': 'unknown'})
        
        order = df_copy['service'].value_counts().index
        
        sns.countplot(
            x='service', 
            data=df_copy, 
            order=order, 
            hue='label', 
            palette=self.palette
        )
        
        plt.title('Traffic Count by Service Type', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Service', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Traffic Class', labels=['Normal', 'Anomaly'])
        plt.tight_layout()
        plt.savefig(self.output_dir / "service_distribution.png", dpi=150)
        plt.close()

    def plot_boxplots(self, df: pd.DataFrame) -> None:
        """Generates boxplots for key features grouped by class."""
        logger.info("Plotting boxplots...")
        
        # Select key numerical features
        features = ['dur', 'spkts', 'sbytes', 'rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        df_copy = df.copy()
        df_copy['Traffic Type'] = df_copy['label'].map({0: 'Normal', 1: 'Anomaly'})
        
        for idx, feat in enumerate(features):
            if feat in df_copy.columns:
                sns.boxplot(
                    x='Traffic Type', 
                    y=feat, 
                    data=df_copy, 
                    ax=axes[idx], 
                    hue='Traffic Type',
                    palette=self.palette,
                    legend=False
                )
                axes[idx].set_title(f'Distribution of {feat}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('Traffic Type', fontsize=10)
                axes[idx].set_ylabel(feat, fontsize=10)
                
                # Apply log scale for highly skewed columns
                if df_copy[feat].max() / (df_copy[feat].min() + 1e-5) > 100:
                    axes[idx].set_yscale('log')
                    axes[idx].set_ylabel(f'{feat} (Log Scale)', fontsize=10)
                    
        plt.suptitle('Boxplots of Key Flow Features', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / "boxplots.png", dpi=150)
        plt.close()

    def plot_histograms(self, df: pd.DataFrame) -> None:
        """Generates histograms of important network variables."""
        logger.info("Plotting histograms...")
        features = ['dur', 'rate', 'smean', 'dmean']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, feat in enumerate(features):
            if feat in df.columns:
                # Plot overlaid distributions using KDE
                # Only use log scale if all values are strictly positive and highly skewed
                use_log = bool((df[feat].min() > 0) and (df[feat].max() / df[feat].min() > 100))
                sns.histplot(
                    data=df, 
                    x=feat, 
                    hue='label', 
                    kde=True, 
                    element="step", 
                    stat="density", 
                    common_norm=False, 
                    ax=axes[idx], 
                    palette=self.palette,
                    log_scale=use_log
                )
                axes[idx].set_title(f'Histogram & Density of {feat}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(feat + (' (Log Scale)' if use_log else ''), fontsize=10)
                axes[idx].set_ylabel('Density', fontsize=10)
                
                # Update legend labels
                legend = axes[idx].get_legend()
                if legend:
                    legend.set_title("Traffic Class")
                    for t, l in zip(legend.texts, ['Normal', 'Anomaly']):
                        t.set_text(l)
                        
        plt.suptitle('Histograms of Selected Network Features', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / "numerical_histograms.png", dpi=150)
        plt.close()

    def plot_scatter(self, df: pd.DataFrame) -> None:
        """Generates scatter plot of sbytes vs dbytes."""
        logger.info("Plotting scatter plot...")
        plt.figure(figsize=(10, 6))
        
        # Subsample to keep scatter plot readable
        sample_df = df.sample(n=min(len(df), 20000), random_state=42).copy()
        sample_df['Traffic Type'] = sample_df['label'].map({0: 'Normal', 1: 'Anomaly'})
        
        sns.scatterplot(
            x='sbytes', 
            y='dbytes', 
            hue='Traffic Type', 
            data=sample_df, 
            alpha=0.4, 
            palette=self.palette
        )
        
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Scatter Plot: Source Bytes vs Destination Bytes (Sample of 20k rows)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Source Bytes (Log Scale)', fontsize=12)
        plt.ylabel('Destination Bytes (Log Scale)', fontsize=12)
        plt.legend(title='Traffic Type')
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_scatter.png", dpi=150)
        plt.close()

    def run_all(self) -> None:
        """Loads data and generates all EDA charts."""
        df = self.load_data()
        
        self.plot_label_distribution(df)
        self.plot_attack_distribution(df)
        self.plot_correlation_heatmap(df)
        self.plot_protocol_distribution(df)
        self.plot_service_distribution(df)
        self.plot_boxplots(df)
        self.plot_histograms(df)
        self.plot_scatter(df)
        logger.info("EDA completed successfully. All figures saved.")

def main() -> None:
    train_path = Path("data/raw/UNSW_NB15_training-set.csv")
    output_dir = Path("reports/figures")
    
    visualizer = EDAVisualizer(train_path, output_dir)
    try:
        visualizer.run_all()
        print("Milestone 3 execution complete! Figures saved in reports/figures/.")
    except Exception as e:
        print(f"Milestone 3 execution failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
