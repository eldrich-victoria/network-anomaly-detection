import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
import torch
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_curve, auc

# Ensure correct path resolution
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.autoencoder_model import PyTorchAutoencoder

# Define aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'legend.fontsize': 10,
})

# Professional cohesive color palette (Cool Tech theme)
COLORS = {
    "Local Outlier Factor": "#2B5C8F",   # Slate Blue (Rank 1)
    "Autoencoder": "#439C93",            # Teal (Rank 2)
    "One-Class SVM": "#E8A33B",          # Warm Ochre (Rank 3)
    "Isolation Forest": "#D95D55"        # Coral (Rank 4)
}

def lof_f1_scorer(estimator, X, y):
    # Predict returns 1 (normal) or -1 (anomaly)
    preds_raw = estimator.predict(X)
    y_pred = np.where(preds_raw == -1, 1, 0)
    return f1_score(y, y_pred, zero_division=0)

def main():
    # Setup directories
    reports_dir = Path("reports/performance")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path("reports/metrics/metrics_comparison.csv")
    
    if not metrics_csv.exists():
        print(f"Error: Metrics comparison file not found at {metrics_csv}")
        sys.exit(1)
        
    # Read metrics
    df_metrics = pd.read_csv(metrics_csv)
    print("Read metrics comparison:")
    print(df_metrics)
    
    # 1. Generate individual metric comparison plots
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        plt.figure(figsize=(7, 5))
        df_sorted = df_metrics.sort_values(by=metric, ascending=False)
        
        barplot = sns.barplot(
            x="Model", 
            y=metric, 
            data=df_sorted, 
            palette=[COLORS[m] for m in df_sorted["Model"]],
            hue="Model",
            legend=False
        )
        
        for p in barplot.patches:
            height = p.get_height()
            barplot.annotate(
                f'{height:.2%}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', 
                va='bottom',
                xytext=(0, 5),
                textcoords='offset points',
                fontweight='semibold',
                size=10
            )
            
        plt.title(f"Model Comparison: {metric}", fontweight='bold', pad=15)
        plt.xlabel("Anomaly Detection Model", labelpad=10)
        plt.ylabel(metric, labelpad=10)
        plt.ylim(0, 1.05)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.tight_layout()
        
        filename = f"{metric.lower().replace('-', '_')}_comparison.png"
        plt.savefig(reports_dir / filename, dpi=150)
        plt.close()
        print(f"Saved {filename}")

    # 2. Combined metrics comparison plot
    plt.figure(figsize=(10, 6))
    df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    barplot = sns.barplot(
        x="Metric", 
        y="Score", 
        hue="Model", 
        data=df_melted, 
        palette=COLORS,
        edgecolor="white",
        linewidth=1
    )
    
    for p in barplot.patches:
        height = p.get_height()
        if height > 0.01:
            barplot.annotate(
                f'{height:.1%}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', 
                va='bottom',
                xytext=(0, 3),
                textcoords='offset points',
                fontweight='normal',
                size=8
            )
            
    plt.title("Comparative Performance Across All Evaluation Metrics", fontweight='bold', pad=15)
    plt.xlabel("Evaluation Metric", labelpad=10)
    plt.ylabel("Score", labelpad=10)
    plt.ylim(0, 1.1)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.legend(title="Anomaly Detection Models", loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(reports_dir / "combined_metrics_comparison.png", dpi=150)
    plt.close()
    print("Saved combined_metrics_comparison.png")

    # 3. Model Ranking (F1-score based ranking visual)
    df_ranked = df_metrics.copy()
    df_ranked["Rank"] = df_ranked["F1-Score"].rank(ascending=False, method="min").astype(int)
    df_ranked = df_ranked.sort_values(by="F1-Score", ascending=True)
    
    plt.figure(figsize=(8, 4.5))
    barplot = sns.barplot(
        x="F1-Score", 
        y="Model", 
        data=df_ranked, 
        palette=[COLORS[m] for m in df_ranked["Model"]],
        hue="Model",
        legend=False
    )
    
    for i, p in enumerate(barplot.patches):
        width = p.get_width()
        rank = df_ranked.iloc[i]["Rank"]
        barplot.text(
            width + 0.02, 
            p.get_y() + p.get_height() / 2, 
            f'Rank {rank} (F1: {width:.2%})', 
            ha='left', 
            va='center', 
            fontweight='bold',
            size=10
        )
        
    plt.title("Model Performance Ranking (Based on F1-Score)", fontweight='bold', pad=15)
    plt.xlabel("F1-Score", labelpad=10)
    plt.ylabel("Model", labelpad=10)
    plt.xlim(0, 1.05)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    plt.tight_layout()
    plt.savefig(reports_dir / "model_ranking.png", dpi=150)
    plt.savefig(reports_dir / "model_performance_ranking.png", dpi=150)
    plt.close()
    print("Saved model_ranking.png and model_performance_ranking.png")

    # 4. Generate summary CSV
    df_summary = df_metrics.copy()
    df_summary["Rank"] = df_summary["F1-Score"].rank(ascending=False, method="min").astype(int)
    df_summary = df_summary.sort_values(by="Rank")
    df_summary.to_csv(reports_dir / "performance_summary.csv", index=False)
    print("Saved performance_summary.csv")

    # 5. Model Comparison Table (publication-quality rendered image)
    plt.figure(figsize=(9.5, 3))
    plt.axis('off')
    
    cell_text = []
    for _, row in df_summary.iterrows():
        cell_text.append([
            row["Model"],
            f'{row["Accuracy"]:.2%}',
            f'{row["Precision"]:.2%}',
            f'{row["Recall"]:.2%}',
            f'{row["F1-Score"]:.2%}',
            f'# {row["Rank"]}'
        ])
        
    columns = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "Overall Rank"]
    
    table = plt.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center"
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(color="white", fontweight="bold", size=11)
            cell.set_facecolor("#1B365D")
            cell.set_edgecolor("#1B365D")
        else:
            cell.set_edgecolor("#E0E0E0")
            if row_idx % 2 == 0:
                cell.set_facecolor("#F7F9FB")
            else:
                cell.set_facecolor("white")
                
            if row_idx == 1:
                cell.set_text_props(fontweight="bold")
                
    plt.title("UNSW-NB15 Anomaly Detection Performance Matrix Summary", fontweight='bold', pad=10, y=0.85)
    plt.tight_layout()
    plt.savefig(reports_dir / "model_comparison_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved model_comparison_table.png")

    # 6. Feature Importance and SHAP Explainability on Local Outlier Factor (LOF)
    print("Starting Feature Importance and SHAP generation...")
    df_test = pd.read_csv("data/processed/test_processed.csv")
    X = df_test.drop(columns=['label', 'attack_cat'], errors='ignore')
    y = df_test['label']
    
    models_dir = Path("models")
    iforest = joblib.load(models_dir / "isolation_forest.pkl")
    lof = joblib.load(models_dir / "lof.pkl")
    oc_svm = joblib.load(models_dir / "one_class_svm.pkl")
    
    checkpoint = torch.load(models_dir / "autoencoder.keras", map_location=torch.device('cpu'))
    ae = PyTorchAutoencoder(checkpoint["input_dim"])
    ae.load_state_dict(checkpoint["model_state_dict"])
    ae.eval()

    # Calculate and print ROC AUC values
    print("Calculating ROC AUC scores on test dataset...")
    # 1. Isolation Forest
    scores_if = -iforest.decision_function(X)
    fpr_if, tpr_if, _ = roc_curve(y, scores_if)
    auc_if = auc(fpr_if, tpr_if)
    
    # 2. LOF
    scores_lof = -lof.decision_function(X)
    fpr_lof, tpr_lof, _ = roc_curve(y, scores_lof)
    auc_lof = auc(fpr_lof, tpr_lof)
    
    # 3. One-Class SVM
    scores_svm = -oc_svm.decision_function(X)
    fpr_svm, tpr_svm, _ = roc_curve(y, scores_svm)
    auc_svm = auc(fpr_svm, tpr_svm)
    
    # 4. Autoencoder
    with torch.no_grad():
        X_tensor = torch.tensor(X.values.astype(np.float32))
        reconstructed = ae(X_tensor)
        mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
    fpr_ae, tpr_ae, _ = roc_curve(y, mse)
    auc_ae = auc(fpr_ae, tpr_ae)
    
    print(f"ROC-AUC Scores: LOF={auc_lof:.4f}, Autoencoder={auc_ae:.4f}, One-Class SVM={auc_svm:.4f}, Isolation Forest={auc_if:.4f}")

    # Sample 1000 instances for permutation importance
    print("Running permutation importance on LOF model with custom scorer...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=1000, replace=False)
    X_sample = X.iloc[sample_indices]
    y_sample = y.iloc[sample_indices]
    
    perm_importance = permutation_importance(
        lof, 
        X_sample, 
        y_sample, 
        scoring=lof_f1_scorer, 
        n_repeats=5, 
        random_state=42,
        n_jobs=-1
    )
    
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    top_n = 15
    top_features = X.columns[sorted_idx[:top_n]]
    top_importances = perm_importance.importances_mean[sorted_idx[:top_n]]
    
    plt.figure(figsize=(9, 6.5))
    sns.barplot(
        x=top_importances, 
        y=top_features, 
        palette="viridis",
        hue=top_features,
        legend=False
    )
    plt.title("Top 15 Most Important Features (LOF Permutation Anomaly Detection)", fontweight='bold', pad=15)
    plt.xlabel("Permutation Importance (F1-score Drop)", labelpad=10)
    plt.ylabel("Network Flow Feature", labelpad=10)
    plt.tight_layout()
    plt.savefig(reports_dir / "feature_importance.png", dpi=150)
    plt.close()
    print("Saved feature_importance.png")
    
    # SHAP Explanations
    print("Running SHAP explainability on LOF...")
    normal_indices = np.where(y == 0)[0]
    np.random.seed(42)
    bg_indices = np.random.choice(normal_indices, size=50, replace=False)
    bg_data = X.iloc[bg_indices]
    
    mixed_indices = np.random.choice(len(X), size=100, replace=False)
    eval_data = X.iloc[mixed_indices]
    
    explainer = shap.KernelExplainer(lof.decision_function, bg_data)
    shap_values = explainer.shap_values(eval_data, nsamples=100)
    
    plt.figure(figsize=(10, 7.5))
    shap.summary_plot(shap_values, eval_data, show=False)
    plt.title("SHAP Feature Attribution Summary (LOF Anomaly Scores)", fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_summary_plot.png", dpi=150)
    plt.close()
    print("Saved shap_summary_plot.png")
    print("All final performance artifacts generated successfully!")

if __name__ == "__main__":
    main()
