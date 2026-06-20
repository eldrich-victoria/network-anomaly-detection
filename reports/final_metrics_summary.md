# Final Metrics Summary: Model Performance Analysis

This document provides a comparative analysis of the performance metrics achieved by the four trained anomaly detection models on the UNSW-NB15 testing set (82,332 records).

---

## 1. Performance Metrics Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | Detection Characteristic |
| --- | --- | --- | --- | --- | --- |
| **Local Outlier Factor (LOF)** | 80.64% | 92.12% | 70.91% | 80.14% | Balanced, high precision with strong recall |
| **PyTorch Autoencoder** | 69.26% | 86.44% | 52.39% | 65.24% | Robust representation, low false alerts |
| **One-Class SVM** | 63.17% | 90.29% | 37.11% | 52.60% | Highly conservative, high precision / low recall |
| **Isolation Forest** | 36.88% | 42.20% | 39.60% | 40.86% | Poor performance under baseline settings |

---

## 2. Key Findings & Analysis

### A. Local Outlier Factor (LOF) - Best Performer
* **Accuracy: 80.64% | F1-Score: 80.14%**
* LOF achieves the highest detection rate among all models. With a precision of **92.12%**, it ensures that the vast majority of raised anomalies are indeed actual network threats (only 7.88% false positives). Its recall of **70.91%** means it successfully identifies 7 out of 10 network anomalies.

### B. PyTorch Autoencoder - Strong Reconstruction Baseline
* **Accuracy: 69.26% | F1-Score: 65.24%**
* The neural network autoencoder generalizes well to clean traffic patterns. By reconstruct-reconstruction MSE thresholding, it flags anomalies with **86.44% precision** and **52.39% recall**. This shows it can learn complex multi-dimensional network states, though it requires further training epochs or structure tuning to capture subtle anomaly signatures.

### C. One-Class SVM - Highly Conservative Boundary
* **Accuracy: 63.17% | F1-Score: 52.60%**
* The One-Class SVM displays very high precision (**90.29%**) but low recall (**37.11%**). It constructs a tight hypersphere containing only the densest regions of normal data, causing it to miss outliers that are close to normal patterns. This model is best suited as a secondary line of defense due to its high alert validity.

### D. Isolation Forest - High Contamination Mismatch
* **Accuracy: 36.88% | F1-Score: 40.86%**
* The baseline Isolation Forest performed poorly on the test set. Because UNSW-NB15 has high attack volume (55%) in test traffic compared to standard contamination assumptions (10%), the tree split thresholds failed to segment anomalies correctly. This highlights the importance of model threshold calibration.

---

## 3. Evaluation Visualizations

* **Confusion Matrix Panel** (`reports/metrics/confusion_matrix.png`): Shows the distribution of True Normals, False Anomalies (False Positives), False Normals (False Negatives), and True Anomalies (True Positives) for all models. LOF and Autoencoder show the highest concentrations of True Anomalies and True Normals.
* **ROC Curves Panel** (`reports/metrics/roc_curve.png`): Overlays the True Positive Rate against False Positive Rate. LOF has the largest Area Under the Curve (AUC), demonstrating superior discriminative power across all decision thresholds.
