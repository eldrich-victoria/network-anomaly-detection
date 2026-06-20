# Confusion Matrix Comparison Report

This report provides a detailed analysis of the confusion matrix results obtained by the four anomaly detection models on the UNSW-NB15 test dataset. The test set contains **82,332** network records:
* **Normal (Clean) traffic**: 37,000 flows (44.94%)
* **Anomaly (Attack) traffic**: 45,332 flows (55.06%)

---

## 1. Summary of Confusion Matrices

Below are the exact counts of True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP) obtained by each model:

| Model Architecture | True Normal (TN) | False Anomaly (FP) | False Normal (FN) | True Anomaly (TP) |
| :--- | :---: | :---: | :---: | :---: |
| **Local Outlier Factor (LOF)** | **34,250** | 2,750 | **13,186** | **32,146** |
| **PyTorch Autoencoder** | 33,276 | 3,724 | 21,583 | 23,749 |
| **One-Class SVM** | **35,191** | **1,809** | 28,511 | 16,821 |
| **Isolation Forest** | 12,411 | 24,589 | 27,379 | 17,953 |

---

## 2. In-Depth Model Breakdown

### A. Local Outlier Factor (LOF) - Balanced & High Recall
* **True Anomaly Rate (Recall)**: **70.91%** (32,146 attacks identified out of 45,332)
* **False Alarm Rate (FAR)**: **7.43%** (2,750 false alerts out of 37,000 normal flows)
* **Analysis**: LOF represents the most operationally viable model. It successfully captures 32,146 network anomalies, leaving 13,186 undetected (False Negatives). In a security environment, its low false positive rate (7.43%) prevents the Security Operations Center (SOC) from being flooded with false alarms while maintaining a robust detection perimeter.

### B. PyTorch Autoencoder - Consistent Generalization
* **True Anomaly Rate (Recall)**: **52.39%** (23,749 attacks identified out of 45,332)
* **False Alarm Rate (FAR)**: **10.06%** (3,724 false alerts out of 37,000 normal flows)
* **Analysis**: The Autoencoder shows strong ability in capturing standard normal behavior, correctly identifying 33,276 normal flows (89.94%). However, its higher False Negatives count (21,583 missed attacks) shows that some attacks reconstructed with relatively low MSE, escaping detection. Calibrating the reconstruction threshold could help trade-off precision for higher recall.

### C. One-Class SVM - Highly Conservative Defense
* **True Anomaly Rate (Recall)**: **37.11%** (16,821 attacks identified out of 45,332)
* **False Alarm Rate (FAR)**: **4.89%** (Only 1,809 false alerts out of 37,000 normal flows)
* **Analysis**: One-Class SVM achieved the lowest False Positive count (1,809), meaning it has a **95.11% True Negative Rate**. Unfortunately, it missed **28,511** network attacks (62.89% False Negative Rate). It constructs a very tight hypersphere around the dense center of normal traffic, leaving any normal traffic on the periphery to be flagged as anomalous, while failing to detect anomalies that lie along the radial paths.

### D. Isolation Forest - High Contamination Mismatch
* **True Anomaly Rate (Recall)**: **39.60%** (17,953 attacks identified out of 45,332)
* **False Alarm Rate (FAR)**: **66.46%** (24,589 false alerts out of 37,000 normal flows)
* **Analysis**: The baseline Isolation Forest performed poorly on this set. By flagging 24,589 normal records as anomalies and missing 27,379 actual attacks, the model suffers from extreme boundary mismatch. This is because the default training contamination threshold (0.1) did not match the high percentage of attacks in the dataset (55%).

---

## 3. Operational Security Implications

1. **Alert Fatigue Prevention**: One-Class SVM and LOF are highly effective at minimizing false positive rates. LOF is preferred because it balances this with a high recall rate (70.91%), whereas One-Class SVM leaves the network highly vulnerable by letting 62.89% of attacks slip through.
2. **Layered Defense Configuration**: In a production deployment, we recommend running **LOF** as the primary high-throughput detector. The **Autoencoder** can run in parallel as a secondary detector to capture complex multi-feature anomaly patterns, while the **One-Class SVM** serves to double-check high-confidence alerts.
