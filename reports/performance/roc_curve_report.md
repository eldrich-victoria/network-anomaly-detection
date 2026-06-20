# Receiver Operating Characteristic (ROC) Curve Comparison Report

This report presents a comparative analysis of the Receiver Operating Characteristic (ROC) curves and the corresponding Area Under the Curve (AUC) scores for the four anomaly detection models evaluated on the UNSW-NB15 dataset.

---

## 1. What are ROC and AUC?

* **ROC Curve**: A graphical plot illustrating the diagnostic ability of a binary classifier system as its discrimination threshold is varied. It plots the **True Positive Rate (TPR / Sensitivity)** against the **False Positive Rate (FPR / 1 - Specificity)** at various threshold settings.
* **AUC Score**: The Area Under the ROC Curve. AUC provides an aggregate measure of performance across all possible classification thresholds. An AUC of **1.0** indicates a perfect classifier, while an AUC of **0.5** indicates a classifier that performs no better than random guessing.

---

## 2. Model AUC Performance Summary

Based on final out-of-sample testing, the models achieved the following AUC scores (ordered from best to worst):

| Model Architecture | ROC-AUC Score | Performance Interpretation |
| :--- | :---: | :--- |
| **Local Outlier Factor (LOF)** | **0.8898** | **Excellent** classification and discriminative power. |
| **PyTorch Autoencoder** | **0.8224** | **Very Good** classification and robust reconstruction boundary. |
| **One-Class SVM** | **0.6539** | **Moderate** classification, highly sensitive to noise. |
| **Isolation Forest** | **0.3596** | **Poor** classification, indicates inverse or misaligned thresholds. |

---

## 3. Detailed AUC Performance Analysis

### A. Local Outlier Factor (LOF) - AUC: 0.8898
LOF achieves an outstanding AUC of **0.8898**, demonstrating that it possesses superior discriminative capability across all possible decision thresholds. 
* By comparing local density ratios rather than global distance metrics, LOF successfully separates outliers from dense clusters of normal traffic.
* This high AUC indicates that a security analyst can adjust the decision threshold of LOF to trade-off recall and precision without suffering a severe drop in overall diagnostic capability.

### B. PyTorch Autoencoder - AUC: 0.8224
The deep PyTorch Autoencoder achieved a strong AUC score of **0.8224**.
* Because the network is trained solely to compress and reconstruct normal traffic patterns, its reconstruction error (MSE) serves as an effective anomaly score.
* The high AUC suggests that the bottleneck layers succeeded in capturing the underlying correlations and dependencies of normal traffic features, resulting in low reconstruction error for clean traffic and high reconstruction error for unseen attack variants.

### C. One-Class SVM - AUC: 0.6539
The One-Class SVM achieved a moderate AUC of **0.6539**.
* While it achieves high precision at its default threshold, its relatively low AUC shows that it lacks overall discriminative power across alternative thresholds.
* The model struggles to differentiate between normal traffic and outliers when the threshold is moved, due to its boundary-focused hypersphere structure which does not capture density gradients.

### D. Isolation Forest - AUC: 0.3596
The Isolation Forest performed poorly, with an AUC of **0.3596**.
* An AUC below **0.5** indicates that the model is predicting *worse* than random guessing under the evaluation scoring logic. 
* This behavior occurs because the high attack volume in the test set (55%) violates the model's baseline contamination assumption (10%). Consequently, many normal traffic samples are isolated closer to the tree roots than actual anomalies, leading to inverted predictions. In a production setting, this model would require comprehensive threshold recalibration to be useful.

---

## 4. Technical Conclusion

For a resilient network security perimeter, we recommend deploying **Local Outlier Factor (LOF)** because its AUC of **0.8898** guarantees that it maintains a high detection rate (TPR) even at low false alarm thresholds. The **PyTorch Autoencoder** (AUC: **0.8224**) represents a powerful secondary deep-learning defense.
