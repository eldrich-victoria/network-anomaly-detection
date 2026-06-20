# Performance Testing & Validation Report: ShieldNet AI Anomaly Detection

This report documents the performance testing phase and validation results of the **ShieldNet AI** network anomaly detection system, evaluated on the UNSW-NB15 benchmark dataset.

---

## 1. Dataset & Sample Counts

We used the industry-standard **UNSW-NB15** dataset, generated in the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS). 

* **Total Records in Benchmark**: 257,673
* **Training Set (Clean Traffic only)**: **175,341** rows
  * Subsampled to fit distance/density models efficiently.
* **Testing Set (Mixed Traffic)**: **82,332** rows
* **Feature Dimensionality**: 45 raw columns, preprocessed to **42** active numerical/categorical inputs after stripping identifiers (`id`, `attack_cat`).
* **Test Class Distribution**:
  * **Normal (Clean) Traffic**: 37,000 flows (44.94%)
  * **Anomaly (Attack) Traffic**: 45,332 flows (55.06%)
  * **Attack Types Evaluated**: Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms.

---

## 2. Evaluation Metrics

To validate the models, we computed five standard classification metrics on out-of-sample test data:
1. **Accuracy**: Overall proportion of correctly identified network records.
2. **Precision**: The proportion of predicted anomalies that were actual attacks (critical for minimizing false alarms).
3. **Recall (Sensitivity)**: The proportion of actual network attacks successfully identified by the system (critical for minimizing security breaches).
4. **F1-Score**: The harmonic mean of Precision and Recall, providing a single balanced metric for model comparison.
5. **ROC-AUC**: The Area Under the Receiver Operating Characteristic Curve, indicating the model's discriminative power across all decision thresholds.

---

## 3. Comparative Test Results

Below is the performance summary of the four trained anomaly detection models on the test set:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Overall Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Local Outlier Factor (LOF)** | **80.64%** | **92.12%** | **70.91%** | **80.14%** | **0.8898** | **Rank #1** |
| **PyTorch Autoencoder** | 69.26% | 86.44% | 52.39% | 65.24% | 0.8224 | **Rank #2** |
| **One-Class SVM** | 63.17% | 90.29% | 37.11% | 52.60% | 0.6539 | **Rank #3** |
| **Isolation Forest** | 36.88% | 42.20% | 39.60% | 40.86% | 0.3596 | **Rank #4** |

---

## 4. Best-Performing Model Analysis

**Local Outlier Factor (LOF)** is selected as the optimal model for production deployment.
* **Why it excelled**: In high-dimensional network flow data, traffic flows form multiple clusters of varying densities depending on protocols and applications. LOF measures local density deviations relative to nearest neighbors, which allows it to construct flexible local boundaries, rather than a rigid global boundary.
* **Operational Strengths**: LOF achieved a precision of **92.12%**, reducing false positive counts to just 2,750 out of 37,000 normal flows. This significantly decreases alert fatigue for security analysts. It also achieved the highest recall (**70.91%**), successfully intercepting 32,146 attacks.
* **Deployment Context**: To run LOF at scale, we use its out-of-sample prediction mode (`novelty=True`) configured with $k=20$.

---

## 5. System Robustness & Safety Checks

During performance testing, the pipeline was validated against common failure points:
* **Unseen Values Safety**: The custom `RobustLabelEncoder` successfully handled nominal values (new protocols or service ports) in test data by placing them in an `'unknown'` bin, preventing runtime crashes.
* **Null Value Safety**: Imputation logic successfully filled missing numerical values with median bounds and nominals with `'unknown'` strings.
* **Memory & Latency Optimization**: Subsampling during the training phase of One-Class SVM and LOF prevented memory allocation errors and kept execution speed high.

---

## 6. Conclusion & Recommendations

1. **Deploy LOF as Primary Engine**: The Local Outlier Factor model should be configured as the primary real-time anomaly detection engine in the ShieldNet AI dashboard.
2. **Employ PyTorch Autoencoder as a Parallel Monitor**: The Autoencoder (AUC: **0.8224**) is recommended as a secondary detector. Its neural network reconstruction MSE is highly effective at identifying complex, distributed patterns of anomalies.
3. **Establish Dynamic Thresholding**: Anomaly severities should continue to be mapped dynamically (Low, Medium, High, Critical) based on score deviations from decision boundaries, ensuring optimal risk-based alert triaging.
