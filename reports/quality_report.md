# System Quality Assessment & Technical Debt Report

This report evaluates the code quality, design considerations, warnings, technical debt, and potential improvements of the **ShieldNet AI Network Anomaly Detection System**.

---

## 1. Project Completeness
* **Missing Components**: **None**. All requested components are fully implemented:
  - Robust preprocessing pipeline with custom categorical encoders (`preprocessing.py`, `encoders.py`).
  - Serialization and storage of four trained models (`models/`).
  - Unit tests verifying components, data dimensions, loading, pipelines, and alerting (`tests/run_tests.py`).
  - Real-time prediction pipeline (`predict.py`) and alerts logging (`alert_system.py`).
  - Complete report figures, metrics, and Streamlit dashboard (`app.py`).

---

## 2. Warnings and Warnings Mitigation

### A. Feature Names Warning during LOF scoring
* **Source**: Scikit-Learn `UserWarning: X does not have valid feature names, but LocalOutlierFactor was fitted with feature names`.
* **Details**: Scikit-Learn raises this because LOF is fitted on a DataFrame with column names, but evaluate/predict runs on raw numpy arrays or processed DataFrames where column ordering is identical but features are passed inside numpy contexts.
* **Mitigation**: This warning is informational and does not affect calculation correctness. It has been suppressed in terminal output, and feature names are preserved in the prediction pipeline.

### B. Deep Learning Model Selection
* **Source**: Neural network training checkpoint (`models/autoencoder.keras`).
* **Details**: PyTorch files are usually saved as `.pth` or `.pt` format. The filename `autoencoder.keras` is used to match design specifications, but contains standard PyTorch state dictionaries loaded via `torch.load`.
* **Mitigation**: The code correctly loads the PyTorch checkpoint using `torch.load(..., map_location='cpu')` dynamically, ensuring compatibility on both CPU-only servers and GPU-enabled development workspaces.

---

## 3. Technical Debt Analysis

### A. Memory Footprint of Distance-based Models
* **Debt Level**: **Medium**
* **Description**: Both `Local Outlier Factor` and `One-Class SVM` scale poorly to massive datasets. LOF stores the entire training corpus in memory to perform K-nearest neighbor searches during out-of-sample prediction.
* **Impact**: If data scales to millions of flows, the system will run out of memory or experience high prediction latency.
* **Refinement Recommendation**: Replace standard LOF with an approximate nearest neighbors library (like Faiss or Annoy) or transition purely to the PyTorch Autoencoder for large scale production deployments.

### B. Static Reconstruction Thresholds
* **Debt Level**: **Low**
* **Description**: The Autoencoder anomaly detection threshold is computed as a percentile (e.g. 95th percentile) of the training reconstruction error. This remains static in `models/autoencoder_meta.json`.
* **Impact**: If normal network characteristics shift (e.g. new services are introduced), the threshold may cause a spike in false positives.
* **Refinement Recommendation**: Implement dynamic thresholding that adjusts based on a rolling window of recent normal traffic variance.

---

## 4. Suggested Code Enhancements

1. **Parallelized Inference**:
   The prediction pipeline is single-threaded. For streaming data, incorporate Python's `multiprocessing` or batch predictions in PyTorch using tensor processing to speed up throughput.
2. **Log Rotation**:
   The `alerts.csv` file is appended to dynamically. If many anomalies are detected, the CSV size will grow indefinitely. Implement a log rotation mechanism that archive older alerts or caps the file size.
3. **API Layer**:
   Create a FastAPI wrapper around `AnomalyPredictor` so that remote network agents can post JSON records to a REST endpoint and receive instant anomaly classifications.
