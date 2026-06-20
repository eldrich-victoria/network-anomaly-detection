# ShieldNet AI - Network Anomaly Detection System User Guide

Welcome to the **ShieldNet AI Network Anomaly Detection System** User Guide. This system leverages state-of-the-art machine learning models (Isolation Forest, Local Outlier Factor, One-Class SVM, and PyTorch Deep Autoencoders) on the UNSW-NB15 dataset to identify and respond to network anomalies.

---

## 💻 1. Installation and Setup

### Prerequisites
- Python 3.10 to 3.14.5
- Windows, macOS, or Linux
- VS Code (optional, recommended)

### Environment Installation Steps

1. **Clone or Open the Project Workspace**:
   Navigate to the workspace folder:
   ```bash
   cd d:\network-anomaly-detection\cyber_project
   ```

2. **Create a Python Virtual Environment**:
   It is highly recommended to isolate dependencies inside a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   - **Windows PowerShell**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Windows Command Prompt**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **Linux/macOS Terminal**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies**:
   Install baseline scientific packages and the PyTorch engine:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 2. Pipeline Execution Instructions

The project is fully modular and follows a pipeline architecture. You can run individual milestones from your terminal:

### Step 1: Data Loader & Inspection (Milestone 1)
Loads raw dataset files, performs a dimensions check, checks for duplicates, and writes the summary file `reports/data_overview.txt`.
```bash
python src/preprocessing/data_loader.py
```

### Step 2: Data Preprocessing & Alignment (Milestone 2)
Cleans features, fills missing values, encodes nominal categoricals, fits standard scalers, and saves cleaned datasets to `data/processed/` along with serialized transformation state to `models/`.
```bash
python src/preprocessing/preprocessing.py
```

### Step 3: Exploratory Data Analysis Plots (Milestone 3)
Generates distribution, skewness, correlation heatmaps, boxplots, histograms, and scatter plots, saving them as high-quality PNGs in `reports/figures/`.
```bash
python src/visualization/eda.py
```

### Step 4: Model Training (Milestone 4)
Trains all four anomaly detection models sequentially (making optimization choices like subsampling One-Class SVM/LOF to ensure performance and avoid OOM crashes).
- **Isolation Forest**: `python src/models/isolation_forest_model.py`
- **Local Outlier Factor**: `python src/models/lof_model.py`
- **One-Class SVM**: `python src/models/one_class_svm_model.py`
- **Deep Autoencoder**: `python src/models/autoencoder_model.py`

### Step 5: Model Evaluation (Milestone 5)
Loads trained models, scores the test dataset, computes standard classification metrics, prints reports, and writes output comparative files to `reports/metrics/`.
```bash
python src/evaluation/evaluate_models.py
```

---

## 🖥️ 3. Streamlit Dashboard Guide

To launch the interactive dashboard, execute the following command in your terminal with the virtual environment activated:
```bash
streamlit run app.py
```
This will spin up a local server and automatically launch the application in your default web browser (typically at `http://localhost:8501`).

### Dashboard Navigation Sections

1. **🖥️ Dashboard Overview**:
   Presents high-level metrics (total flows, logged anomalies, active models) and a data grid preview of the UNSW-NB15 training dataset.

2. **📊 Exploratory Data Analysis**:
   Provides interactive Plotly charts (pie splits, attack distributions, scatter flow metrics, protocol splits) and a selector to view pre-rendered static matplotlib charts.

3. **🎯 Model Performance**:
   Compares metrics (Accuracy, Precision, Recall, F1) using clean graphical bar charts, and displays saved confusion matrices and ROC curves.

4. **🛡️ Real-Time Prediction**:
   The prediction engine. Allows network admins to select a model (e.g. PyTorch Autoencoder), upload a CSV of raw flows, and run inference. Displays summary cards, prediction tables, and provides a download button for the predicted report.

5. **🚨 Security Alerts**:
   Displays the logged incidents from the root `alerts.csv` with a threat severity status badge, total incidents counter, and options to clear historical entries.

6. **📖 About Project**:
   Explains the project background, details of the UNSW-NB15 dataset, and mathematical descriptions of the models.

---

## 🚨 4. Threat Severity Classification & Incident Response

### Severity Threshold Mapping
Anomalies are scored based on deviation distances from normal traffic boundaries:
- **Low**: Traffic is classified as normal (within normal reconstruction boundaries).
- **Medium**: Flow is anomalous but close to the normal cluster (reconstruction error is within 1 standard deviation of normal range).
- **High**: Flow is highly anomalous (between 1 and 3 standard deviations of normal range).
- **Critical**: Extremely aberrant traffic signature, indicating a high-probability zero-day exploit or active intrusion payload (above 3 standard deviations).

### Recommended SOC Action Checklist
1. **Low Alerts**: Informational. No immediate action required.
2. **Medium Alerts**: Investigate connection rate (`rate`) and service characteristics. Log details in routine security files.
3. **High Alerts**: Potential threat. Identify source IP address and destination port, and run targeted diagnostics.
4. **Critical Alerts**: Active Threat. Isolate source network components immediately, block traffic on the affected firewall ports, and initiate incident logging.
