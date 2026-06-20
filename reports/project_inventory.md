# Project Inventory: ShieldNet AI Network Anomaly Detection System

This inventory lists all folder structures, source code modules, preprocessed data files, serialized model checkpoints, evaluations, figures, and UI screenshots generated for the project.

---

## 1. Directory Tree Structure

```text
cyber_project/
├── .venv/                          # Local Python virtual environment
├── app.py                          # Streamlit interactive dashboard web app
├── alerts.csv                      # Historical and real-time security alerts log
├── sample_test.csv                 # Clean/Attack mixed 10-row validation flow file
├── requirements.txt                # Python package dependency list
├── README.md                       # High-level overview and execution guide
├── data/                           # Data storage
│   ├── raw/                        # Original benchmark files
│   │   ├── NUSW-NB15_features.csv
│   │   ├── UNSW_NB15_testing-set.csv
│   │   └── UNSW_NB15_training-set.csv
│   └── processed/                  # Transformed/Cleaned files
│       ├── test_processed.csv
│       └── train_processed.csv
├── models/                         # Serialized ML files
│   ├── autoencoder.keras           # PyTorch deep autoencoder state dictionary
│   ├── autoencoder_meta.json       # Autoencoder threshold configuration
│   ├── encoders.pkl                # Custom label encoders
│   ├── isolation_forest.pkl        # Serialized Isolation Forest estimator
│   ├── lof.pkl                     # Serialized Local Outlier Factor estimator
│   ├── one_class_svm.pkl           # Serialized One-Class SVM estimator
│   ├── preprocessing_config.pkl    # Core categorical/numerical list mapping
│   └── scaler.pkl                  # Serialized StandardScaler estimator
├── reports/                        # Documentation and reporting
│   ├── User_Guide.md               # User guide for running scripts
│   ├── data_overview.txt           # Text summary of features
│   ├── project_summary.md          # Early project summary
│   ├── testing_report.md           # Standard unit testing suite report
│   ├── final_results.md            # Overall execution and architecture results
│   ├── final_metrics_summary.md    # Multi-model metrics comparison
│   ├── project_inventory.md        # [This File] Inventory directory mapping
│   ├── final_conclusion.md         # Final project conclusions and scope
│   ├── quality_report.md           # Code quality assessment and future debt
│   ├── figures/                    # EDA charts
│   │   ├── attack_distribution.png
│   │   ├── boxplots.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_scatter.png
│   │   ├── label_distribution.png
│   │   ├── numerical_histograms.png
│   │   ├── protocol_distribution.png
│   │   └── service_distribution.png
│   └── metrics/                    # Model evaluation plots
│       ├── confusion_matrix.png
│       ├── metrics_comparison.csv
│       └── roc_curve.png
├── screenshots/                    # Real Streamlit Web UI Page Captures
│   ├── alerts_page.png             # Incident alerts log screen
│   ├── dashboard_homepage.png      # Homepage dashboard KPI overview screen
│   ├── eda_page.png                # EDA interactive pie/bar charts screen
│   ├── model_comparison_page.png   # Model performance metrics charts screen
│   └── prediction_page.png         # CSV upload real-time inference run screen
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── evaluation/
│   │   └── evaluate_models.py      # Batch scoring and performance assessment
│   ├── models/                     # Model definitions
│   │   ├── autoencoder_model.py
│   │   ├── isolation_forest_model.py
│   │   ├── lof_model.py
│   │   └── one_class_svm_model.py
│   ├── preprocessing/              # Transformations
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── encoders.py
│   │   └── preprocessing.py
│   ├── utils/                      # Helper pipelines
│   │   ├── alert_system.py         # Incident logging and classification
│   │   ├── predict.py              # Real-time CSV scoring pipeline
│   │   └── retrain_model.py        # Model retraining script
│   └── visualization/
│       └── eda.py                  # Exploratory data distribution analysis
└── tests/
    └── run_tests.py                # Automated system test suite runner
```

---

## 2. Inventory Metrics
* **Total Python Source Files**: 15 (`app.py`, `tests/run_tests.py`, and 13 modules in `src/`)
* **Trained Models Serialized**: 4 core estimators (`isolation_forest.pkl`, `lof.pkl`, `one_class_svm.pkl`, `autoencoder.keras`) + 2 preprocessing files (`encoders.pkl`, `scaler.pkl`).
* **Visual Diagrams Rendered**: 10 plots (8 EDA charts + 2 model evaluation charts).
* **Logged Incident Records**: Stored in `alerts.csv` containing timestamps, anomaly scores, severity ratings, and network attributes (duration, protocols, rates).
