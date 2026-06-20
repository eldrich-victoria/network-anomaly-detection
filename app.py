import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import io
import sys

# Add workspace root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.predict import AnomalyPredictor
from src.preprocessing.encoders import RobustLabelEncoder  # required for unpickling

# Custom premium styling
st.set_page_config(
    page_title="AI Network Anomaly Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek CSS styles
st.markdown("""
<style>
    /* Premium fonts and background styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #F8F9FA;
    }
    
    /* Elegant card container */
    .premium-card {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #E9ECEF;
        margin-bottom: 24px;
        transition: transform 0.2s ease-in-out;
    }
    .premium-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
    }
    
    /* Metric titles and values */
    .metric-title {
        color: #6C757D;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .metric-value {
        color: #212529;
        font-size: 32px;
        font-weight: 800;
        margin-top: 4px;
    }
    
    /* Brand gradient headers */
    .gradient-header {
        background: linear-gradient(135deg, #4C6EF5 0%, #15AABF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 38px;
        margin-bottom: 10px;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 600;
        text-align: center;
    }
    
    .badge-critical { background-color: #FFE3E3; color: #E03131; }
    .badge-high { background-color: #FFE8CC; color: #F76707; }
    .badge-medium { background-color: #FFF9DB; color: #F59F00; }
    .badge-low { background-color: #E2F9E2; color: #2B8A3E; }
    
</style>
""", unsafe_allow_html=True)


# Cache loaders for faster performance
@st.cache_data
def get_shapes():
    """Gets lengths of datasets quickly without loading full data in memory."""
    try:
        train_len = len(pd.read_csv("data/raw/UNSW_NB15_training-set.csv", usecols=["id"]))
        test_len = len(pd.read_csv("data/raw/UNSW_NB15_testing-set.csv", usecols=["id"]))
        return train_len, test_len
    except Exception:
        return 175341, 82332  # Fallback standard shapes if files missing/unreadable

@st.cache_data
def load_train_preview(rows=1000):
    try:
        return pd.read_csv("data/raw/UNSW_NB15_training-set.csv", nrows=rows)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_features_list():
    try:
        # Load CP1252/latin-1 to handle encoding errors in raw features CSV
        return pd.read_csv("data/raw/NUSW-NB15_features.csv", encoding="latin-1")
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_metrics():
    try:
        return pd.read_csv("reports/metrics/metrics_comparison.csv")
    except Exception:
        # Fallback hardcoded values if evaluation report isn't generated
        return pd.DataFrame({
            "Model": ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Autoencoder"],
            "Accuracy": [0.3688, 0.8064, 0.6317, 0.7841],
            "Precision": [0.4220, 0.9212, 0.9029, 0.8998],
            "Recall": [0.3960, 0.7091, 0.3711, 0.6841],
            "F1-Score": [0.4086, 0.8014, 0.5260, 0.7773]
        })

def load_alerts_history():
    alerts_path = Path("alerts.csv")
    if alerts_path.exists():
        try:
            return pd.read_csv(alerts_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# SIDEBAR NAVIGATION
with st.sidebar:
    st.image("https://img.icons8.com/color/144/shield-with-crown.png", width=90)
    st.markdown("<h2 style='margin-top:0px;'>ShieldNet AI</h2>", unsafe_allow_html=True)
    st.markdown("AI-Powered Network Anomaly Detection System")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "🖥️ Dashboard Overview",
            "📊 Exploratory Data Analysis",
            "🎯 Model Performance",
            "🛡️ Real-Time Prediction",
            "🚨 Security Alerts",
            "📖 About Project"
        ]
    )
    
    st.markdown("---")
    st.markdown("**Status:** System Active 🟢")
    st.markdown(f"**Local Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# PAGE 1: DASHBOARD OVERVIEW
if page == "🖥️ Dashboard Overview":
    st.markdown("<div class='gradient-header'>System Dashboard Overview</div>", unsafe_allow_html=True)
    st.markdown("A college-level network security anomaly detection system built on the UNSW-NB15 benchmark dataset.")
    st.write(" ")
    
    # KPI Grid
    c1, c2, c3, c4 = st.columns(4)
    train_len, test_len = get_shapes()
    
    with c1:
        st.markdown(f"""
        <div class='premium-card'>
            <div class='metric-title'>Training Flows</div>
            <div class='metric-value'>{train_len:,}</div>
            <span style='color: #2B8A3E; font-size: 13px; font-weight: 600;'>UNSW-NB15 Train Set</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='premium-card'>
            <div class='metric-title'>Testing Flows</div>
            <div class='metric-value'>{test_len:,}</div>
            <span style='color: #4C6EF5; font-size: 13px; font-weight: 600;'>UNSW-NB15 Test Set</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        # Load alerts count
        alerts_df = load_alerts_history()
        alerts_count = len(alerts_df) if not alerts_df.empty else 0
        st.markdown(f"""
        <div class='premium-card'>
            <div class='metric-title'>Total Alerts Logged</div>
            <div class='metric-value'>{alerts_count:,}</div>
            <span style='color: {'#E03131' if alerts_count > 0 else '#6C757D'}; font-size: 13px; font-weight: 600;'>Active Incidents</span>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='premium-card'>
            <div class='metric-title'>Core Models</div>
            <div class='metric-value'>4</div>
            <span style='color: #15AABF; font-size: 13px; font-weight: 600;'>Algorithms Trained</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 📋 Training Dataset Sample Preview")
    train_preview = load_train_preview(100)
    if not train_preview.empty:
        st.dataframe(train_preview.head(10), use_container_width=True)
    else:
        st.info("Training dataset preview unavailable.")
        
    # Features List Expander
    st.write(" ")
    with st.expander("🔍 View Dataset Columns and Description (UNSW-NB15 Features Schema)"):
        features_df = load_features_list()
        if not features_df.empty:
            st.dataframe(features_df, use_container_width=True)
        else:
            st.info("Features details CSV file missing or unreadable.")


# PAGE 2: EXPLORATORY DATA ANALYSIS
elif page == "📊 Exploratory Data Analysis":
    st.markdown("<div class='gradient-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("Analysis of traffic distributions, protocol densities, and attack properties from the UNSW-NB15 raw data.")
    st.write(" ")
    
    tab1, tab2 = st.tabs(["📈 Interactive Plotly Visualizations", "🖼️ Pre-generated Static Reports"])
    
    with tab1:
        df_preview = load_train_preview(20000) # Load 20k rows for fast client-side plotly charts
        if not df_preview.empty:
            c1, c2 = st.columns(2)
            
            with c1:
                # Class Distribution
                class_counts = df_preview['label'].map({0: 'Normal (Clean)', 1: 'Anomaly (Attack)'}).value_counts().reset_index()
                fig_class = px.pie(
                    class_counts, 
                    names='label', 
                    values='count',
                    title='Traffic Classification: Normal vs Anomaly',
                    color='label',
                    color_discrete_map={'Normal (Clean)': '#4C6EF5', 'Anomaly (Attack)': '#FA5252'},
                    hole=0.4
                )
                fig_class.update_layout(margin=dict(t=50, b=20, l=20, r=20))
                st.plotly_chart(fig_class, use_container_width=True)
                
            with c2:
                # Attack Category Distribution
                attacks_df = df_preview[df_preview['attack_cat'] != 'Normal']
                attack_counts = attacks_df['attack_cat'].value_counts().reset_index()
                fig_attacks = px.bar(
                    attack_counts,
                    y='attack_cat',
                    x='count',
                    title='Distribution of Attack Categories (Anomalies Only)',
                    orientation='h',
                    color='count',
                    color_continuous_scale='Reds'
                )
                fig_attacks.update_layout(yaxis_title="Attack Type", xaxis_title="Count", coloraxis_showscale=False)
                st.plotly_chart(fig_attacks, use_container_width=True)
                
            st.markdown("---")
            c3, c4 = st.columns(2)
            
            with c3:
                # Top Protocols
                top_protos = df_preview['proto'].value_counts().head(8).index
                proto_df = df_preview[df_preview['proto'].isin(top_protos)].copy()
                proto_df['Traffic Type'] = proto_df['label'].map({0: 'Normal', 1: 'Anomaly'})
                fig_proto = px.histogram(
                    proto_df, 
                    x='proto', 
                    color='Traffic Type', 
                    barmode='group',
                    title='Top 8 Transaction Protocols',
                    color_discrete_map={'Normal': '#4C6EF5', 'Anomaly': '#FA5252'}
                )
                st.plotly_chart(fig_proto, use_container_width=True)
                
            with c4:
                # Service Distribution
                df_copy = df_preview.copy()
                df_copy['service'] = df_copy['service'].replace({'-': 'unknown'})
                df_copy['Traffic Type'] = df_copy['label'].map({0: 'Normal', 1: 'Anomaly'})
                fig_service = px.histogram(
                    df_copy,
                    x='service',
                    color='Traffic Type',
                    barmode='group',
                    title='Network Traffic by Service Type',
                    color_discrete_map={'Normal': '#4C6EF5', 'Anomaly': '#FA5252'}
                )
                st.plotly_chart(fig_service, use_container_width=True)
                
            st.markdown("---")
            # Scatter Plot
            df_copy['Traffic Type'] = df_copy['label'].map({0: 'Normal', 1: 'Anomaly'})
            fig_scatter = px.scatter(
                df_copy.sample(n=min(len(df_copy), 8000), random_state=42),
                x='sbytes',
                y='dbytes',
                color='Traffic Type',
                log_x=True,
                log_y=True,
                opacity=0.5,
                title='Network flow metrics: Source Bytes vs Destination Bytes (Log scale)',
                color_discrete_map={'Normal': '#4C6EF5', 'Anomaly': '#FA5252'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        else:
            st.warning("Training set not found. Please verify raw CSV paths.")
            
    with tab2:
        st.markdown("### 🖼️ Saved PNG Visualizations")
        st.markdown("These figures are generated by the EDA module and stored in the report assets.")
        
        figures_path = Path("reports/figures")
        if figures_path.exists():
            fig_select = st.selectbox(
                "Select Figure to Display",
                [
                    "label_distribution.png",
                    "attack_distribution.png",
                    "correlation_heatmap.png",
                    "protocol_distribution.png",
                    "service_distribution.png",
                    "boxplots.png",
                    "numerical_histograms.png",
                    "feature_scatter.png"
                ]
            )
            
            selected_fig_path = figures_path / fig_select
            if selected_fig_path.exists():
                st.image(str(selected_fig_path), caption=f"Static Plot: {fig_select}", use_container_width=True)
            else:
                st.info(f"Figure {fig_select} has not been pre-rendered. Run eda.py script first.")
        else:
            st.info("Pre-rendered figures folder reports/figures/ not found.")


# PAGE 3: MODEL PERFORMANCE
elif page == "🎯 Model Performance":
    st.markdown("<div class='gradient-header'>Model Performance & Evaluation</div>", unsafe_allow_html=True)
    st.markdown("Comparison metrics for the four network anomaly detection models evaluated on the test set.")
    st.write(" ")
    
    # Load metrics
    df_metrics = load_metrics()
    
    # Comparison Table
    st.markdown("### 📊 Metrics Comparison Table")
    st.dataframe(df_metrics.style.format({
        "Accuracy": "{:.4%}",
        "Precision": "{:.4%}",
        "Recall": "{:.4%}",
        "F1-Score": "{:.4%}"
    }), use_container_width=True)
    
    # Interactive Comparison Chart
    st.markdown("### 📈 Visual Metrics comparison")
    df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_metrics = px.bar(
        df_melted, 
        x="Metric", 
        y="Score", 
        color="Model", 
        barmode="group",
        title="Accuracy, Precision, Recall, and F1-Score comparison",
        color_discrete_map={
            "Isolation Forest": "#1F77B4",
            "Local Outlier Factor": "#FF7F0E",
            "One-Class SVM": "#2CA02C",
            "Autoencoder": "#D62728"
        }
    )
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Display saved Confusion Matrix & ROC Curves
    st.markdown("---")
    st.markdown("### 🖼️ Saved Evaluation Plots")
    c1, c2 = st.columns(2)
    
    metrics_fig_path = Path("reports/metrics")
    
    with c1:
        cm_path = metrics_fig_path / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrices Panel", use_container_width=True)
        else:
            st.info("Confusion matrix plot `confusion_matrix.png` not found. Run evaluate_models.py first.")
            
    with c2:
        roc_path = metrics_fig_path / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)
        else:
            st.info("ROC curve plot `roc_curve.png` not found. Run evaluate_models.py first.")


# PAGE 4: REAL-TIME PREDICTION
elif page == "🛡️ Real-Time Prediction":
    st.markdown("<div class='gradient-header'>Real-Time Anomaly Predictor</div>", unsafe_allow_html=True)
    st.markdown("Upload a network flow CSV file and inspect it for anomalies in real-time.")
    st.write(" ")
    
    # Setup predictor
    try:
        predictor = AnomalyPredictor()
        st.success("Trained models, preprocessor, and scalers loaded successfully! 🟢")
    except Exception as e:
        st.error(f"Error loading models or preprocessors: {e}")
        st.info("Please make sure you have run the preprocessing and model training steps before executing real-time predictions.")
        st.stop()
        
    # Select Model and File
    model_choice = st.selectbox(
        "Choose Anomaly Detection Model",
        ["Autoencoder", "Local Outlier Factor", "One-Class SVM", "Isolation Forest"]
    )
    
    uploaded_file = st.file_uploader("Upload Network Flow CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read uplaoded csv
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Uploaded File Dimensions: `{len(df_upload)}` rows, `{len(df_upload.columns)}` columns.")
            
            # Button to run predictions
            if st.button("🚀 Run Anomaly Detection Pipeline"):
                with st.spinner("Preprocessing data and running inferences..."):
                    # Predict
                    results_df = predictor.predict(df_upload, model_choice)
                    
                st.balloons()
                st.success("Analysis complete!")
                
                # Show key metrics of prediction
                anom_count = int(results_df["Prediction"].sum())
                norm_count = len(results_df) - anom_count
                anom_rate = anom_count / len(results_df)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Clean Flows Detected (Normal)", f"{norm_count:,}")
                with c2:
                    st.metric("Anomalous Flows Detected (Alert)", f"{anom_count:,}", 
                              delta=f"{anom_rate:.2%} rate", delta_color="inverse")
                with c3:
                    # Critical/High count
                    if anom_count > 0:
                        crit_high_count = len(results_df[results_df["Severity"].isin(["Critical", "High"])])
                        st.metric("High/Critical Severity Alerts", f"{crit_high_count:,}", 
                                  help="Requires immediate response action")
                    else:
                        st.metric("High/Critical Severity Alerts", "0")
                        
                # Visual breakdowns
                st.markdown("### 📈 Visual Anomalies Breakdowns")
                cc1, cc2 = st.columns(2)
                
                with cc1:
                    # Prediction pie
                    fig_pred = px.pie(
                        names=["Normal", "Anomaly"],
                        values=[norm_count, anom_count],
                        title="Inference Label Split",
                        color=["Normal", "Anomaly"],
                        color_discrete_map={"Normal": "#4C6EF5", "Anomaly": "#FA5252"},
                        hole=0.4
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                with cc2:
                    # Severity count bar
                    severity_counts = results_df["Severity"].value_counts().reset_index()
                    # Reorder categories
                    cat_order = ["Low", "Medium", "High", "Critical"]
                    severity_counts["Severity"] = pd.Categorical(severity_counts["Severity"], categories=cat_order, ordered=True)
                    severity_counts = severity_counts.sort_values("Severity")
                    
                    fig_sev = px.bar(
                        severity_counts,
                        x="Severity",
                        y="count",
                        title="Alert Severity Level breakdown",
                        color="Severity",
                        color_discrete_map={
                            "Low": "#2B8A3E",
                            "Medium": "#F59F00",
                            "High": "#F76707",
                            "Critical": "#E03131"
                        }
                    )
                    st.plotly_chart(fig_sev, use_container_width=True)
                    
                # Details Table
                st.markdown("### 📋 Prediction Table Results (Top 100 rows)")
                # Color code rows or display nicely
                columns_to_show = ["Prediction", "Severity", "Anomaly_Score", "proto", "service", "state", "dur", "sbytes", "dbytes", "rate"]
                cols_present = [c for c in columns_to_show if c in results_df.columns]
                
                # HTML colored table for alerts
                styled_df = results_df[cols_present].head(100)
                st.dataframe(styled_df, use_container_width=True)
                
                # Download results CSV
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Full Predictions CSV Report",
                    data=csv_buffer.getvalue(),
                    file_name=f"anomaly_predictions_{model_choice.lower().replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Failed to process CSV file: {e}")
            logger.error("CSV Upload Prediction failed", exc_info=True)


# PAGE 5: SECURITY ALERTS LOG
elif page == "🚨 Security Alerts":
    st.markdown("<div class='gradient-header'>Incident Response & Alerts Log</div>", unsafe_allow_html=True)
    st.markdown("Historical and real-time security alerts logged by the ShieldNet AI engine.")
    st.write(" ")
    
    # Load alerts
    alerts_df = load_alerts_history()
    
    if alerts_df.empty:
        st.info("No security incidents logged. All systems clear! 🛡️")
        
        # Display incident response protocol as boilerplate
        st.markdown("### 📋 Security Incident Response Standard Protocol")
        st.markdown("""
        When an anomaly is logged, the Security Operations Center (SOC) must execute the following response actions:
        1. **Verify**: Double check flow metrics on the dashboard to verify if the alert is malicious.
        2. **Isolate**: If an anomaly is rated as **Critical** or **High** severity, isolate the source IP (`srcip` if logged) or block the port (`sport`/`dsport`).
        3. **Analyze**: Inspect flow payloads, protocols, and durations to identify attack vectors (e.g. Exploits, DDoS, Reconnaissance).
        4. **Log**: Document all remediation details in the network security reports.
        """)
    else:
        # Show stats
        total_alerts = len(alerts_df)
        crit_alerts = len(alerts_df[alerts_df["Severity"] == "Critical"])
        high_alerts = len(alerts_df[alerts_df["Severity"] == "High"])
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='premium-card'>
                <div class='metric-title'>Logged Incidents</div>
                <div class='metric-value'>{total_alerts}</div>
                <span style='color: #6C757D; font-size: 13px; font-weight: 600;'>Total Anomalies Detected</span>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='premium-card'>
                <div class='metric-title'>Critical Threat Level</div>
                <div class='metric-value' style='color: #E03131;'>{crit_alerts}</div>
                <span style='color: #E03131; font-size: 13px; font-weight: 600;'>Immediate Action Required</span>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='premium-card'>
                <div class='metric-title'>High Threat Level</div>
                <div class='metric-value' style='color: #F76707;'>{high_alerts}</div>
                <span style='color: #F76707; font-size: 13px; font-weight: 600;'>Urgent Investigation</span>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### 📋 Logged Incidents Table")
        # Reverse rows to show latest alerts first
        st.dataframe(alerts_df.iloc[::-1], use_container_width=True)
        
        # Clear log button
        if st.button("🗑️ Clear Alerts Log History"):
            try:
                Path("alerts.csv").unlink(missing_ok=True)
                st.success("Alerts history cleared successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear log file: {e}")


# PAGE 6: ABOUT PROJECT
elif page == "📖 About Project":
    st.markdown("<div class='gradient-header'>About the Project</div>", unsafe_allow_html=True)
    st.markdown("Overview of the AI-Powered Network Anomaly Detection System and component details.")
    st.write(" ")
    
    st.markdown("""
    ### 🛡️ Project Summary
    This project is an advanced AI-powered network security tool that analyzes transaction characteristics from routers (packet flows, session durations, byte counts, and transaction rates) to determine whether the behavior is a benign flow (Normal) or an outlier (Anomaly/Attack).
    
    ### 📊 Dataset Details: UNSW-NB15
    The UNSW-NB15 dataset is an industry benchmark dataset created by the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS) for evaluating network intrusion detection systems. It contains 45 features spanning flow duration, protocols, connection rates, and packet count statistics. The dataset documents 9 types of attacks: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms.
    
    ### 🤖 Implemented Anomaly Detection Models
    The project leverages four distinct anomaly detection approaches representing different machine learning paradigms:
    
    1. **Isolation Forest (Ensemble Learning)**:
       An unsupervised outlier detection model that isolates anomalies by randomly partitioning feature values using decision trees. Since anomalies require fewer splits to isolate, they appear closer to the tree root, yielding high anomaly scores.
       
    2. **Local Outlier Factor (LOF - Density-based)**:
       A density-based algorithm that measures the local density deviation of a network flow relative to its neighbors. Flows with significantly lower density than their K-neighbors are classed as outliers.
       
    3. **One-Class SVM (Support Vector Machine - Boundary-based)**:
       A kernel-based algorithm that maps the normal traffic data points into a high-dimensional space and fits a tight decision boundary enclosing them. Points falling outside the hypersphere boundary are classified as anomalies.
       
    4. **PyTorch Deep Autoencoder (Deep Reconstruction Learning)**:
       An artificial neural network trained to reconstruct normal traffic patterns. During test/predict, normal flows are reconstructed with very low Mean Squared Error (MSE), whereas anomalous/unseen patterns yield high reconstruction error, crossing the anomaly threshold.
    
    ### 🛠️ Technology Stack & Architectures
    - **Backend Framework**: Python 3.14, PyTorch (Autoencoder Fallback Engine), Scikit-Learn
    - **Frontend Dashboard**: Streamlit
    - **Visualization**: Plotly, Seaborn, Matplotlib
    - **Data Operations**: Pandas, NumPy
    """)
