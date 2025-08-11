"""Streamlit demo app for Credit Card Fraud Detection."""
import os
import sys
import logging
from typing import Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
project_root = os.path.abspath(project_root)
sys.path.insert(0, project_root)

# Also add current working directory to path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from src.utils import load_object
    from src.inference import predict_from_row, predict_from_df, create_sample_transaction
    from src.evaluate import plot_confusion_matrix
    from src.config import FEATURE_COLUMNS, MODEL_DIR
    from src.explain import explain_shap, plot_feature_importance
    from src.business_impact import (
        calculate_fraud_detection_rate, calculate_cost_savings, 
        threshold_cost_analysis, plot_cost_vs_threshold, 
        plot_cost_breakdown, generate_business_report
    )
    from src.model_drift import (
        check_data_drift, generate_drift_summary, 
        format_drift_report, interpret_drift_level
    )
    from src.data_utils import load_data
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure you are running from the project root directory")
    st.error(f"Current working directory: {os.getcwd()}")
    st.error(f"Project root: {project_root}")
    st.error("**Correct command:** `streamlit run app/streamlit_app.py` from project root")
    st.error("**Alternative:** `python -m streamlit run app/streamlit_app.py`")
    
    st.code("""
# Quick fix - run these commands:
cd "credit-card-fraud"  # Navigate to project root
python -m streamlit run app/streamlit_app.py
    """)
    st.stop()


# Configure page
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.fraud-alert {
    background-color: #ffebee;
    color: #c62828;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #c62828;
}
.normal-alert {
    background-color: #e8f5e8;
    color: #2e7d32;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #2e7d32;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models and pipeline."""
    try:
        pipeline_path = os.path.join(MODEL_DIR, "pipeline.joblib")
        xgb_path = os.path.join(MODEL_DIR, "xgb_model.joblib")
        
        if not os.path.exists(pipeline_path) or not os.path.exists(xgb_path):
            return None, None, "Models not found. Please run training first."
        
        pipeline = load_object(pipeline_path)
        model = load_object(xgb_path)
        
        # Verify model is working by testing a small prediction with proper DataFrame
        test_data = {feature: 0.0 for feature in FEATURE_COLUMNS}
        test_df = pd.DataFrame([test_data])
        test_processed = pipeline.transform(test_df)
        test_prob = model.predict_proba(test_processed)[0, 1]
        
        return pipeline, model, f"Models loaded successfully! Test prediction: {test_prob:.4f}"
    except Exception as e:
        return None, None, f"Error loading models: {str(e)}"


def create_feature_input_form():
    """Create input form for single transaction features."""
    st.subheader("📝 Manual Transaction Input")
    
    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)
        
        # Time and Amount
        with col1:
            time_val = st.number_input("Time", value=50000.0, help="Time in seconds from first transaction")
            amount_val = st.number_input("Amount", value=100.0, min_value=0.0, help="Transaction amount")
        
        # V1-V10
        with col2:
            st.write("**PCA Features V1-V10:**")
            v1_10 = {}
            for i in range(1, 11):
                v1_10[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.6f", step=0.1)
        
        # V11-V20
        with col3:
            st.write("**PCA Features V11-V20:**")
            v11_20 = {}
            for i in range(11, 21):
                v11_20[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.6f", step=0.1)
        
        # V21-V28
        st.write("**PCA Features V21-V28:**")
        col4, col5 = st.columns(2)
        with col4:
            v21_25 = {}
            for i in range(21, 26):
                v21_25[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.6f", step=0.1)
        
        with col5:
            v26_28 = {}
            for i in range(26, 29):
                v26_28[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.6f", step=0.1)
        
        # Combine all values
        transaction_data = {'Time': time_val, 'Amount': amount_val}
        transaction_data.update(v1_10)
        transaction_data.update(v11_20)
        transaction_data.update(v21_25)
        transaction_data.update(v26_28)
        
        submitted = st.form_submit_button("🔍 Predict Transaction", use_container_width=True)
        
        return transaction_data if submitted else None


def display_prediction_result(result: Dict[str, Any]):
    """Display prediction result with styling."""
    prob_fraud = result['prob_fraud']
    is_fraud = result['prediction'] == 1
    
    # Main prediction display
    if is_fraud:
        st.markdown(
            f'<div class="fraud-alert"><strong>⚠️ FRAUD DETECTED</strong><br>'
            f'Fraud Probability: <strong>{prob_fraud:.1%}</strong></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="normal-alert"><strong>✅ NORMAL TRANSACTION</strong><br>'
            f'Fraud Probability: <strong>{prob_fraud:.1%}</strong></div>',
            unsafe_allow_html=True
        )
    
    # Detailed metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fraud Probability", f"{prob_fraud:.1%}")
    with col2:
        st.metric("Normal Probability", f"{result['prob_normal']:.1%}")
    with col3:
        risk_level = "HIGH" if prob_fraud > 0.7 else "MEDIUM" if prob_fraud > 0.3 else "LOW"
        st.metric("Risk Level", risk_level)


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
    
    # Load models
    pipeline, model, status_message = load_models()
    
    if pipeline is None or model is None:
        st.error(status_message)
        st.info("**To train models:**")
        st.code("python -m src.train", language="bash")
        st.info("**Expected files:**")
        st.code(f"{MODEL_DIR}/pipeline.joblib\n{MODEL_DIR}/xgb_model.joblib")
        return
    
    st.success(status_message)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    threshold = st.sidebar.slider(
        "Decision Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.01,
        help="Threshold for classifying as fraud"
    )
    
    show_explanations = st.sidebar.checkbox("Show Model Explanations", value=False)
    
    # Add cache clearing button
    if st.sidebar.button("🔄 Reload Models"):
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.info("💡 If models aren't updating, click 'Reload Models' button above.")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Single Prediction", 
        "📊 Batch Prediction", 
        "📈 Model Info", 
        "🧠 Explanations",
        "💰 Business Impact",
        "📉 Model Drift"
    ])
    
    # Tab 1: Single Transaction Prediction
    with tab1:
        st.header("Single Transaction Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Manual input form
            transaction_data = create_feature_input_form()
            
            if transaction_data:
                try:
                    result = predict_from_row(model, pipeline, transaction_data)
                    display_prediction_result(result)
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        
        with col2:
            st.subheader("🎲 Quick Test")
            st.write("Generate a random sample transaction:")
            
            if st.button("Generate Sample Transaction", use_container_width=True):
                sample_data = create_sample_transaction()
                try:
                    result = predict_from_row(model, pipeline, sample_data)
                    st.write("**Sample Transaction:**")
                    st.json({k: f"{v:.4f}" for k, v in sample_data.items() if k in ['Time', 'Amount', 'V1', 'V2', 'V3']})
                    display_prediction_result(result)
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Transaction Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with transactions", 
            type=['csv'],
            help="CSV should contain columns: Time, Amount, V1-V28"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"**Uploaded:** {len(df)} transactions")
                
                # Show data preview
                with st.expander("Data Preview"):
                    st.dataframe(df.head())
                
                if st.button("🔍 Analyze All Transactions", use_container_width=True):
                    with st.spinner("Analyzing transactions..."):
                        results_df = predict_from_df(model, pipeline, df, threshold)
                        
                        # Summary metrics
                        total_transactions = len(results_df)
                        fraud_predictions = sum(results_df['pred_label'])
                        fraud_rate = fraud_predictions / total_transactions
                        avg_fraud_prob = results_df['prob_fraud'].mean()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Transactions", total_transactions)
                        with col2:
                            st.metric("Predicted Fraud", fraud_predictions)
                        with col3:
                            st.metric("Fraud Rate", f"{fraud_rate:.1%}")
                        with col4:
                            st.metric("Avg Fraud Prob", f"{avg_fraud_prob:.1%}")
                        
                        # Probability distribution
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Histogram of fraud probabilities
                        ax1.hist(results_df['prob_fraud'], bins=50, alpha=0.7, edgecolor='black')
                        ax1.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
                        ax1.set_xlabel('Fraud Probability')
                        ax1.set_ylabel('Count')
                        ax1.set_title('Distribution of Fraud Probabilities')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Prediction counts
                        pred_counts = results_df['prediction'].value_counts()
                        ax2.bar(pred_counts.index, pred_counts.values, color=['green', 'red'], alpha=0.7)
                        ax2.set_xlabel('Prediction')
                        ax2.set_ylabel('Count')
                        ax2.set_title('Prediction Counts')
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show detailed results
                        with st.expander("Detailed Results"):
                            st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )
                        
                        # Confusion matrix if true labels available
                        if 'Class' in df.columns:
                            st.subheader("Performance Metrics")
                            from sklearn.metrics import confusion_matrix, classification_report
                            
                            cm = confusion_matrix(df['Class'], results_df['pred_label'])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                fig, ax = plt.subplots(figsize=(6, 5))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                          xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                ax.set_title('Confusion Matrix')
                                st.pyplot(fig)
                            
                            with col2:
                                report = classification_report(df['Class'], results_df['pred_label'], output_dict=True)
                                st.write("**Classification Report:**")
                                st.json(report)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Tab 3: Model Information
    with tab3:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Model Details")
            st.info(f"""
            **Model Type:** XGBoost Classifier
            **Features:** {len(FEATURE_COLUMNS)} (Time, Amount, V1-V28)
            **Training:** SMOTE oversampling + RandomizedSearchCV
            **Preprocessing:** StandardScaler
            """)
            
            st.subheader("🎯 Performance Notes")
            st.info("""
            - Model trained on imbalanced credit card data
            - SMOTE applied to balance training set
            - Hyperparameters tuned with cross-validation
            - Focus on minimizing false negatives (missed fraud)
            """)
        
        with col2:
            st.subheader("📊 Feature Information")
            st.info("""
            **Time:** Seconds elapsed between transactions
            **Amount:** Transaction amount
            **V1-V28:** PCA-transformed features (anonymized)
            
            All features are scaled using StandardScaler.
            """)
            
            st.subheader("⚡ Usage Tips")
            st.info("""
            - Adjust threshold based on business requirements
            - Lower threshold = catch more fraud (more false positives)
            - Higher threshold = fewer false alarms (may miss fraud)
            - Monitor model performance regularly
            """)
    
    # Tab 4: Model Explanations
    with tab4:
        st.header("Model Explanations")
        
        if show_explanations:
            st.info("Loading model explanations... This may take a moment.")
            
            # Check for saved SHAP plots
            shap_summary_path = os.path.join(MODEL_DIR, 'shap_summary.png')
            shap_bar_path = os.path.join(MODEL_DIR, 'shap_bar.png')
            
            if os.path.exists(shap_summary_path):
                st.subheader("🔍 SHAP Feature Importance")
                st.image(shap_summary_path, caption="SHAP Summary Plot")
                
                if os.path.exists(shap_bar_path):
                    st.image(shap_bar_path, caption="SHAP Bar Plot")
            else:
                st.warning("SHAP plots not found. Generate them by running the explanation analysis.")
                st.code("# In your training script, add:\nfrom src.explain import explain_shap\nexplain_shap(model, X_sample, feature_names)")
        else:
            st.info("Enable 'Show Model Explanations' in the sidebar to view SHAP analysis.")
        
        st.subheader("📖 How to Interpret")
        st.info("""
        **Feature Importance:**
        - Higher absolute SHAP values = more important for prediction
        - Red features push prediction toward fraud
        - Blue features push prediction toward normal
        
        **Business Insights:**
        - Amount and timing patterns are often important
        - V-features capture transaction behavior patterns
        - Model learns complex interactions between features
        """)
    
    # Tab 5: Business Impact Analysis
    with tab5:
        business_impact_tab(model, pipeline)
    
    # Tab 6: Model Drift Detection
    with tab6:
        model_drift_tab()


def business_impact_tab(model, pipeline):
    """Business Impact Analysis tab implementation."""
    st.header("💰 Business Impact Analysis")
    
    # Cost parameters
    st.subheader("⚙️ Cost Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        cost_fp = st.slider(
            "Cost per False Positive ($)", 
            min_value=0.1, max_value=10.0, value=1.0, step=0.1,
            help="Cost of blocking a legitimate transaction"
        )
    
    with col2:
        cost_fn = st.slider(
            "Cost per False Negative ($)", 
            min_value=1.0, max_value=50.0, value=5.0, step=1.0,
            help="Cost of missing a fraudulent transaction"
        )
    
    # Data source selection
    st.subheader("📊 Analysis Dataset")
    data_source = st.radio(
        "Choose data source:",
        ["Test Set (Built-in)", "Upload CSV File"],
        horizontal=True
    )
    
    try:
        if data_source == "Test Set (Built-in)":
            # Load test data
            X, y = load_data()
            from sklearn.model_selection import train_test_split
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Make predictions
            X_test_processed = pipeline.transform(X_test)
            y_pred = model.predict(X_test_processed)
            y_proba = model.predict_proba(X_test_processed)[:, 1]
            
            st.success(f"✅ Analyzed {len(X_test):,} test transactions")
            
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file with transaction data",
                type=['csv'],
                help="CSV should contain features and 'Class' column (0=normal, 1=fraud)"
            )
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                
                if 'Class' in df.columns:
                    y_test = df['Class'].values
                    X_test = df.drop('Class', axis=1)
                    
                    # Ensure proper feature columns
                    missing_cols = set(FEATURE_COLUMNS) - set(X_test.columns)
                    if missing_cols:
                        st.error(f"Missing required columns: {missing_cols}")
                        return
                    
                    X_test = X_test[FEATURE_COLUMNS]
                    
                    # Make predictions
                    X_test_processed = pipeline.transform(X_test)
                    y_pred = model.predict(X_test_processed)
                    y_proba = model.predict_proba(X_test_processed)[:, 1]
                    
                    st.success(f"✅ Analyzed {len(X_test):,} uploaded transactions")
                else:
                    st.error("CSV file must contain 'Class' column with true labels")
                    return
            else:
                st.info("Please upload a CSV file to analyze business impact")
                return
        
        # Generate business analysis
        business_report = generate_business_report(y_test, y_pred, y_proba, cost_fp, cost_fn)
        
        # Display key metrics
        st.subheader("🎯 Key Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fraud_rate = business_report['fraud_detection_rate']
            st.metric(
                "Fraud Detection Rate",
                f"{fraud_rate:.1f}%",
                help="Percentage of actual fraud cases detected"
            )
        
        with col2:
            savings_100k = business_report['cost_analysis']['savings_per_100k']
            st.metric(
                "Savings per 100K Transactions",
                f"${savings_100k:,.0f}",
                help="Cost savings compared to baseline model"
            )
        
        with col3:
            optimal_threshold = business_report['optimal_threshold']
            st.metric(
                "Optimal Threshold",
                f"{optimal_threshold:.3f}",
                help="Classification threshold that minimizes total cost"
            )
        
        with col4:
            total_savings = business_report['cost_analysis']['cost_saved']
            st.metric(
                "Total Cost Savings",
                f"${total_savings:,.0f}",
                help="Total savings on this dataset"
            )
        
        # Cost analysis visualizations
        st.subheader("📈 Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(business_report['cost_plot'], use_container_width=True)
        
        with col2:
            st.plotly_chart(business_report['breakdown_plot'], use_container_width=True)
        
        # Detailed breakdown
        st.subheader("📋 Detailed Analysis")
        
        cost_analysis = business_report['cost_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Model Performance:**")
            st.write(f"• False Positives: {cost_analysis['fp_current']:,}")
            st.write(f"• False Negatives: {cost_analysis['fn_current']:,}")
            st.write(f"• Total Cost: ${cost_analysis['current_cost']:,.0f}")
        
        with col2:
            st.write("**Baseline Model (All Negative):**")
            st.write(f"• False Positives: {cost_analysis['fp_baseline']:,}")
            st.write(f"• False Negatives: {cost_analysis['fn_baseline']:,}")
            st.write(f"• Total Cost: ${cost_analysis['baseline_cost']:,.0f}")
        
        # Threshold sensitivity analysis
        with st.expander("🔍 Threshold Sensitivity Analysis"):
            threshold_df = business_report['threshold_df']
            st.dataframe(
                threshold_df.style.format({
                    'threshold': '{:.3f}',
                    'total_cost': '${:,.0f}',
                    'precision': '{:.3f}',
                    'recall': '{:.3f}',
                    'cost_per_100k': '${:,.0f}'
                }),
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"Error in business impact analysis: {str(e)}")
        st.error("Please ensure your data format is correct and models are loaded properly.")


def model_drift_tab():
    """Model Drift Detection tab implementation."""
    st.header("📉 Model Drift Detection")
    
    st.info("""
    **Model Drift Detection** helps identify when new data significantly differs from training data,
    which could indicate that model performance may degrade and retraining might be needed.
    """)
    
    # Configuration
    st.subheader("⚙️ Drift Detection Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        psi_threshold = st.slider(
            "PSI Threshold", 
            min_value=0.1, max_value=0.5, value=0.25, step=0.05,
            help="Population Stability Index threshold (>0.25 indicates significant drift)"
        )
    
    with col2:
        p_value_threshold = st.slider(
            "P-value Threshold", 
            min_value=0.01, max_value=0.1, value=0.05, step=0.01,
            help="KS test p-value threshold (<0.05 indicates significant distribution change)"
        )
    
    # Data upload
    st.subheader("📊 Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Reference Data (Training Set)**")
        use_builtin_reference = st.checkbox("Use built-in training data", value=True)
        
        if not use_builtin_reference:
            reference_file = st.file_uploader(
                "Upload reference CSV",
                type=['csv'],
                key="reference_data"
            )
        else:
            reference_file = None
    
    with col2:
        st.write("**New Data (Current/Test Set)**")
        new_file = st.file_uploader(
            "Upload new data CSV",
            type=['csv'],
            key="new_data",
            help="Upload current data to check for drift"
        )
    
    if st.button("🔍 Analyze Drift", type="primary"):
        try:
            # Load reference data
            if use_builtin_reference:
                X_ref, _ = load_data()
                st.success("✅ Loaded built-in training data as reference")
            elif reference_file is not None:
                X_ref = pd.read_csv(reference_file)
                # Remove target column if present
                if 'Class' in X_ref.columns:
                    X_ref = X_ref.drop('Class', axis=1)
                st.success("✅ Loaded uploaded reference data")
            else:
                st.error("Please provide reference data")
                return
            
            # Load new data
            if new_file is not None:
                X_new = pd.read_csv(new_file)
                # Remove target column if present
                if 'Class' in X_new.columns:
                    X_new = X_new.drop('Class', axis=1)
                st.success("✅ Loaded new data for drift analysis")
            else:
                st.error("Please upload new data file")
                return
            
            # Perform drift analysis
            with st.spinner("Analyzing data drift..."):
                drift_df = check_data_drift(X_new, X_ref, psi_threshold, p_value_threshold)
                
                if drift_df.empty:
                    st.warning("No common numeric features found for drift analysis")
                    return
                
                summary = generate_drift_summary(drift_df)
            
            # Display results
            st.subheader("📊 Drift Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Features Analyzed",
                    summary['total_features']
                )
            
            with col2:
                drift_color = "red" if summary['drift_percentage'] > 25 else "orange" if summary['drift_percentage'] > 10 else "green"
                st.metric(
                    "Features with Drift",
                    f"{summary['features_with_drift']} ({summary['drift_percentage']:.1f}%)"
                )
            
            with col3:
                psi_color = "red" if summary['avg_psi'] > 0.25 else "orange" if summary['avg_psi'] > 0.1 else "green"
                st.metric(
                    "Average PSI",
                    f"{summary['avg_psi']:.3f}"
                )
            
            with col4:
                st.metric(
                    "Max PSI",
                    f"{summary['max_psi']:.3f}"
                )
            
            # Drift status alert
            if summary['drift_percentage'] > 50:
                st.error("🚨 **HIGH ALERT**: Major data drift detected. Consider model retraining immediately.")
            elif summary['drift_percentage'] > 25:
                st.warning("⚠️ **MODERATE ALERT**: Significant drift detected. Monitor closely and consider retraining.")
            elif summary['drift_percentage'] > 0:
                st.info("💡 **LOW ALERT**: Minor drift detected. Continue monitoring.")
            else:
                st.success("✅ **NO ALERT**: Data distribution appears stable.")
            
            # Detailed drift table
            st.subheader("📋 Detailed Drift Analysis")
            
            # Add drift level interpretation
            drift_df['drift_level'] = drift_df['psi'].apply(interpret_drift_level)
            
            # Format the dataframe for display
            display_df = drift_df[['feature', 'psi', 'drift_level', 'ks_p_value', 'drift_detected', 'mean_shift']].copy()
            display_df.columns = ['Feature', 'PSI', 'Drift Level', 'KS P-value', 'Drift Detected', 'Mean Shift (σ)']
            
            # Color-code the dataframe
            def highlight_drift(row):
                if row['Drift Detected']:
                    return ['background-color: #ffcdd2'] * len(row)
                elif row['PSI'] > 0.1:
                    return ['background-color: #fff3e0'] * len(row)
                else:
                    return ['background-color: #e8f5e8'] * len(row)
            
            styled_df = display_df.style.apply(highlight_drift, axis=1).format({
                'PSI': '{:.4f}',
                'KS P-value': '{:.4f}',
                'Mean Shift (σ)': '{:.2f}'
            })
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Feature-specific insights
            if summary['features_high_psi']:
                st.subheader("🚨 High-Risk Features")
                for feature in summary['features_high_psi'][:5]:  # Show top 5
                    feature_row = drift_df[drift_df['feature'] == feature].iloc[0]
                    with st.expander(f"📊 {feature} (PSI: {feature_row['psi']:.4f})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Reference Distribution:**")
                            st.write(f"Mean: {feature_row['ref_mean']:.4f}")
                            st.write(f"Std: {feature_row['ref_std']:.4f}")
                        with col2:
                            st.write("**New Distribution:**")
                            st.write(f"Mean: {feature_row['new_mean']:.4f}")
                            st.write(f"Std: {feature_row['new_std']:.4f}")
                        
                        st.write(f"**Mean Shift:** {feature_row['mean_shift']:.2f} standard deviations")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if summary['drift_percentage'] > 50:
                st.markdown("""
                **Immediate Actions Required:**
                - 🔄 **Retrain model** with recent data
                - 📊 **Investigate data sources** for systematic changes
                - ⚠️ **Monitor model performance** closely
                - 🚨 **Consider A/B testing** before full deployment
                """)
            elif summary['drift_percentage'] > 25:
                st.markdown("""
                **Recommended Actions:**
                - 📈 **Increase monitoring frequency**
                - 🔍 **Investigate drifted features** for business changes
                - 📋 **Prepare for potential retraining**
                - 📊 **Collect more recent training data**
                """)
            elif summary['drift_percentage'] > 0:
                st.markdown("""
                **Monitoring Actions:**
                - ✅ **Continue regular drift monitoring**
                - 📊 **Track feature importance changes**
                - 🔍 **Document any business context changes**
                """)
            else:
                st.markdown("""
                **Status: Stable**
                - ✅ **Model appears stable** on current data
                - 📅 **Continue regular monitoring schedule**
                - 🎯 **Focus on performance optimization**
                """)
                
        except Exception as e:
            st.error(f"Error in drift analysis: {str(e)}")
            st.error("Please check your data format and ensure files contain numeric features.")


if __name__ == "__main__":
    main()
