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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.utils import load_object
    from src.inference import predict_from_row, predict_from_df, create_sample_transaction
    from src.evaluate import plot_confusion_matrix
    from src.config import FEATURE_COLUMNS, MODEL_DIR
    from src.explain import explain_shap, plot_feature_importance
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure you are running from the project root directory")
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
        
        return pipeline, model, "Models loaded successfully!"
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
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction", "📈 Model Info", "🧠 Explanations"])
    
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


if __name__ == "__main__":
    main()
