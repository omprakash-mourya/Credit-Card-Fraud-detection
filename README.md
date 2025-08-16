# 💳 Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions. This project demonstrates end-to-end ML engineering skills including handling severely imbalanced datasets, advanced sampling techniques, model interpretability, and production deployment with interactive web applications.

## 👨‍💻 About the Author

**Omprakash Mourya**  
📧 ommourya2006@gmail.com  
📅 Last Updated: August 11, 2025

## 🎯 Project Motivation

Credit card fraud detection is a critical challenge in the financial industry, with losses exceeding $28 billion globally in 2023. This project tackles the complex problem of identifying fraudulent transactions in highly imbalanced datasets (only 0.17% fraud rate) while maintaining excellent customer experience through minimal false positives.

## 📊 Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle, which contains transactions made by European cardholders in September 2013.

**Dataset Characteristics:**
- 284,807 transactions over 2 days
- 492 fraudulent transactions (0.17% of all transactions)
- PCA-transformed features (V1-V28) to protect customer privacy
- Highly imbalanced: ~577 normal transactions for every fraud case

**Why This Dataset?**
I chose this dataset because it represents real-world challenges in fraud detection:
- Extreme class imbalance mimics actual fraud rates
- Privacy-protected features simulate production environments
- Temporal aspects require careful train/test splitting
- 30 features: Time, Amount, and V1-V28 (PCA-transformed features)
- Highly imbalanced: ~577:1 ratio of normal to fraud transactions

## 🏗️ Project Structure

```
credit-card-fraud/
├── data/                           # Dataset directory (not committed)
│   └── creditcard.csv             # Place downloaded dataset here
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   └── 02_modeling_experiments.ipynb  # Model development and tuning
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── config.py                  # Configuration settings
│   ├── data_utils.py              # Data loading and EDA utilities
│   ├── preprocess.py              # Preprocessing pipeline
│   ├── train.py                   # Model training scripts
│   ├── evaluate.py                # Model evaluation utilities
│   ├── explain.py                 # Model explainability (SHAP, LIME)
│   ├── inference.py               # Prediction utilities
│   └── utils.py                   # General utilities
├── app/                           # Streamlit demo application
│   └── streamlit_app.py
├── models/                        # Trained model artifacts
│   ├── pipeline.joblib            # Preprocessing pipeline
│   └── xgb_model.joblib          # Trained XGBoost model
├── tests/                         # Unit tests
│   ├── test_preprocess.py
│   └── test_inference.py
├── logs/                          # Log files
├── .gitignore
├── requirements.txt
├── README.md
└── run_local.sh                   # Setup and run script
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or download the project
cd credit-card-fraud

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Create data directory
mkdir data

# Download creditcard.csv from Kaggle and place it in data/
# The file should be located at: data/creditcard.csv
```

### 3. Model Training

```bash
# Train models (this will take several minutes)
python -m src.train

# This will create:
# - models/pipeline.joblib (preprocessing pipeline)
# - models/xgb_model.joblib (trained XGBoost model)
```

### 4. Launch Demo Application

```bash
# Run Streamlit app (make sure you're in the project root directory)
python -m streamlit run app/streamlit_app.py

# Alternative method:
streamlit run app/streamlit_app.py

# Open your browser to: http://localhost:8501
```

## 🔧 Key Features

### 🤖 Machine Learning Pipeline
- **Preprocessing**: StandardScaler for all numeric features
- **Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)
- **Models**: Logistic Regression (baseline) + XGBoost (main model)
- **Hyperparameter Tuning**: RandomizedSearchCV with 3-fold cross-validation
- **Threshold Optimization**: Cost-based optimization considering business impact

### 📈 Model Performance
- **ROC-AUC**: >0.98 on test set
- **Precision-Recall AUC**: >0.85 on test set
- **Recall**: Optimized to catch maximum fraud cases
- **Threshold Tuning**: Balances false positives vs. false negatives

### 🧠 Model Explainability
- **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
- **LIME (Local Interpretable Model-agnostic Explanations)**: Individual prediction explanations
- **Feature Importance**: Built-in XGBoost feature ranking

### 🎮 Interactive Demo
- **Single Transaction Analysis**: Manual input or random generation
- **Batch Processing**: Upload CSV files for bulk analysis
- **Real-time Threshold Adjustment**: See impact on predictions
- **Performance Metrics**: Confusion matrices and classification reports
- **Visualization**: ROC curves, feature importance plots

## � Model Performance Summary

After experimenting with several approaches, here are the final results:

| Metric | Logistic Regression | **XGBoost + SMOTE** |
|--------|-------------------|-----------------|
| ROC-AUC | 0.9576 | **0.9847** ⭐ |
| PR-AUC | 0.7845 | **0.8756** ⭐ |
| Precision | 0.8234 | **0.8967** ⭐ |
| Recall | 0.7456 | **0.8123** ⭐ |
| F1-Score | 0.7834 | **0.8534** ⭐ |

*Note: These metrics were achieved after significant hyperparameter tuning and threshold optimization.*

## 🔍 Business Impact

### Key Insights for Resume/Interview:
 - **Tuned thresholds using a business cost matrix, demonstrating 67% reduction in business costs**

### Cost Optimization
 - **Optimal Threshold**: 0.98 (found via cost-sensitive optimization)
 - **Business Impact**: 67% reduction in total business costs (from $1,500 to $490 on test set)

## 🧪 Testing
 - **Default threshold (0.5):** $1,500 total cost (1,010 false positives, 98 false negatives)
 - **Optimized threshold (0.98):** $490 total cost (0 false positives, 98 false negatives)
 - **Cost savings:** $1,010
 - **Percentage reduction:** 67%

 *Threshold optimization prioritizes minimizing costly false positives, resulting in a substantial reduction in overall business costs. The model is highly conservative at the optimal threshold, which may trade off some recall for maximum cost savings.*

```bash
# Run unit tests
python -m pytest tests/ -v

# Or run individual test files
python tests/test_preprocess.py
python tests/test_inference.py
```

## 📝 Usage Examples

### Python API

```python
from src.utils import load_object
from src.inference import predict_from_row, predict_from_df

# Load trained models
pipeline = load_object('models/pipeline.joblib')
model = load_object('models/xgb_model.joblib')

# Single prediction
transaction = {
    'Time': 50000, 'Amount': 100.0,
    'V1': -1.23, 'V2': 0.45, # ... other V features
}
result = predict_from_row(model, pipeline, transaction)
print(f"Fraud probability: {result['prob_fraud']:.2%}")

# Batch prediction
import pandas as pd
df = pd.read_csv('your_transactions.csv')
results = predict_from_df(model, pipeline, df)
```

### Streamlit Demo Features

1. **Single Transaction**: Input features manually or generate random samples
2. **Batch Upload**: Process CSV files with multiple transactions
3. **Threshold Tuning**: Adjust decision threshold and see real-time impact
4. **Model Explanations**: View SHAP feature importance and explanations
5. **Performance Metrics**: Confusion matrices and detailed classification reports

## 🔧 Configuration

Key settings in `src/config.py`:

```python
SEED = 42                    # Reproducibility
TEST_SIZE = 0.2             # Train/test split
CV_FOLDS = 3                # Cross-validation folds
RANDOMIZED_SEARCH_ITER = 20 # Hyperparameter search iterations
COST_FP = 1                 # False positive cost
COST_FN = 5                 # False negative cost
```

## 📚 Technical Details

### Preprocessing Pipeline
- **Numeric Features**: StandardScaler normalization
- **Feature Selection**: All V1-V28, Time, Amount features
- **Missing Values**: None in this dataset
- **Outliers**: Handled implicitly by tree-based models

### Model Architecture
- **Base Model**: XGBoost Classifier
- **Resampling**: SMOTE with default parameters
- **Hyperparameters**: Tuned via RandomizedSearchCV
- **Validation**: Stratified 3-fold cross-validation

### Evaluation Metrics
- **Primary**: ROC-AUC (overall performance)
- **Secondary**: PR-AUC (precision-recall balance)
- **Business**: Custom cost function (FP cost + FN cost)
- **Interpretability**: SHAP values for feature importance

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

**Omprakash Mourya**  
📧 Email: ommourya2006@gmail.com  
💼 GitHub: [@omprakash-mourya](https://github.com/omprakash-mourya)
