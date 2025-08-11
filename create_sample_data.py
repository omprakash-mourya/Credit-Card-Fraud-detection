"""
Create sample data files for testing the new Streamlit features.
"""
import pandas as pd
import numpy as np
from src.config import FEATURE_COLUMNS

np.random.seed(42)

# Create sample test data with ground truth
n_samples = 500
n_fraud = 10

# Generate normal transactions
normal_data = np.random.normal(0, 1, (n_samples - n_fraud, len(FEATURE_COLUMNS)))

# Generate fraud transactions (slightly different distribution)
fraud_data = np.random.normal(0.3, 1.2, (n_fraud, len(FEATURE_COLUMNS)))

# Combine data
X_data = np.vstack([normal_data, fraud_data])
y_data = np.hstack([np.zeros(n_samples - n_fraud), np.ones(n_fraud)])

# Create DataFrame
sample_df = pd.DataFrame(X_data, columns=FEATURE_COLUMNS)
sample_df['Class'] = y_data.astype(int)

# Save sample files
sample_df.to_csv('data/sample_test_data.csv', index=False)
print(f"Created sample_test_data.csv with {n_samples} transactions ({n_fraud} fraud cases)")

# Create drift data (different distribution)
drift_data = np.random.normal(0.5, 1.5, (300, len(FEATURE_COLUMNS)))
drift_df = pd.DataFrame(drift_data, columns=FEATURE_COLUMNS)
drift_df.to_csv('data/sample_drift_data.csv', index=False)
print(f"Created sample_drift_data.csv with 300 transactions for drift testing")

print("✅ Sample data files created successfully!")
