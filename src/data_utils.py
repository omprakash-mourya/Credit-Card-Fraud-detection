"""Data loading and EDA utilities for Credit Card Fraud Detection."""
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from src.config import DATA_PATH, SEED


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the credit card fraud dataset.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If the data file is not found
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path}: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Data file not found at {path}")
        logging.error("Please download creditcard.csv from Kaggle and place it in data/creditcard.csv")
        raise


def basic_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform basic exploratory data analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing EDA results
    """
    if 'Class' not in df.columns:
        raise ValueError("DataFrame must contain 'Class' column")
    
    class_counts = df['Class'].value_counts()
    total_samples = len(df)
    fraud_fraction = class_counts.get(1, 0) / total_samples
    
    eda_results = {
        'total_samples': total_samples,
        'class_counts': class_counts.to_dict(),
        'fraud_fraction': fraud_fraction,
        'normal_fraction': 1 - fraud_fraction,
        'missing_values': df.isnull().sum().sum(),
        'feature_count': df.shape[1] - 1,  # Excluding target
        'descriptive_stats': df.describe().head().to_dict()
    }
    
    logging.info(f"EDA completed: {fraud_fraction:.4f} fraud rate, {total_samples} samples")
    return eda_results


def save_sample_for_demo(df: pd.DataFrame, n: int = 5000, output_path: str = "data/demo_sample.csv") -> None:
    """Save a balanced sample for quick demo purposes.
    
    Args:
        df: Input DataFrame
        n: Total number of samples to save
        output_path: Path to save the sample
    """
    if 'Class' not in df.columns:
        raise ValueError("DataFrame must contain 'Class' column")
    
    # Get equal number of fraud and normal cases
    fraud_cases = df[df['Class'] == 1].sample(n=min(n//2, len(df[df['Class'] == 1])), random_state=SEED)
    normal_cases = df[df['Class'] == 0].sample(n=n//2, random_state=SEED)
    
    sample_df = pd.concat([fraud_cases, normal_cases]).sample(frac=1, random_state=SEED)
    sample_df.to_csv(output_path, index=False)
    
    logging.info(f"Saved {len(sample_df)} samples to {output_path}")
    logging.info(f"Fraud rate in sample: {sample_df['Class'].mean():.4f}")


def get_feature_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get information about features in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with feature information
    """
    feature_info = {
        'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features': df.select_dtypes(exclude=[np.number]).columns.tolist(),
        'feature_ranges': {},
        'correlation_with_target': {}
    }
    
    # Calculate feature ranges for numeric features
    for col in feature_info['numeric_features']:
        if col != 'Class':
            feature_info['feature_ranges'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
    
    # Calculate correlation with target if it exists
    if 'Class' in df.columns:
        for col in feature_info['numeric_features']:
            if col != 'Class':
                feature_info['correlation_with_target'][col] = df[col].corr(df['Class'])
    
    return feature_info
