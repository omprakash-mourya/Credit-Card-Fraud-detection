"""Preprocessing pipeline for Credit Card Fraud Detection."""
import logging
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.utils import save_object, load_object


def get_preprocessing_pipeline() -> ColumnTransformer:
    """Create preprocessing pipeline for credit card fraud data.
    
    Returns:
        ColumnTransformer pipeline
    """
    # All features in the credit card dataset are numeric
    numeric_features = FEATURE_COLUMNS
    
    # Create preprocessing pipeline
    numeric_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'
    )
    
    logging.info(f"Created preprocessing pipeline with {len(numeric_features)} numeric features")
    return preprocessor


def fit_transform_pipeline(df: pd.DataFrame) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    """Fit preprocessing pipeline and transform data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (fitted_pipeline, X_transformed, y)
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"DataFrame must contain target column '{TARGET_COLUMN}'")
    
    # Separate features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # Create and fit pipeline
    pipeline = get_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)
    
    logging.info(f"Fitted pipeline and transformed data: {X_transformed.shape}")
    return pipeline, X_transformed, y.values


def transform_pipeline(pipeline: ColumnTransformer, df: pd.DataFrame) -> np.ndarray:
    """Transform data using fitted pipeline.
    
    Args:
        pipeline: Fitted preprocessing pipeline
        df: Input DataFrame
        
    Returns:
        Transformed data
    """
    X = df[FEATURE_COLUMNS]
    X_transformed = pipeline.transform(X)
    
    logging.info(f"Transformed data: {X_transformed.shape}")
    return X_transformed


def save_pipeline(pipeline: ColumnTransformer, path: str) -> None:
    """Save preprocessing pipeline.
    
    Args:
        pipeline: Fitted preprocessing pipeline
        path: File path to save to
    """
    save_object(pipeline, path)


def load_pipeline(path: str) -> ColumnTransformer:
    """Load preprocessing pipeline.
    
    Args:
        path: File path to load from
        
    Returns:
        Loaded preprocessing pipeline
    """
    return load_object(path)


def get_feature_names(pipeline: ColumnTransformer) -> List[str]:
    """Get feature names after preprocessing.
    
    Args:
        pipeline: Fitted preprocessing pipeline
        
    Returns:
        List of feature names
    """
    try:
        # Try to get feature names from the transformer
        if hasattr(pipeline, 'get_feature_names_out'):
            return pipeline.get_feature_names_out().tolist()
        else:
            # Fallback to original feature names
            return FEATURE_COLUMNS
    except AttributeError:
        # If all else fails, return original feature names
        return FEATURE_COLUMNS


def validate_input_data(df: pd.DataFrame) -> bool:
    """Validate input data format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        True if valid, raises ValueError if not
    """
    missing_features = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check for missing values
    if df[FEATURE_COLUMNS].isnull().any().any():
        logging.warning("Input data contains missing values")
    
    # Check for infinite values
    if np.isinf(df[FEATURE_COLUMNS].values).any():
        raise ValueError("Input data contains infinite values")
    
    logging.info("Input data validation passed")
    return True
