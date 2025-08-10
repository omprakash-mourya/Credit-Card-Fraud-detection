"""Inference utilities for Credit Card Fraud Detection."""
import logging
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from src.config import FEATURE_COLUMNS
from src.preprocess import validate_input_data


def predict_from_row(model, pipeline, row_dict: Dict[str, float]) -> Dict[str, Any]:
    """Predict fraud probability for a single transaction.
    
    Args:
        model: Trained model
        pipeline: Preprocessing pipeline
        row_dict: Dictionary with feature names as keys and values
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Convert to DataFrame
        df_row = pd.DataFrame([row_dict])
        
        # Validate input
        validate_input_data(df_row)
        
        # Preprocess
        X_processed = pipeline.transform(df_row)
        
        # Predict
        prob_fraud = model.predict_proba(X_processed)[0, 1]
        prediction = model.predict(X_processed)[0]
        
        result = {
            'prob_fraud': float(prob_fraud),
            'prob_normal': float(1 - prob_fraud),
            'prediction': int(prediction),
            'prediction_label': 'Fraud' if prediction == 1 else 'Normal'
        }
        
        logging.info(f"Single prediction: {result['prediction_label']} (prob: {prob_fraud:.4f})")
        return result
        
    except Exception as e:
        logging.error(f"Error in single prediction: {str(e)}")
        raise


def predict_from_df(model, pipeline, df: pd.DataFrame, 
                   threshold: float = 0.5) -> pd.DataFrame:
    """Predict fraud probabilities for a DataFrame of transactions.
    
    Args:
        model: Trained model
        pipeline: Preprocessing pipeline
        df: Input DataFrame
        threshold: Decision threshold for binary prediction
        
    Returns:
        DataFrame with added prediction columns
    """
    try:
        # Validate input
        validate_input_data(df)
        
        # Preprocess
        X_processed = pipeline.transform(df)
        
        # Predict
        probabilities = model.predict_proba(X_processed)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['prob_fraud'] = probabilities
        result_df['prob_normal'] = 1 - probabilities
        result_df['pred_label'] = predictions
        result_df['prediction'] = np.where(predictions == 1, 'Fraud', 'Normal')
        
        logging.info(f"Batch prediction completed: {len(df)} transactions, "
                    f"{sum(predictions)} predicted as fraud")
        
        return result_df
        
    except Exception as e:
        logging.error(f"Error in batch prediction: {str(e)}")
        raise


def predict_with_threshold(model, pipeline, df: pd.DataFrame, 
                          threshold: float) -> Dict[str, Any]:
    """Make predictions with custom threshold and return detailed results.
    
    Args:
        model: Trained model
        pipeline: Preprocessing pipeline
        df: Input DataFrame
        threshold: Custom decision threshold
        
    Returns:
        Dictionary with detailed prediction results
    """
    try:
        # Get predictions
        result_df = predict_from_df(model, pipeline, df, threshold)
        
        # Calculate summary statistics
        total_transactions = len(result_df)
        predicted_fraud = sum(result_df['pred_label'])
        predicted_normal = total_transactions - predicted_fraud
        avg_fraud_prob = result_df['prob_fraud'].mean()
        max_fraud_prob = result_df['prob_fraud'].max()
        min_fraud_prob = result_df['prob_fraud'].min()
        
        # If true labels are available, calculate performance metrics
        performance_metrics = {}
        if 'Class' in df.columns:
            from sklearn.metrics import confusion_matrix, classification_report
            
            y_true = df['Class']
            y_pred = result_df['pred_label']
            
            cm = confusion_matrix(y_true, y_pred)
            performance_metrics = {
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
        
        summary = {
            'threshold_used': threshold,
            'total_transactions': total_transactions,
            'predicted_fraud': int(predicted_fraud),
            'predicted_normal': int(predicted_normal),
            'fraud_rate': predicted_fraud / total_transactions,
            'avg_fraud_probability': float(avg_fraud_prob),
            'max_fraud_probability': float(max_fraud_prob),
            'min_fraud_probability': float(min_fraud_prob),
            'detailed_predictions': result_df.to_dict('records'),
            'performance_metrics': performance_metrics
        }
        
        return summary
        
    except Exception as e:
        logging.error(f"Error in threshold prediction: {str(e)}")
        raise


def create_sample_transaction(feature_means: Dict[str, float] = None) -> Dict[str, float]:
    """Create a sample transaction for testing.
    
    Args:
        feature_means: Dictionary of feature means (optional)
        
    Returns:
        Dictionary representing a sample transaction
    """
    if feature_means is None:
        # Default values (approximating normal transaction)
        sample = {}
        
        # V1-V28 features (PCA components, typically near 0)
        for i in range(1, 29):
            sample[f'V{i}'] = np.random.normal(0, 1)
        
        # Time (seconds since first transaction, arbitrary value)
        sample['Time'] = 50000.0
        
        # Amount (typical transaction amount)
        sample['Amount'] = 100.0
    else:
        # Use provided means with some random variation
        sample = {}
        for feature in FEATURE_COLUMNS:
            if feature in feature_means:
                mean_val = feature_means[feature]
                # Add small random variation
                sample[feature] = mean_val + np.random.normal(0, abs(mean_val) * 0.1)
            else:
                sample[feature] = 0.0
    
    return sample


def validate_prediction_input(data: Union[Dict, pd.DataFrame]) -> bool:
    """Validate input data for prediction.
    
    Args:
        data: Input data (dict or DataFrame)
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If input is invalid
    """
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Check required features
    missing_features = set(FEATURE_COLUMNS) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check for invalid values
    numeric_data = data[FEATURE_COLUMNS]
    if numeric_data.isnull().any().any():
        raise ValueError("Input contains missing values")
    
    if np.isinf(numeric_data.values).any():
        raise ValueError("Input contains infinite values")
    
    return True
