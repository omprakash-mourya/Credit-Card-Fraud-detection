"""Training module for Credit Card Fraud Detection models."""
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.config import (
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, RANDOMIZED_SEARCH_ITER, 
    XGBOOST_SEARCH_PARAMS, MODEL_DIR
)
from src.utils import save_object, set_seed
from src.data_utils import load_data
from src.preprocess import fit_transform_pipeline, save_pipeline
from src.evaluate import evaluate_model


def train_baseline_logistic(X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray = None, y_test: np.ndarray = None) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Train baseline logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logging.info("Training baseline logistic regression model...")
    
    # Train model with balanced class weights
    model = LogisticRegression(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver='liblinear'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate if test data provided
    metrics = {}
    if X_test is not None and y_test is not None:
        metrics = evaluate_model(model, X_test, y_test)
        logging.info(f"Logistic Regression ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
    
    return model, metrics


def train_xgboost_with_smote(X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray = None, y_test: np.ndarray = None) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Train XGBoost model with SMOTE resampling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        
    Returns:
        Tuple of (best_model, metrics_dict)
    """
    logging.info("Applying SMOTE to training data...")
    
    # Apply SMOTE to training data only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    original_fraud_count = sum(y_train == 1)
    resampled_fraud_count = sum(y_train_resampled == 1)
    logging.info(f"SMOTE: {original_fraud_count} -> {resampled_fraud_count} fraud cases")
    
    logging.info("Training XGBoost with RandomizedSearchCV...")
    
    # Base XGBoost model
    xgb_model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=XGBOOST_SEARCH_PARAMS,
        n_iter=RANDOMIZED_SEARCH_ITER,
        cv=CV_FOLDS,
        scoring='roc_auc',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    random_search.fit(X_train_resampled, y_train_resampled)
    
    best_model = random_search.best_estimator_
    logging.info(f"Best XGBoost parameters: {random_search.best_params_}")
    logging.info(f"Best CV score: {random_search.best_score_:.4f}")
    
    # Save the best model
    model_path = f"{MODEL_DIR}/xgb_model.joblib"
    save_object(best_model, model_path)
    
    # Evaluate if test data provided
    metrics = {}
    if X_test is not None and y_test is not None:
        metrics = evaluate_model(best_model, X_test, y_test)
        metrics['best_cv_score'] = random_search.best_score_
        metrics['best_params'] = random_search.best_params_
        logging.info(f"XGBoost ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
    
    return best_model, metrics


def main():
    """Main training pipeline."""
    # Setup
    set_seed(RANDOM_STATE)
    logging.basicConfig(level=logging.INFO)
    
    logging.info("Starting training pipeline...")
    
    try:
        # Load and preprocess data
        df = load_data()
        pipeline, X, y = fit_transform_pipeline(df)
        
        # Save preprocessing pipeline
        pipeline_path = f"{MODEL_DIR}/pipeline.joblib"
        save_pipeline(pipeline, pipeline_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        logging.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
        logging.info(f"Train fraud rate: {y_train.mean():.4f}")
        logging.info(f"Test fraud rate: {y_test.mean():.4f}")
        
        # Train baseline logistic regression
        lr_model, lr_metrics = train_baseline_logistic(X_train, y_train, X_test, y_test)
        
        # Save logistic regression model
        lr_path = f"{MODEL_DIR}/lr_model.joblib"
        save_object(lr_model, lr_path)
        
        # Train XGBoost with SMOTE
        xgb_model, xgb_metrics = train_xgboost_with_smote(X_train, y_train, X_test, y_test)
        
        # Print results summary
        print("\n" + "="*50)
        print("TRAINING RESULTS SUMMARY")
        print("="*50)
        print(f"Logistic Regression ROC-AUC: {lr_metrics.get('roc_auc', 'N/A'):.4f}")
        print(f"XGBoost ROC-AUC: {xgb_metrics.get('roc_auc', 'N/A'):.4f}")
        print(f"XGBoost Best CV Score: {xgb_metrics.get('best_cv_score', 'N/A'):.4f}")
        print("\nModels saved to:")
        print(f"- {pipeline_path}")
        print(f"- {lr_path}")
        print(f"- {MODEL_DIR}/xgb_model.joblib")
        print("="*50)
        
        logging.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
