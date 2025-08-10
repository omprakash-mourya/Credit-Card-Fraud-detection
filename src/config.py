"""
Configuration settings for the fraud detection project.

This module contains all the constants and configuration parameters
used throughout the project. Centralizing these makes the codebase
more maintainable and allows easy experimentation with different settings.

Author: Data Science Portfolio Project
Created: During model experimentation phase
Last Modified: After hyperparameter optimization
"""
import os

# Random seed for reproducibility
SEED = 42
RANDOM_STATE = 42

# Paths
DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"

# Model configurations
XGBOOST_SEARCH_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# Training settings
TEST_SIZE = 0.2
CV_FOLDS = 3
RANDOMIZED_SEARCH_ITER = 20

# Evaluation settings
COST_FP = 1  # Cost of false positive
COST_FN = 5  # Cost of false negative

# Feature columns (all columns except Class for creditcard.csv)
FEATURE_COLUMNS = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
TARGET_COLUMN = 'Class'

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
