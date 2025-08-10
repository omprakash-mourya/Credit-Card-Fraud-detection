"""Utility functions for the Credit Card Fraud Detection project."""
import logging
import os
import random
from typing import Any
import joblib
import numpy as np
from sklearn.base import BaseEstimator


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_object(obj: Any, path: str) -> None:
    """Save object using joblib.
    
    Args:
        obj: Object to save
        path: File path to save to
    """
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)
    logging.info(f"Saved object to {path}")


def load_object(path: str) -> Any:
    """Load object using joblib.
    
    Args:
        path: File path to load from
        
    Returns:
        Loaded object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    obj = joblib.load(path)
    logging.info(f"Loaded object from {path}")
    return obj


def ensure_dir(path: str) -> None:
    """Ensure directory exists.
    
    Args:
        path: Directory path
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def setup_logging(log_file: str = "logs/project.log") -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_file: Path to log file
        
    Returns:
        Configured logger
    """
    ensure_dir(os.path.dirname(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
