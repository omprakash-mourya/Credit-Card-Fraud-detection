"""Model explainability using SHAP and LIME."""
import logging
import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

from src.config import MODEL_DIR
from src.utils import ensure_dir


def explain_shap(model, X_sample: np.ndarray, feature_names: List[str], 
                 save_dir: str = MODEL_DIR) -> Dict[str, Any]:
    """Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained model
        X_sample: Sample data for explanation
        feature_names: List of feature names
        save_dir: Directory to save plots
        
    Returns:
        Dictionary containing SHAP values and plots
    """
    if not SHAP_AVAILABLE:
        logging.error("SHAP is not available. Please install: pip install shap")
        return {}
    
    try:
        logging.info("Generating SHAP explanations...")
        
        # Initialize SHAP explainer
        if hasattr(model, 'predict_proba'):
            # For tree-based models
            if hasattr(model, 'get_booster'):  # XGBoost
                explainer = shap.TreeExplainer(model)
            else:
                # For other models, use KernelExplainer with a smaller sample
                background = X_sample[:min(100, len(X_sample))]
                explainer = shap.KernelExplainer(model.predict_proba, background)
        else:
            explainer = shap.KernelExplainer(model.predict, X_sample[:100])
        
        # Calculate SHAP values
        sample_size = min(1000, len(X_sample))
        shap_values = explainer.shap_values(X_sample[:sample_size])
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        
        ensure_dir(save_dir)
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample[:sample_size], 
                         feature_names=feature_names, show=False)
        summary_path = os.path.join(save_dir, 'shap_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample[:sample_size], 
                         feature_names=feature_names, plot_type="bar", show=False)
        bar_path = os.path.join(save_dir, 'shap_bar.png')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(0)
        importance_dict = dict(zip(feature_names, feature_importance))
        
        results = {
            'shap_values': shap_values,
            'feature_importance': importance_dict,
            'summary_plot_path': summary_path,
            'bar_plot_path': bar_path
        }
        
        logging.info(f"SHAP explanations saved to {save_dir}")
        return results
        
    except Exception as e:
        logging.error(f"Error generating SHAP explanations: {str(e)}")
        return {}


def explain_lime(model, pipeline, X_row: np.ndarray, feature_names: List[str],
                 training_data: np.ndarray = None) -> Dict[str, Any]:
    """Generate LIME explanation for a single prediction.
    
    Args:
        model: Trained model
        pipeline: Preprocessing pipeline
        X_row: Single row to explain
        feature_names: List of feature names
        training_data: Training data for LIME background
        
    Returns:
        Dictionary containing LIME explanation
    """
    if not LIME_AVAILABLE:
        logging.error("LIME is not available. Please install: pip install lime")
        return {}
    
    try:
        logging.info("Generating LIME explanation...")
        
        # Prepare training data for LIME
        if training_data is None:
            # Use the input row repeated as background (not ideal but functional)
            training_data = np.tile(X_row, (100, 1))
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=['Normal', 'Fraud'],
            mode='classification',
            discretize_continuous=True
        )
        
        # Define prediction function
        def predict_fn(x):
            return model.predict_proba(x)
        
        # Generate explanation
        explanation = explainer.explain_instance(
            X_row.flatten(),
            predict_fn,
            num_features=min(10, len(feature_names))
        )
        
        # Extract feature weights
        feature_weights = dict(explanation.as_list())
        
        results = {
            'explanation': explanation,
            'feature_weights': feature_weights,
            'html_explanation': explanation.as_html()
        }
        
        logging.info("LIME explanation generated successfully")
        return results
        
    except Exception as e:
        logging.error(f"Error generating LIME explanation: {str(e)}")
        return {}


def plot_feature_importance(importance_dict: Dict[str, float], 
                          title: str = "Feature Importance", 
                          save_path: str = None, 
                          top_n: int = 15) -> plt.Figure:
    """Plot feature importance.
    
    Args:
        importance_dict: Dictionary of feature names and importance scores
        title: Plot title
        save_path: Path to save plot
        top_n: Number of top features to show
        
    Returns:
        Matplotlib figure
    """
    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importances = zip(*top_features)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red' if imp < 0 else 'blue' for imp in importances]
    bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax.text(imp + 0.01 * max(importances), i, f'{imp:.3f}', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Feature importance plot saved to {save_path}")
    
    return fig


def create_explanation_summary(shap_results: Dict[str, Any], 
                             lime_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a summary of model explanations.
    
    Args:
        shap_results: Results from SHAP analysis
        lime_results: Results from LIME analysis (optional)
        
    Returns:
        Summary dictionary
    """
    summary = {
        'top_shap_features': [],
        'shap_plots_available': False,
        'lime_available': False
    }
    
    if shap_results and 'feature_importance' in shap_results:
        # Get top SHAP features
        importance_dict = shap_results['feature_importance']
        sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        summary['top_shap_features'] = sorted_features[:10]
        summary['shap_plots_available'] = 'summary_plot_path' in shap_results
    
    if lime_results and 'feature_weights' in lime_results:
        summary['lime_available'] = True
        summary['lime_top_features'] = list(lime_results['feature_weights'].items())[:10]
    
    return summary
