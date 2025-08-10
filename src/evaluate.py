"""Evaluation utilities for Credit Card Fraud Detection models."""
import logging
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    average_precision_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score
)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate model performance with comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': classification_report(y_test, y_pred)
    }
    
    logging.info(f"Model evaluation completed - ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    return metrics


def plot_roc_curve(model, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None) -> plt.Figure:
    """Plot ROC curve.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"ROC curve saved to {save_path}")
    
    return fig


def plot_precision_recall_curve(model, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None) -> plt.Figure:
    """Plot Precision-Recall curve.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.axhline(y=y_test.mean(), color='red', linestyle='--', label=f'Baseline ({y_test.mean():.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"PR curve saved to {save_path}")
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def threshold_tuning(probabilities: np.ndarray, y_true: np.ndarray, 
                    cost_fp: float = 1, cost_fn: float = 5) -> Tuple[float, Dict[str, Any]]:
    """Tune decision threshold based on cost function.
    
    Args:
        probabilities: Predicted probabilities
        y_true: True labels
        cost_fp: Cost of false positive
        cost_fn: Cost of false negative
        
    Returns:
        Tuple of (optimal_threshold, results_dict)
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    costs = []
    metrics_by_threshold = []
    
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases where only one class is predicted
            if len(np.unique(y_pred)) == 1:
                if y_pred[0] == 0:  # All predicted as negative
                    tp, fp, fn, tn = 0, 0, sum(y_true), len(y_true) - sum(y_true)
                else:  # All predicted as positive
                    tp, fp, fn, tn = sum(y_true), len(y_true) - sum(y_true), 0, 0
            else:
                continue
        
        total_cost = fp * cost_fp + fn * cost_fn
        costs.append(total_cost)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_by_threshold.append({
            'threshold': threshold,
            'cost': total_cost,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })
    
    # Find optimal threshold
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs[optimal_idx]
    
    results = {
        'optimal_threshold': optimal_threshold,
        'optimal_cost': optimal_cost,
        'all_metrics': metrics_by_threshold,
        'thresholds': thresholds,
        'costs': costs
    }
    
    logging.info(f"Optimal threshold: {optimal_threshold:.3f} (cost: {optimal_cost:.2f})")
    return optimal_threshold, results


def plot_threshold_analysis(threshold_results: Dict[str, Any], save_path: str = None) -> plt.Figure:
    """Plot threshold tuning analysis.
    
    Args:
        threshold_results: Results from threshold_tuning function
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure
    """
    thresholds = threshold_results['thresholds']
    costs = threshold_results['costs']
    optimal_threshold = threshold_results['optimal_threshold']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cost vs Threshold
    ax1.plot(thresholds, costs, 'b-', linewidth=2)
    ax1.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal: {optimal_threshold:.3f}')
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('Total Cost')
    ax1.set_title('Cost vs Decision Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision, Recall, F1 vs Threshold
    metrics_df = threshold_results['all_metrics']
    precisions = [m['precision'] for m in metrics_df]
    recalls = [m['recall'] for m in metrics_df]
    f1s = [m['f1'] for m in metrics_df]
    
    ax2.plot(thresholds, precisions, 'g-', label='Precision', linewidth=2)
    ax2.plot(thresholds, recalls, 'b-', label='Recall', linewidth=2)
    ax2.plot(thresholds, f1s, 'r-', label='F1-Score', linewidth=2)
    ax2.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Metrics vs Decision Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Threshold analysis plot saved to {save_path}")
    
    return fig
