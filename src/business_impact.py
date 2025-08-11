"""
Business Impact Analysis for Credit Card Fraud Detection

This module provides functions to calculate business metrics and cost analysis
for fraud detection models, helping stakeholders understand the financial
implications of different model configurations.

Author: Omprakash Mourya
Created: August 2025
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_fraud_detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the fraud detection rate (recall for fraud class).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        float: Fraud detection rate as percentage
    """
    fraud_mask = y_true == 1
    if fraud_mask.sum() == 0:
        return 0.0
    
    true_positives = ((y_pred == 1) & (y_true == 1)).sum()
    total_fraud = fraud_mask.sum()
    
    return (true_positives / total_fraud) * 100


def calculate_cost_savings(y_true: np.ndarray, y_pred: np.ndarray, 
                          cost_fp: float = 1.0, cost_fn: float = 5.0,
                          baseline_pred: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate cost savings compared to baseline model.
    
    Args:
        y_true: True labels
        y_pred: Current model predictions
        cost_fp: Cost of false positive (USD)
        cost_fn: Cost of false negative (USD)
        baseline_pred: Baseline model predictions (if None, assumes all negative)
        
    Returns:
        dict: Cost analysis metrics
    """
    # Current model costs
    fp_current = ((y_pred == 1) & (y_true == 0)).sum()
    fn_current = ((y_pred == 0) & (y_true == 1)).sum()
    total_cost_current = fp_current * cost_fp + fn_current * cost_fn
    
    # Baseline costs (assume all predictions are 0 if no baseline provided)
    if baseline_pred is None:
        fp_baseline = 0
        fn_baseline = (y_true == 1).sum()
    else:
        fp_baseline = ((baseline_pred == 1) & (y_true == 0)).sum()
        fn_baseline = ((baseline_pred == 0) & (y_true == 1)).sum()
    
    total_cost_baseline = fp_baseline * cost_fp + fn_baseline * cost_fn
    
    # Cost savings
    cost_saved = total_cost_baseline - total_cost_current
    savings_per_100k = (cost_saved / len(y_true)) * 100000
    
    return {
        'current_cost': total_cost_current,
        'baseline_cost': total_cost_baseline,
        'cost_saved': cost_saved,
        'savings_per_100k': savings_per_100k,
        'fp_current': fp_current,
        'fn_current': fn_current,
        'fp_baseline': fp_baseline,
        'fn_baseline': fn_baseline
    }


def threshold_cost_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                           cost_fp: float = 1.0, cost_fn: float = 5.0,
                           thresholds: np.ndarray = None) -> pd.DataFrame:
    """
    Analyze costs across different probability thresholds.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        cost_fp: Cost of false positive
        cost_fn: Cost of false negative
        thresholds: Array of thresholds to evaluate
        
    Returns:
        DataFrame with threshold analysis results
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        total_cost = fp * cost_fp + fn * cost_fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'total_cost': total_cost,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'cost_per_100k': (total_cost / len(y_true)) * 100000
        })
    
    return pd.DataFrame(results)


def plot_cost_vs_threshold(threshold_df: pd.DataFrame) -> go.Figure:
    """
    Create cost vs threshold plot.
    
    Args:
        threshold_df: DataFrame from threshold_cost_analysis
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Total cost line
    fig.add_trace(go.Scatter(
        x=threshold_df['threshold'],
        y=threshold_df['total_cost'],
        mode='lines+markers',
        name='Total Cost',
        line=dict(color='red', width=3),
        hovertemplate='Threshold: %{x:.2f}<br>Total Cost: $%{y:,.0f}<extra></extra>'
    ))
    
    # Find optimal threshold
    optimal_idx = threshold_df['total_cost'].idxmin()
    optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
    optimal_cost = threshold_df.loc[optimal_idx, 'total_cost']
    
    # Mark optimal point
    fig.add_trace(go.Scatter(
        x=[optimal_threshold],
        y=[optimal_cost],
        mode='markers',
        name=f'Optimal (θ={optimal_threshold:.2f})',
        marker=dict(color='green', size=12, symbol='star'),
        hovertemplate=f'Optimal Threshold: {optimal_threshold:.2f}<br>Minimum Cost: ${optimal_cost:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Cost vs Classification Threshold',
        xaxis_title='Classification Threshold',
        yaxis_title='Total Cost ($)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_cost_breakdown(cost_analysis: Dict[str, float], cost_fp: float, cost_fn: float) -> go.Figure:
    """
    Create cost breakdown bar chart.
    
    Args:
        cost_analysis: Results from calculate_cost_savings
        cost_fp: Cost per false positive
        cost_fn: Cost per false negative
        
    Returns:
        Plotly figure
    """
    categories = ['Baseline Model', 'Current Model']
    fp_costs = [
        cost_analysis['fp_baseline'] * cost_fp,
        cost_analysis['fp_current'] * cost_fp
    ]
    fn_costs = [
        cost_analysis['fn_baseline'] * cost_fn,
        cost_analysis['fn_current'] * cost_fn
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='False Positive Cost',
        x=categories,
        y=fp_costs,
        marker_color='lightcoral',
        hovertemplate='%{x}<br>FP Cost: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='False Negative Cost',
        x=categories,
        y=fn_costs,
        marker_color='lightsteelblue',
        hovertemplate='%{x}<br>FN Cost: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Cost Breakdown: Baseline vs Current Model',
        xaxis_title='Model Type',
        yaxis_title='Total Cost ($)',
        barmode='stack',
        template='plotly_white'
    )
    
    # Add cost savings annotation
    savings = cost_analysis['cost_saved']
    fig.add_annotation(
        x=1, y=max(fp_costs[1] + fn_costs[1], fp_costs[0] + fn_costs[0]) * 1.1,
        text=f"Savings: ${savings:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="green",
        font=dict(size=14, color="green")
    )
    
    return fig


def generate_business_report(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
                           cost_fp: float = 1.0, cost_fn: float = 5.0) -> Dict[str, Any]:
    """
    Generate comprehensive business impact report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        cost_fp: Cost per false positive
        cost_fn: Cost per false negative
        
    Returns:
        dict: Complete business analysis
    """
    # Basic metrics
    fraud_detection_rate = calculate_fraud_detection_rate(y_true, y_pred)
    cost_analysis = calculate_cost_savings(y_true, y_pred, cost_fp, cost_fn)
    
    # Threshold analysis
    threshold_df = threshold_cost_analysis(y_true, y_proba, cost_fp, cost_fn)
    optimal_threshold = threshold_df.loc[threshold_df['total_cost'].idxmin(), 'threshold']
    
    # Generate plots
    cost_plot = plot_cost_vs_threshold(threshold_df)
    breakdown_plot = plot_cost_breakdown(cost_analysis, cost_fp, cost_fn)
    
    return {
        'fraud_detection_rate': fraud_detection_rate,
        'cost_analysis': cost_analysis,
        'optimal_threshold': optimal_threshold,
        'threshold_df': threshold_df,
        'cost_plot': cost_plot,
        'breakdown_plot': breakdown_plot
    }
