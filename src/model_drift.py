"""
Model Drift Detection for Credit Card Fraud Detection

This module provides functions to detect data drift between training
and new incoming data using statistical methods like Population Stability Index (PSI)
and Kolmogorov-Smirnov tests.

Author: Omprakash Mourya
Created: August 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import logging
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.
    
    PSI measures the shift in distribution between two samples:
    - PSI < 0.1: No significant shift
    - 0.1 <= PSI < 0.25: Moderate shift
    - PSI >= 0.25: Significant shift (model drift warning)
    
    Args:
        expected: Reference/training distribution
        actual: New/test distribution  
        buckets: Number of quantile buckets for discretization
        
    Returns:
        float: PSI value
    """
    def scale_range(input_data, min_val, max_val):
        input_data = (input_data - min_val) / (max_val - min_val)
        return input_data
    
    # Handle edge cases
    if len(expected) == 0 or len(actual) == 0:
        return float('inf')
    
    # Scale both arrays to the same range
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    
    if max_val == min_val:
        return 0.0  # No variation
    
    expected_scaled = scale_range(expected, min_val, max_val)
    actual_scaled = scale_range(actual, min_val, max_val)
    
    # Create buckets based on expected distribution quantiles
    breakpoints = np.linspace(0, 1, buckets + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Convert to actual values
    if buckets <= len(np.unique(expected_scaled)):
        breakpoints = np.quantile(expected_scaled, np.linspace(0, 1, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
    
    # Calculate frequencies
    expected_freq = np.histogram(expected_scaled, breakpoints)[0]
    actual_freq = np.histogram(actual_scaled, breakpoints)[0]
    
    # Convert to percentages and handle zeros
    expected_perc = expected_freq / len(expected_scaled)
    actual_perc = actual_freq / len(actual_scaled)
    
    # Add small constant to avoid log(0)
    epsilon = 1e-6
    expected_perc = np.where(expected_perc == 0, epsilon, expected_perc)
    actual_perc = np.where(actual_perc == 0, epsilon, actual_perc)
    
    # Calculate PSI
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    
    return psi


def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test for distribution comparison.
    
    Args:
        reference: Reference distribution
        current: Current distribution
        
    Returns:
        tuple: (KS statistic, p-value)
    """
    try:
        ks_stat, p_value = stats.ks_2samp(reference, current)
        return ks_stat, p_value
    except Exception as e:
        logger.warning(f"KS test failed: {e}")
        return np.nan, np.nan


def check_data_drift(new_data: pd.DataFrame, reference_data: pd.DataFrame,
                    psi_threshold: float = 0.25, p_value_threshold: float = 0.05) -> pd.DataFrame:
    """
    Check for data drift between reference and new datasets.
    
    Args:
        new_data: New/current dataset
        reference_data: Reference/training dataset
        psi_threshold: PSI threshold for drift detection (default: 0.25)
        p_value_threshold: P-value threshold for KS test (default: 0.05)
        
    Returns:
        DataFrame with drift analysis results
    """
    # Get common numeric columns
    numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
    common_cols = [col for col in numeric_cols if col in new_data.columns]
    
    if not common_cols:
        logger.warning("No common numeric columns found for drift analysis")
        return pd.DataFrame()
    
    results = []
    
    for col in common_cols:
        try:
            ref_values = reference_data[col].dropna().values
            new_values = new_data[col].dropna().values
            
            if len(ref_values) == 0 or len(new_values) == 0:
                continue
            
            # Calculate PSI
            psi_value = calculate_psi(ref_values, new_values)
            psi_drift = psi_value >= psi_threshold
            
            # Perform KS test
            ks_stat, p_value = ks_test(ref_values, new_values)
            ks_drift = p_value < p_value_threshold if not np.isnan(p_value) else False
            
            # Determine overall drift status
            drift_detected = psi_drift or ks_drift
            
            # Calculate basic statistics for interpretation
            ref_mean, ref_std = np.mean(ref_values), np.std(ref_values)
            new_mean, new_std = np.mean(new_values), np.std(new_values)
            
            mean_shift = abs(new_mean - ref_mean) / ref_std if ref_std > 0 else 0
            
            results.append({
                'feature': col,
                'psi': psi_value,
                'psi_drift': psi_drift,
                'ks_statistic': ks_stat,
                'ks_p_value': p_value,
                'ks_drift': ks_drift,
                'drift_detected': drift_detected,
                'mean_shift': mean_shift,
                'ref_mean': ref_mean,
                'new_mean': new_mean,
                'ref_std': ref_std,
                'new_std': new_std
            })
            
        except Exception as e:
            logger.error(f"Error processing column {col}: {e}")
            continue
    
    drift_df = pd.DataFrame(results)
    
    if not drift_df.empty:
        # Sort by PSI value (descending) to show most concerning features first
        drift_df = drift_df.sort_values('psi', ascending=False)
    
    return drift_df


def generate_drift_summary(drift_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for drift analysis.
    
    Args:
        drift_df: Drift analysis results DataFrame
        
    Returns:
        dict: Summary statistics
    """
    if drift_df.empty:
        return {
            'total_features': 0,
            'features_with_drift': 0,
            'drift_percentage': 0,
            'avg_psi': 0,
            'max_psi': 0,
            'features_high_psi': [],
            'features_ks_drift': []
        }
    
    total_features = len(drift_df)
    features_with_drift = drift_df['drift_detected'].sum()
    drift_percentage = (features_with_drift / total_features) * 100
    
    avg_psi = drift_df['psi'].mean()
    max_psi = drift_df['psi'].max()
    
    # Features with high PSI (>= 0.25)
    features_high_psi = drift_df[drift_df['psi'] >= 0.25]['feature'].tolist()
    
    # Features with significant KS test results
    features_ks_drift = drift_df[drift_df['ks_drift'] == True]['feature'].tolist()
    
    return {
        'total_features': total_features,
        'features_with_drift': features_with_drift,
        'drift_percentage': drift_percentage,
        'avg_psi': avg_psi,
        'max_psi': max_psi,
        'features_high_psi': features_high_psi,
        'features_ks_drift': features_ks_drift
    }


def interpret_drift_level(psi_value: float) -> str:
    """
    Interpret PSI value into drift level description.
    
    Args:
        psi_value: PSI value
        
    Returns:
        str: Drift level interpretation
    """
    if psi_value < 0.1:
        return "No Shift"
    elif psi_value < 0.25:
        return "Moderate Shift"
    else:
        return "Significant Shift"


def format_drift_report(drift_df: pd.DataFrame, summary: Dict[str, Any]) -> str:
    """
    Format drift analysis results into a readable report.
    
    Args:
        drift_df: Drift analysis DataFrame
        summary: Summary statistics
        
    Returns:
        str: Formatted report
    """
    if drift_df.empty:
        return "No features available for drift analysis."
    
    report = f"""
    📊 Model Drift Analysis Report
    ═══════════════════════════════
    
    📈 Summary Statistics:
    • Total Features Analyzed: {summary['total_features']}
    • Features with Drift: {summary['features_with_drift']} ({summary['drift_percentage']:.1f}%)
    • Average PSI: {summary['avg_psi']:.3f}
    • Maximum PSI: {summary['max_psi']:.3f}
    
    🚨 High-Risk Features (PSI ≥ 0.25):
    {', '.join(summary['features_high_psi']) if summary['features_high_psi'] else 'None'}
    
    📊 Statistical Significance (KS Test):
    {', '.join(summary['features_ks_drift']) if summary['features_ks_drift'] else 'None'}
    
    💡 Recommendations:
    """
    
    if summary['drift_percentage'] > 50:
        report += "• HIGH ALERT: Major data drift detected. Consider model retraining.\n"
    elif summary['drift_percentage'] > 25:
        report += "• MODERATE ALERT: Significant drift in multiple features. Monitor closely.\n"
    elif summary['drift_percentage'] > 0:
        report += "• LOW ALERT: Minor drift detected in some features. Continue monitoring.\n"
    else:
        report += "• NO ALERT: Data distribution appears stable.\n"
    
    return report
