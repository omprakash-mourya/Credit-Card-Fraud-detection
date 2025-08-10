"""Test inference functionality."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from unittest.mock import Mock

from src.inference import (
    predict_from_row,
    predict_from_df,
    create_sample_transaction,
    validate_prediction_input
)
from src.config import FEATURE_COLUMNS


class TestInference:
    """Test inference functionality."""
    
    def create_mock_model_and_pipeline(self):
        """Create mock model and pipeline for testing."""
        # Mock model
        model = Mock()
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.9, 0.1]])
        model.predict.return_value = np.array([0, 0])
        
        # Mock pipeline
        pipeline = Mock()
        pipeline.transform.return_value = np.array([[1, 2, 3], [4, 5, 6]])
        
        return model, pipeline
    
    def test_predict_from_row(self):
        """Test single row prediction."""
        model, pipeline = self.create_mock_model_and_pipeline()
        
        # Create sample transaction
        row_dict = create_sample_transaction()
        
        # Test prediction
        result = predict_from_row(model, pipeline, row_dict)
        
        assert isinstance(result, dict)
        assert 'prob_fraud' in result
        assert 'prob_normal' in result
        assert 'prediction' in result
        assert 'prediction_label' in result
        
        assert 0 <= result['prob_fraud'] <= 1
        assert 0 <= result['prob_normal'] <= 1
        assert abs(result['prob_fraud'] + result['prob_normal'] - 1.0) < 1e-6
        assert result['prediction'] in [0, 1]
        assert result['prediction_label'] in ['Normal', 'Fraud']
    
    def test_predict_from_df(self):
        """Test batch prediction."""
        model, pipeline = self.create_mock_model_and_pipeline()
        
        # Create sample DataFrame
        sample_data = {}
        for feature in FEATURE_COLUMNS:
            sample_data[feature] = [1.0, 2.0]
        
        df = pd.DataFrame(sample_data)
        
        # Test prediction
        result_df = predict_from_df(model, pipeline, df)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'prob_fraud' in result_df.columns
        assert 'prob_normal' in result_df.columns
        assert 'pred_label' in result_df.columns
        assert 'prediction' in result_df.columns
        
        assert len(result_df) == len(df)
        assert all(0 <= prob <= 1 for prob in result_df['prob_fraud'])
        assert all(0 <= prob <= 1 for prob in result_df['prob_normal'])
    
    def test_create_sample_transaction(self):
        """Test sample transaction creation."""
        sample = create_sample_transaction()
        
        assert isinstance(sample, dict)
        assert len(sample) == len(FEATURE_COLUMNS)
        
        # Check all required features are present
        for feature in FEATURE_COLUMNS:
            assert feature in sample
            assert isinstance(sample[feature], (int, float))
        
        # Test with custom means
        custom_means = {feature: 1.0 for feature in FEATURE_COLUMNS}
        sample_custom = create_sample_transaction(custom_means)
        
        assert isinstance(sample_custom, dict)
        assert len(sample_custom) == len(FEATURE_COLUMNS)
    
    def test_validate_prediction_input(self):
        """Test input validation for predictions."""
        # Valid data
        sample_data = {}
        for feature in FEATURE_COLUMNS:
            sample_data[feature] = [1.0, 2.0, 3.0]
        
        df_valid = pd.DataFrame(sample_data)
        assert validate_prediction_input(df_valid) == True
        
        # Valid dict input
        row_dict = {feature: 1.0 for feature in FEATURE_COLUMNS}
        assert validate_prediction_input(row_dict) == True
        
        # Invalid data - missing features
        df_invalid = pd.DataFrame({'V1': [1, 2, 3]})
        try:
            validate_prediction_input(df_invalid)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Invalid data - infinite values
        sample_data['V1'] = [1.0, np.inf, 3.0]
        df_inf = pd.DataFrame(sample_data)
        try:
            validate_prediction_input(df_inf)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected


if __name__ == "__main__":
    # Run tests
    test_instance = TestInference()
    
    print("Running inference tests...")
    
    try:
        test_instance.test_predict_from_row()
        print("✅ test_predict_from_row passed")
    except Exception as e:
        print(f"❌ test_predict_from_row failed: {e}")
    
    try:
        test_instance.test_predict_from_df()
        print("✅ test_predict_from_df passed")
    except Exception as e:
        print(f"❌ test_predict_from_df failed: {e}")
    
    try:
        test_instance.test_create_sample_transaction()
        print("✅ test_create_sample_transaction passed")
    except Exception as e:
        print(f"❌ test_create_sample_transaction failed: {e}")
    
    try:
        test_instance.test_validate_prediction_input()
        print("✅ test_validate_prediction_input passed")
    except Exception as e:
        print(f"❌ test_validate_prediction_input failed: {e}")
    
    print("Inference tests completed!")
