"""Test preprocessing functionality."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocess import (
    get_preprocessing_pipeline, 
    fit_transform_pipeline,
    validate_input_data,
    get_feature_names
)
from src.config import FEATURE_COLUMNS, TARGET_COLUMN


class TestPreprocessing:
    """Test preprocessing functionality."""
    
    def test_get_preprocessing_pipeline(self):
        """Test that preprocessing pipeline is created correctly."""
        pipeline = get_preprocessing_pipeline()
        
        assert isinstance(pipeline, ColumnTransformer)
        assert len(pipeline.transformers) == 1
        assert pipeline.transformers[0][0] == 'num'
    
    def test_fit_transform_pipeline(self):
        """Test pipeline fitting and transformation."""
        # Create sample data
        np.random.seed(42)
        sample_data = {}
        
        # Add feature columns
        for feature in FEATURE_COLUMNS:
            sample_data[feature] = np.random.normal(0, 1, 100)
        
        # Add target column
        sample_data[TARGET_COLUMN] = np.random.randint(0, 2, 100)
        
        df = pd.DataFrame(sample_data)
        
        # Test pipeline fitting
        pipeline, X_transformed, y = fit_transform_pipeline(df)
        
        assert isinstance(pipeline, ColumnTransformer)
        assert X_transformed.shape[0] == 100
        assert X_transformed.shape[1] == len(FEATURE_COLUMNS)
        assert len(y) == 100
        assert X_transformed.dtype in [np.float32, np.float64]
    
    def test_validate_input_data(self):
        """Test input data validation."""
        # Valid data
        sample_data = {}
        for feature in FEATURE_COLUMNS:
            sample_data[feature] = [1.0, 2.0, 3.0]
        
        df_valid = pd.DataFrame(sample_data)
        assert validate_input_data(df_valid) == True
        
        # Invalid data - missing features
        df_invalid = pd.DataFrame({'V1': [1, 2, 3]})
        with pytest.raises(ValueError):
            validate_input_data(df_invalid)
        
        # Invalid data - infinite values
        sample_data['V1'] = [1.0, np.inf, 3.0]
        df_inf = pd.DataFrame(sample_data)
        with pytest.raises(ValueError):
            validate_input_data(df_inf)
    
    def test_get_feature_names(self):
        """Test feature name extraction."""
        pipeline = get_preprocessing_pipeline()
        
        # Create sample data to fit pipeline
        sample_data = {}
        for feature in FEATURE_COLUMNS:
            sample_data[feature] = [1.0, 2.0, 3.0]
        
        df = pd.DataFrame(sample_data)
        pipeline.fit(df)
        
        feature_names = get_feature_names(pipeline)
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0


if __name__ == "__main__":
    # Run tests
    test_instance = TestPreprocessing()
    
    print("Running preprocessing tests...")
    
    try:
        test_instance.test_get_preprocessing_pipeline()
        print("✅ test_get_preprocessing_pipeline passed")
    except Exception as e:
        print(f"❌ test_get_preprocessing_pipeline failed: {e}")
    
    try:
        test_instance.test_fit_transform_pipeline()
        print("✅ test_fit_transform_pipeline passed")
    except Exception as e:
        print(f"❌ test_fit_transform_pipeline failed: {e}")
    
    try:
        test_instance.test_validate_input_data()
        print("✅ test_validate_input_data passed")
    except Exception as e:
        print(f"❌ test_validate_input_data failed: {e}")
    
    try:
        test_instance.test_get_feature_names()
        print("✅ test_get_feature_names passed")
    except Exception as e:
        print(f"❌ test_get_feature_names failed: {e}")
    
    print("Preprocessing tests completed!")
