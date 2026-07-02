"""
Unit tests for data ingestion and preprocessing modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ingestion import DataIngestor
from src.data.cleaning import DataCleaner
from src.data.preprocessing import DataPreprocessor


class TestDataIngestion:
    """Test data ingestion functionality."""
    
    def test_load_data(self):
        """Test loading data."""
        ingestor = DataIngestor()
        df = ingestor.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'customerID' in df.columns
        assert 'Churn' in df.columns
    
    def test_validate_data(self):
        """Test data validation."""
        ingestor = DataIngestor()
        df = ingestor.load_data()
        
        # Validation should pass
        assert ingestor.validate_data(df) is True


class TestDataCleaning:
    """Test data cleaning functionality."""
    
    def test_handle_missing_values(self):
        """Test handling missing values."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            'TotalCharges': ['100', '200', None],
            'tenure': [1, 2, 3]
        })
        
        df_cleaned = cleaner.handle_missing_values(df)
        
        # Check that missing values are handled
        assert df_cleaned['TotalCharges'].isnull().sum() == 0
        assert pd.to_numeric(df_cleaned['TotalCharges']).dtype == 'float64'
    
    def test_remove_duplicates(self):
        """Test removing duplicates."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            'id': [1, 2, 2, 3],
            'value': ['a', 'b', 'b', 'c']
        })
        
        df_cleaned = cleaner.remove_duplicates(df)
        
        assert len(df_cleaned) == 3
        assert df_cleaned['id'].value_counts()[2] == 1
    
    def test_clean_categorical_values(self):
        """Test cleaning categorical values."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['yes', 'No', 'Y']
        })
        
        df_cleaned = cleaner.clean_categorical_values(df)
        
        assert df_cleaned['PaperlessBilling'].str.lower().str.strip().str.startswith('y').any()


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    def test_split_data(self):
        """Test train-test split."""
        preprocessor = DataPreprocessor()
        
        # Create sample data
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'Churn': np.random.choice(['Yes', 'No'], 100)
        })
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(df)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_create_preprocessing_pipeline(self):
        """Test creating preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        
        # Create sample data
        df = pd.DataFrame({
            'tenure': np.random.randint(1, 72, 100),
            'MonthlyCharges': np.random.uniform(20, 150, 100),
            'TotalCharges': np.random.uniform(100, 5000, 100),
            'gender': np.random.choice(['Female', 'Male'], 100),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 100),
            'Churn': np.random.choice(['Yes', 'No'], 100)
        })
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(df)
        pipeline = preprocessor.create_preprocessing_pipeline(X_train)
        
        # Transform data
        X_transformed = pipeline.transform(X_train)
        
        assert X_transformed is not None
        assert X_transformed.shape[0] == len(X_train)