"""
Unit tests for model training and evaluation modules.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train import ModelTrainer


class TestModelTraining:
    """Test model training functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    def test_get_model(self):
        """Test getting model instances."""
        trainer = ModelTrainer()
        
        model = trainer.get_model('LogisticRegression')
        assert model is not None
        
        model = trainer.get_model('RandomForest')
        assert model is not None
    
    def test_train_model(self, sample_data):
        """Test training a single model."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        model = trainer.train_model(X, y, 'RandomForest', use_tuning=False)
        
        assert model is not None
        assert hasattr(model, 'predict')
        
        # Test prediction
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
    
    def test_evaluate_model(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        # Train a simple model
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(X[:150], y[:150])
        
        # Evaluate
        metrics = trainer.evaluate_model(model, X[150:], y[150:])
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
    
    def test_train_all_models(self, sample_data):
        """Test training all models."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        # Train only a subset of models for speed
        trainer.models_config = [
            {'name': 'LogisticRegression', 'enabled': True},
            {'name': 'RandomForest', 'enabled': True}
        ]
        
        results = trainer.train_all_models(X[:150], y[:150], X[150:], y[150:])
        
        assert len(results) > 0
        assert 'LogisticRegression' in results
        assert 'RandomForest' in results