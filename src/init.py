"""
Customer Churn Prediction Package
A comprehensive machine learning solution for predicting customer churn.

This package provides end-to-end functionality for:
- Data ingestion and validation
- Feature engineering
- Model training and evaluation
- Prediction and explanation
- API deployment
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

from src.utils.logger import setup_logger
from src.utils.exceptions import ChurnPredictionError

# Set up root logger
logger = setup_logger()

__all__ = [
    "logger",
    "ChurnPredictionError",
]