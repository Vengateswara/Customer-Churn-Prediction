"""
Custom exceptions for the Churn Prediction project.
"""

class ChurnPredictionError(Exception):
    """Base exception for churn prediction errors."""
    
    def __init__(self, message: str, error_code: str = "CHURN-000"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class DataIngestionError(ChurnPredictionError):
    """Exception raised for data ingestion errors."""
    
    def __init__(self, message: str):
        super().__init__(message, error_code="CHURN-001")


class DataValidationError(ChurnPredictionError):
    """Exception raised for data validation errors."""
    
    def __init__(self, message: str):
        super().__init__(message, error_code="CHURN-002")


class DataCleaningError(ChurnPredictionError):
    """Exception raised for data cleaning errors."""
    
    def __init__(self, message: str):
        super().__init__(message, error_code="CHURN-003")


class FeatureEngineeringError(ChurnPredictionError):
    """Exception raised for feature engineering errors."""
    
    def __init__(self, message: str):
        super().__init__(message, error_code="CHURN-004")


class ModelTrainingError(ChurnPredictionError):
    """Exception raised for model training errors."""
    
    def __init__(self, message: str):
        super().__init__(message, error_code="CHURN-005")


class ModelPredictionError(ChurnPredictionError):
    """Exception raised for model prediction errors."""
    
    def __init__(self, message: str):
        super().__init__(message, error_code="CHURN-006")


class ConfigurationError(ChurnPredictionError):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str):
        super().__init__(message, error_code="CHURN-007")