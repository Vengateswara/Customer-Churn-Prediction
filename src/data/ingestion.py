"""
Data ingestion module for loading and validating customer churn data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import yaml
from loguru import logger

from src.utils.exceptions import DataIngestionError, DataValidationError
from src.utils.logger import LoggerMixin


class DataIngestor(LoggerMixin):
    """
    Handles data loading and initial validation for the churn prediction project.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the DataIngestor with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.data_path = Path(self.config["data"]["raw_path"])
        self.target_column = self.config["data"]["target_column"]
        self.logger.info(f"DataIngestor initialized with config from {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the config file
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            DataIngestionError: If config file cannot be loaded
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            raise DataIngestionError(f"Failed to load config file: {str(e)}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the raw customer churn dataset.
        
        Returns:
            DataFrame containing the raw data
            
        Raises:
            DataIngestionError: If data cannot be loaded
        """
        try:
            if not self.data_path.exists():
                raise DataIngestionError(f"Data file not found: {self.data_path}")
            
            self.logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise DataIngestionError(f"Failed to load data: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the loaded data for required columns and basic quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.info("Validating data quality...")
            
            # Check if DataFrame is empty
            if df.empty:
                raise DataValidationError("DataFrame is empty")
            
            # Check for required columns
            required_columns = [
                'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
            ]
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise DataValidationError(f"Missing required columns: {missing_cols}")
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                self.logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            
            # Check data types
            expected_dtypes = {
                'customerID': 'object',
                'gender': 'object',
                'SeniorCitizen': 'int64',
                'Partner': 'object',
                'Dependents': 'object',
                'tenure': 'int64',
                'PhoneService': 'object',
                'MultipleLines': 'object',
                'InternetService': 'object',
                'OnlineSecurity': 'object',
                'OnlineBackup': 'object',
                'DeviceProtection': 'object',
                'TechSupport': 'object',
                'StreamingTV': 'object',
                'StreamingMovies': 'object',
                'Contract': 'object',
                'PaperlessBilling': 'object',
                'PaymentMethod': 'object',
                'MonthlyCharges': 'float64',
                'TotalCharges': 'object',  # May contain missing values
                'Churn': 'object'
            }
            
            for col, expected_dtype in expected_dtypes.items():
                if col in df.columns:
                    actual_dtype = df[col].dtype
                    if actual_dtype != expected_dtype and col != 'TotalCharges':
                        self.logger.warning(f"Column {col} has dtype {actual_dtype}, expected {expected_dtype}")
            
            # Validate target column values
            if self.target_column in df.columns:
                unique_values = df[self.target_column].unique()
                expected_values = ['Yes', 'No']
                if not all(val in expected_values for val in unique_values):
                    self.logger.warning(f"Unexpected values in target column: {unique_values}")
            
            self.logger.info("Data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise DataValidationError(f"Data validation failed: {str(e)}")
    
    def ingest_and_validate(self) -> pd.DataFrame:
        """
        Complete data ingestion pipeline: load and validate.
        
        Returns:
            Validated DataFrame
        """
        df = self.load_data()
        self.validate_data(df)
        return df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Save the processed data to a file.
        
        Args:
            df: DataFrame to save
            output_path: Path to save the data (optional)
        """
        try:
            if output_path is None:
                output_path = self.config["data"]["processed_path"]
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_path, index=False)
            self.logger.info(f"Data saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")
            raise DataIngestionError(f"Failed to save data: {str(e)}")


if __name__ == "__main__":
    # Quick test
    ingestor = DataIngestor()
    df = ingestor.ingest_and_validate()
    print(f"Data loaded successfully: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())