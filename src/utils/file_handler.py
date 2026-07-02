"""
File Handler - Validates and processes uploaded files for churn prediction.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json
import csv

from src.utils.logger import LoggerMixin
from src.utils.exceptions import DataValidationError


class FileHandler(LoggerMixin):
    """
    Handles file uploads, validation, and processing.
    """
    
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.json'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    def __init__(self, upload_dir: str = "uploads"):
        """
        Initialize file handler.
        
        Args:
            upload_dir: Directory for uploaded files
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"FileHandler initialized with upload dir: {upload_dir}")
    
    def validate_file(self, file) -> Tuple[bool, str]:
        """
        Validate uploaded file.
        
        Args:
            file: File object from request
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check if file exists
        if not file:
            return False, "No file provided"
        
        # Check filename
        if file.filename == '':
            return False, "No file selected"
        
        # Check file extension
        ext = Path(file.filename).suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"File type {ext} not allowed. Allowed: {self.ALLOWED_EXTENSIONS}"
        
        # Check file size
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > self.MAX_FILE_SIZE:
            return False, f"File too large. Max size: {self.MAX_FILE_SIZE / 1024 / 1024}MB"
        
        return True, "File valid"
    
    def read_file(self, file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Read uploaded file into DataFrame.
        
        Args:
            file: File object from request
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        try:
            ext = Path(file.filename).suffix.lower()
            filename = file.filename
            
            if ext == '.csv':
                df = pd.read_csv(file)
                metadata = {'format': 'CSV'}
            elif ext == '.xlsx':
                df = pd.read_excel(file)
                metadata = {'format': 'Excel'}
            elif ext == '.json':
                data = json.load(file)
                df = pd.DataFrame(data)
                metadata = {'format': 'JSON'}
            else:
                raise ValueError(f"Unsupported format: {ext}")
            
            metadata.update({
                'rows': len(df),
                'columns': len(df.columns),
                'filename': filename
            })
            
            self.logger.info(f"Read file: {filename} - {len(df)} rows, {len(df.columns)} columns")
            return df, metadata
            
        except Exception as e:
            self.logger.error(f"Error reading file: {str(e)}")
            raise DataValidationError(f"Failed to read file: {str(e)}")
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate DataFrame structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check if empty
        if df.empty:
            return False, "DataFrame is empty"
        
        # Check required columns
        required_columns = ['customerID', 'gender', 'tenure', 'Contract', 'Churn']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return False, f"Missing required columns: {missing}"
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(df['tenure']):
            return False, "Tenure must be numeric"
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            self.logger.warning(f"Missing values found: {missing_values[missing_values > 0]}")
        
        return True, "Data valid"
    
    def save_uploaded_file(self, file, filename: Optional[str] = None) -> str:
        """
        Save uploaded file to disk.
        
        Args:
            file: File object from request
            filename: Custom filename (optional)
            
        Returns:
            Saved file path
        """
        if filename is None:
            filename = file.filename
        
        # Create safe filename
        safe_filename = filename.replace(' ', '_')
        filepath = self.upload_dir / safe_filename
        
        # Save file
        file.seek(0)
        file.save(str(filepath))
        self.logger.info(f"File saved: {filepath}")
        
        return str(filepath)
    
    def get_sample_data_structure(self) -> Dict[str, Any]:
        """
        Get expected data structure for templates.
        
        Returns:
            Dictionary with column descriptions
        """
        return {
            'required_columns': [
                {'name': 'customerID', 'type': 'string', 'description': 'Unique customer identifier'},
                {'name': 'gender', 'type': 'string', 'description': 'Male/Female'},
                {'name': 'SeniorCitizen', 'type': 'integer', 'description': '0 or 1'},
                {'name': 'Partner', 'type': 'string', 'description': 'Yes/No'},
                {'name': 'Dependents', 'type': 'string', 'description': 'Yes/No'},
                {'name': 'tenure', 'type': 'integer', 'description': 'Months with company (1-72)'},
                {'name': 'Contract', 'type': 'string', 'description': 'Month-to-month/One year/Two year'},
                {'name': 'PaymentMethod', 'type': 'string', 'description': 'Electronic check/Mailed check/Bank transfer/Credit card'},
                {'name': 'MonthlyCharges', 'type': 'float', 'description': 'Monthly bill amount'},
                {'name': 'TotalCharges', 'type': 'float', 'description': 'Total spent'},
                {'name': 'Churn', 'type': 'string', 'description': 'Yes/No (target variable)'},
            ],
            'optional_columns': [
                {'name': 'PhoneService', 'type': 'string', 'description': 'Yes/No'},
                {'name': 'InternetService', 'type': 'string', 'description': 'DSL/Fiber optic/No'},
                {'name': 'TechSupport', 'type': 'string', 'description': 'Yes/No/No internet service'},
                {'name': 'OnlineSecurity', 'type': 'string', 'description': 'Yes/No/No internet service'},
            ]
        }


if __name__ == "__main__":
    # Test file handler
    handler = FileHandler()
    print("=" * 60)
    print("🧪 Testing File Handler")
    print("=" * 60)
    print(f"✅ FileHandler initialized")
    print(f"📁 Upload directory: {handler.upload_dir}")
    print(f"📋 Allowed extensions: {handler.ALLOWED_EXTENSIONS}")
    print(f"📊 Max file size: {handler.MAX_FILE_SIZE / 1024 / 1024}MB")
    print("\n📋 Sample data structure:")
    print(handler.get_sample_data_structure())