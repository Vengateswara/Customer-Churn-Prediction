"""
Data cleaning module for handling missing values, outliers, and data quality issues.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from scipy import stats
from pathlib import Path
import yaml

from src.utils.exceptions import DataCleaningError
from src.utils.logger import LoggerMixin


class DataCleaner(LoggerMixin):
    """
    Handles data cleaning operations for the churn prediction dataset.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the DataCleaner with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.target_column = self.config["data"]["target_column"]
        self.logger.info("DataCleaner initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise DataCleaningError(f"Failed to load config: {str(e)}")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        self.logger.info("Handling missing values...")
        df_cleaned = df.copy()
        
        # Check for missing values
        missing_before = df_cleaned.isnull().sum()
        if missing_before.any():
            self.logger.info(f"Missing values before cleaning:\n{missing_before[missing_before > 0]}")
            
            # Handle TotalCharges: Convert to numeric and fill missing with median
            if 'TotalCharges' in df_cleaned.columns:
                df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
                median_charges = df_cleaned['TotalCharges'].median()
                df_cleaned['TotalCharges'].fillna(median_charges, inplace=True)
                self.logger.info(f"Filled {df_cleaned['TotalCharges'].isnull().sum()} missing TotalCharges with median {median_charges}")
            
            # For categorical columns, fill with mode
            categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != 'customerID' and df_cleaned[col].isnull().any():
                    mode_value = df_cleaned[col].mode()[0]
                    df_cleaned[col].fillna(mode_value, inplace=True)
                    self.logger.info(f"Filled missing values in {col} with mode: {mode_value}")
            
            # For numerical columns, fill with median
            numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
            for col in numerical_cols:
                if df_cleaned[col].isnull().any():
                    median_value = df_cleaned[col].median()
                    df_cleaned[col].fillna(median_value, inplace=True)
                    self.logger.info(f"Filled missing values in {col} with median: {median_value}")
            
            # Check if any missing values remain
            missing_after = df_cleaned.isnull().sum()
            if missing_after.any():
                self.logger.warning(f"Remaining missing values:\n{missing_after[missing_after > 0]}")
            else:
                self.logger.info("All missing values handled successfully")
        
        return df_cleaned
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate records from the dataset.
        
        Args:
            df: DataFrame with potential duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        self.logger.info("Removing duplicates...")
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            self.logger.info(f"Found {duplicate_count} duplicate records")
            
            # Remove duplicates, keeping first occurrence
            df_cleaned = df.drop_duplicates(keep='first')
            self.logger.info(f"Removed {duplicate_count} duplicates. {len(df_cleaned)} records remain")
            return df_cleaned
        else:
            self.logger.info("No duplicates found")
            return df
    
    def correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct data types of columns.
        
        Args:
            df: DataFrame with potential incorrect data types
            
        Returns:
            DataFrame with corrected data types
        """
        self.logger.info("Correcting data types...")
        df_corrected = df.copy()
        
        # Convert SeniorCitizen to int
        if 'SeniorCitizen' in df_corrected.columns:
            df_corrected['SeniorCitizen'] = df_corrected['SeniorCitizen'].astype(int)
        
        # Convert tenure to int
        if 'tenure' in df_corrected.columns:
            df_corrected['tenure'] = df_corrected['tenure'].astype(int)
        
        # Convert MonthlyCharges to float
        if 'MonthlyCharges' in df_corrected.columns:
            df_corrected['MonthlyCharges'] = df_corrected['MonthlyCharges'].astype(float)
        
        # Convert TotalCharges to float
        if 'TotalCharges' in df_corrected.columns:
            df_corrected['TotalCharges'] = pd.to_numeric(df_corrected['TotalCharges'], errors='coerce')
            if df_corrected['TotalCharges'].isnull().any():
                df_corrected['TotalCharges'].fillna(df_corrected['TotalCharges'].median(), inplace=True)
            df_corrected['TotalCharges'] = df_corrected['TotalCharges'].astype(float)
        
        self.logger.info("Data types corrected")
        return df_corrected
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, List[int]]:
        """
        Detect outliers in numerical columns.
        
        Args:
            df: DataFrame to analyze
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            Dictionary mapping column names to outlier indices
        """
        self.logger.info(f"Detecting outliers using {method} method...")
        outlier_dict = {}
        
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col != 'SeniorCitizen']  # Exclude binary
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_idx = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_idx = df[z_scores > 3].index.tolist()
            
            if outlier_idx:
                outlier_dict[col] = outlier_idx
                self.logger.info(f"Found {len(outlier_idx)} outliers in {col}")
        
        return outlier_dict
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        """
        Handle outliers in numerical columns.
        
        Args:
            df: DataFrame with outliers
            method: Method to handle outliers ('cap', 'drop', or 'winsorize')
            
        Returns:
            DataFrame with outliers handled
        """
        self.logger.info(f"Handling outliers using {method} method...")
        df_handled = df.copy()
        
        numerical_cols = df_handled.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col != 'SeniorCitizen']  # Exclude binary
        
        for col in numerical_cols:
            Q1 = df_handled[col].quantile(0.25)
            Q3 = df_handled[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = df_handled[(df_handled[col] < lower_bound) | (df_handled[col] > upper_bound)].shape[0]
            
            if method == 'cap':
                # Cap outliers to boundaries
                df_handled[col] = df_handled[col].clip(lower=lower_bound, upper=upper_bound)
                self.logger.info(f"Capped {outlier_count} outliers in {col}")
                
            elif method == 'winsorize':
                # Winsorize outliers (replace with boundary values)
                from scipy.stats.mstats import winsorize
                df_handled[col] = winsorize(df_handled[col], limits=[0.05, 0.05])
                self.logger.info(f"Winsorized outliers in {col}")
                
            elif method == 'drop':
                # Drop rows with outliers
                mask = (df_handled[col] >= lower_bound) & (df_handled[col] <= upper_bound)
                df_handled = df_handled[mask]
                self.logger.info(f"Dropped {outlier_count} outliers in {col}")
        
        return df_handled
    
    def clean_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize categorical values.
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            DataFrame with standardized categorical values
        """
        self.logger.info("Cleaning categorical values...")
        df_cleaned = df.copy()
        
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'customerID']
        
        # Standardize common values
        for col in categorical_cols:
            # Strip whitespace
            df_cleaned[col] = df_cleaned[col].str.strip()
            
            # Standardize Yes/No values
            if df_cleaned[col].isin(['Yes', 'No']).any():
                df_cleaned[col] = df_cleaned[col].replace({
                    'yes': 'Yes',
                    'no': 'No',
                    'Y': 'Yes',
                    'N': 'No'
                })
            
            # Standardize contract types
            if col == 'Contract':
                df_cleaned[col] = df_cleaned[col].replace({
                    'Month-to-month': 'Month-to-month',
                    'One year': 'One year',
                    'Two year': 'Two year'
                })
        
        return df_cleaned
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete data cleaning pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned and preprocessed DataFrame
        """
        self.logger.info("Starting data cleaning pipeline...")
        
        # Step 1: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 2: Correct data types
        df = self.correct_data_types(df)
        
        # Step 3: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 4: Clean categorical values
        df = self.clean_categorical_values(df)
        
        # Step 5: Detect and handle outliers
        # First detect outliers
        outlier_dict = self.detect_outliers(df, method='iqr')
        if outlier_dict:
            # Handle outliers using cap method
            df = self.handle_outliers(df, method='cap')
        
        self.logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        return df


if __name__ == "__main__":
    # Test the data cleaning pipeline
    from src.data.ingestion import DataIngestor
    
    # Load data
    ingestor = DataIngestor()
    df = ingestor.ingest_and_validate()
    
    # Clean data
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean_data(df)
    
    print(f"Clean data shape: {df_cleaned.shape}")
    print("\nData types:")
    print(df_cleaned.dtypes)
    print("\nMissing values:")
    print(df_cleaned.isnull().sum())