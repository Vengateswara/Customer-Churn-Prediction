"""
Data preprocessing module for encoding, scaling, and splitting data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from pathlib import Path
import yaml

from src.utils.exceptions import DataCleaningError
from src.utils.logger import LoggerMixin


class DataPreprocessor(LoggerMixin):
    """
    Handles data preprocessing including encoding, scaling, and splitting.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the DataPreprocessor with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.target_column = self.config["data"]["target_column"]
        self.test_size = self.config["data"]["test_size"]
        self.random_state = self.config["data"]["random_state"]
        self.scaling_method = self.config["preprocessing"]["scaling"]
        self.encoding_method = self.config["preprocessing"]["encoding"]
        self.handle_imbalance = self.config["preprocessing"]["handle_imbalance"]
        
        self.label_encoder = LabelEncoder()
        self.scaler = None
        self.feature_names = None
        
        self.logger.info("DataPreprocessor initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise DataCleaningError(f"Failed to load config: {str(e)}")
    
    def _get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get categorical columns from the dataset."""
        categorical_cols = self.config["features"]["categorical"]
        # Filter to columns that exist in the dataframe
        return [col for col in categorical_cols if col in df.columns]
    
    def _get_numerical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numerical columns from the dataset."""
        numerical_cols = self.config["features"]["numerical"]
        return [col for col in numerical_cols if col in df.columns]
    
    def _get_engineered_features(self, df: pd.DataFrame) -> List[str]:
        """Get engineered features from the dataset."""
        engineered_cols = self.config["features"]["engineered_features"]
        # Exclude categorical engineered features
        categorical_engineered = ['tenure_group', 'contract_risk_level', 'customer_segment']
        return [col for col in engineered_cols if col in df.columns and col not in categorical_engineered]
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Splitting data into train and test sets...")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Encode target if needed
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
            self.logger.info(f"Target encoded: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        self.logger.info(f"Train set: {len(X_train)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def create_preprocessing_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        """
        Create a preprocessing pipeline for the data.
        
        Args:
            X_train: Training data
            
        Returns:
            Preprocessing pipeline
        """
        self.logger.info("Creating preprocessing pipeline...")
        
        # Get column types
        categorical_cols = self._get_categorical_columns(X_train)
        numerical_cols = self._get_numerical_columns(X_train)
        engineered_cols = self._get_engineered_features(X_train)
        
        # Include tenure_group in categorical if it exists
        if 'tenure_group' in X_train.columns:
            categorical_cols.append('tenure_group')
        
        if 'contract_risk_level' in X_train.columns:
            categorical_cols.append('contract_risk_level')
        
        if 'customer_segment' in X_train.columns:
            categorical_cols.append('customer_segment')
        
        self.logger.info(f"Categorical columns: {categorical_cols}")
        self.logger.info(f"Numerical columns: {numerical_cols}")
        self.logger.info(f"Engineered columns: {engineered_cols}")
        
        # Create transformers
        transformers = []
        
        # Numerical transformer
        if numerical_cols or engineered_cols:
            all_numerical = numerical_cols + engineered_cols
            num_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', self._get_scaler())
            ])
            transformers.append(('num', num_transformer, all_numerical))
        
        # Categorical transformer
        if categorical_cols:
            cat_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', self._get_encoder())
            ])
            transformers.append(('cat', cat_transformer, categorical_cols))
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        # Fit the preprocessor on training data
        preprocessor.fit(X_train)
        
        self.logger.info("Preprocessing pipeline created successfully")
        return preprocessor
    
    def _get_scaler(self):
        """Get the appropriate scaler based on configuration."""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()
    
    def _get_encoder(self):
        """Get the appropriate encoder based on configuration."""
        if self.encoding_method == 'onehot':
            return OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        elif self.encoding_method == 'label':
            return LabelEncoder()
        else:
            return OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    def create_full_pipeline(self, preprocessor: Pipeline) -> Pipeline:
        """
        Create a full pipeline with preprocessing and optional imbalance handling.
        
        Args:
            preprocessor: Preprocessing pipeline
            
        Returns:
            Full pipeline (including SMOTE if configured)
        """
        if self.handle_imbalance == 'smote':
            self.logger.info("Adding SMOTE to pipeline for class imbalance...")
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=self.random_state))
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor)
            ])
        
        return pipeline
    
    def save_preprocessor(self, preprocessor: Pipeline, save_path: str) -> None:
        """
        Save the preprocessor for use in inference.
        
        Args:
            preprocessor: Preprocessing pipeline
            save_path: Path to save the preprocessor
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(preprocessor, save_path)
            self.logger.info(f"Preprocessor saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save preprocessor: {str(e)}")
            raise DataCleaningError(f"Failed to save preprocessor: {str(e)}")
    
    def load_preprocessor(self, load_path: str) -> Pipeline:
        """
        Load a saved preprocessor.
        
        Args:
            load_path: Path to the saved preprocessor
            
        Returns:
            Loaded preprocessor pipeline
        """
        try:
            preprocessor = joblib.load(load_path)
            self.logger.info(f"Preprocessor loaded from {load_path}")
            return preprocessor
        except Exception as e:
            self.logger.error(f"Failed to load preprocessor: {str(e)}")
            raise DataCleaningError(f"Failed to load preprocessor: {str(e)}")
    
    def save_encoders(self, label_encoder: LabelEncoder, save_path: str) -> None:
        """
        Save the label encoder.
        
        Args:
            label_encoder: Label encoder
            save_path: Path to save the encoder
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(label_encoder, save_path)
            self.logger.info(f"Label encoder saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save label encoder: {str(e)}")
            raise DataCleaningError(f"Failed to save label encoder: {str(e)}")
    
    def load_label_encoder(self, load_path: str) -> LabelEncoder:
        """
        Load a saved label encoder.
        
        Args:
            load_path: Path to the saved label encoder
            
        Returns:
            Loaded label encoder
        """
        try:
            label_encoder = joblib.load(load_path)
            self.logger.info(f"Label encoder loaded from {load_path}")
            return label_encoder
        except Exception as e:
            self.logger.error(f"Failed to load label encoder: {str(e)}")
            raise DataCleaningError(f"Failed to load label encoder: {str(e)}")
    
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline: split, transform, and handle imbalance.
        
        Args:
            df: DataFrame to preprocess
            is_training: Whether this is training data
            
        Returns:
            Tuple of (features, target) after preprocessing
        """
        if is_training:
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(df)
            
            # Create preprocessing pipeline
            preprocessor = self.create_preprocessing_pipeline(X_train)
            
            # Transform data
            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            # Save preprocessor for inference
            self.save_preprocessor(preprocessor, "saved_models/preprocessor.pkl")
            self.save_encoders(self.label_encoder, "saved_models/label_encoder.pkl")
            
            return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor
        else:
            # For inference
            preprocessor = self.load_preprocessor("saved_models/preprocessor.pkl")
            label_encoder = self.load_label_encoder("saved_models/label_encoder.pkl")
            
            X = df
            if self.target_column in X.columns:
                X = X.drop(columns=[self.target_column])
            
            X_transformed = preprocessor.transform(X)
            
            return X_transformed, preprocessor, label_encoder


if __name__ == "__main__":
    # Test the preprocessing pipeline
    from src.data.ingestion import DataIngestor
    from src.data.cleaning import DataCleaner
    
    ingestor = DataIngestor()
    cleaner = DataCleaner()
    
    df = ingestor.ingest_and_validate()
    df = cleaner.clean_data(df)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.preprocess_data(df)
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")