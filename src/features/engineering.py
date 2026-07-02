"""
Feature engineering module for creating business-driven features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
import os

from src.utils.exceptions import FeatureEngineeringError
from src.utils.logger import LoggerMixin


class FeatureEngineer(LoggerMixin):
    """
    Creates business-driven features for churn prediction.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the FeatureEngineer with configuration.
        
        Args:
            config_path: Path to the configuration file. If None, tries to find it.
        """
        if config_path is None:
            config_path = self._find_config_file()
        
        self.config = self._load_config(config_path)
        self.engineered_features = self.config["features"]["engineered_features"]
        self.logger.info("FeatureEngineer initialized")
    
    def _find_config_file(self) -> str:
        """Find the config file in multiple possible locations."""
        possible_paths = [
            "configs/config.yaml",
            "../configs/config.yaml",
            "../../configs/config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "config.yaml"),
        ]
        
        if os.path.basename(os.getcwd()) == "deployment":
            possible_paths.append(os.path.join(os.path.dirname(os.getcwd()), "configs", "config.yaml"))
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"Found config file at: {path}")
                return path
        
        raise FeatureEngineeringError(f"Config file not found. Tried: {possible_paths}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to load config: {str(e)}")
    
    def _ensure_numeric(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Ensure specified columns are numeric."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    df[col].fillna(df[col].median() if not df[col].isnull().all() else 0, inplace=True)
        return df
    
    def create_average_monthly_spend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Average Monthly Spend feature."""
        self.logger.info("Creating Average Monthly Spend feature...")
        df = df.copy()
        
        # Ensure numeric columns
        df = self._ensure_numeric(df, ['TotalCharges', 'MonthlyCharges', 'tenure'])
        
        # Calculate average monthly spend
        if 'tenure' in df.columns and 'TotalCharges' in df.columns:
            df['avg_monthly_spend'] = np.where(
                df['tenure'] > 0,
                df['TotalCharges'] / df['tenure'],
                df['MonthlyCharges']
            )
        else:
            df['avg_monthly_spend'] = df['MonthlyCharges']
        
        # Cap extreme values
        if not df['avg_monthly_spend'].isnull().all():
            df['avg_monthly_spend'] = df['avg_monthly_spend'].clip(
                lower=df['avg_monthly_spend'].quantile(0.01),
                upper=df['avg_monthly_spend'].quantile(0.99)
            )
        
        self.logger.info("Average Monthly Spend feature created")
        return df
    
    def create_customer_lifetime_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Customer Lifetime Value (CLV) estimate."""
        self.logger.info("Creating Customer Lifetime Value estimate...")
        df = df.copy()
        
        df = self._ensure_numeric(df, ['MonthlyCharges', 'tenure'])
        
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            df['clv_estimate'] = df['MonthlyCharges'] * df['tenure'] * 1.2
        else:
            df['clv_estimate'] = df['MonthlyCharges'] * 12
        
        if not df['clv_estimate'].isnull().all():
            df['clv_estimate'] = df['clv_estimate'].clip(
                lower=0,
                upper=df['clv_estimate'].quantile(0.99)
            )
        
        self.logger.info("Customer Lifetime Value estimate created")
        return df
    
    def create_tenure_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tenure groups categorical feature."""
        self.logger.info("Creating Tenure Groups feature...")
        df = df.copy()
        
        if 'tenure' in df.columns:
            df = self._ensure_numeric(df, ['tenure'])
            bins = [0, 6, 12, 24, 48, 72, float('inf')]
            labels = ['0-6 months', '6-12 months', '1-2 years', '2-4 years', '4-6 years', '6+ years']
            
            df['tenure_group'] = pd.cut(
                df['tenure'],
                bins=bins,
                labels=labels,
                right=False
            )
            self.logger.info("Tenure Groups feature created")
        else:
            self.logger.warning("Tenure column not found, skipping tenure groups")
            df['tenure_group'] = 'Unknown'
        
        return df
    
    def create_payment_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Payment Risk Score."""
        self.logger.info("Creating Payment Risk Score feature...")
        df = df.copy()
        
        df['payment_risk_score'] = 0
        
        if 'PaperlessBilling' in df.columns:
            df['payment_risk_score'] += (df['PaperlessBilling'] == 'Yes').astype(int) * 2
        
        if 'PaymentMethod' in df.columns:
            payment_risk = {
                'Electronic check': 3,
                'Mailed check': 2,
                'Bank transfer (automatic)': 1,
                'Credit card (automatic)': 1
            }
            df['payment_risk_score'] += df['PaymentMethod'].map(payment_risk).fillna(1)
        
        max_risk = 5
        df['payment_risk_score'] = (df['payment_risk_score'] / max_risk * 10).round(1)
        
        self.logger.info("Payment Risk Score feature created")
        return df
    
    def create_contract_risk_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Contract Risk Level feature."""
        self.logger.info("Creating Contract Risk Level feature...")
        df = df.copy()
        
        if 'Contract' in df.columns:
            risk_mapping = {
                'Month-to-month': 'High Risk',
                'One year': 'Medium Risk',
                'Two year': 'Low Risk'
            }
            df['contract_risk_level'] = df['Contract'].map(risk_mapping).fillna('Unknown')
            
            risk_score_mapping = {
                'Month-to-month': 3,
                'One year': 2,
                'Two year': 1
            }
            df['contract_risk_score'] = df['Contract'].map(risk_score_mapping).fillna(1)
            
            self.logger.info("Contract Risk Level feature created")
        else:
            df['contract_risk_level'] = 'Unknown'
            df['contract_risk_score'] = 1
        
        return df
    
    def create_service_diversity_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Service Diversity Index."""
        self.logger.info("Creating Service Diversity Index feature...")
        df = df.copy()
        
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        existing_services = [col for col in service_columns if col in df.columns]
        
        if existing_services:
            service_count = 0
            for col in existing_services:
                service_count += ((df[col] != 'No') & (df[col] != 'No internet service')).astype(int)
            
            df['service_diversity_index'] = service_count / len(existing_services)
            self.logger.info("Service Diversity Index feature created")
        else:
            df['service_diversity_index'] = 0
        
        return df
    
    def create_engagement_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Engagement Score."""
        self.logger.info("Creating Engagement Score feature...")
        df = df.copy()
        
        df['engagement_score'] = 0
        
        if 'tenure' in df.columns:
            df = self._ensure_numeric(df, ['tenure'])
            tenure_score = np.where(df['tenure'] < 12, 1,
                          np.where(df['tenure'] < 24, 2,
                          np.where(df['tenure'] < 48, 3, 4)))
            df['engagement_score'] += tenure_score
        
        if 'service_diversity_index' in df.columns:
            df['engagement_score'] += df['service_diversity_index'] * 3
        
        if 'Contract' in df.columns:
            contract_score = np.where(df['Contract'] == 'Month-to-month', 1,
                            np.where(df['Contract'] == 'One year', 2, 3))
            df['engagement_score'] += contract_score
        
        max_score = 4 + 3 + 3
        if max_score > 0:
            df['engagement_score'] = (df['engagement_score'] / max_score * 10).round(1)
        else:
            df['engagement_score'] = 0
        
        self.logger.info("Engagement Score feature created")
        return df
    
    def create_loyalty_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Loyalty Score."""
        self.logger.info("Creating Loyalty Score feature...")
        df = df.copy()
        
        df['loyalty_score'] = 0
        
        if 'tenure' in df.columns:
            df = self._ensure_numeric(df, ['tenure'])
            tenure_score = np.where(df['tenure'] < 12, 1,
                          np.where(df['tenure'] < 24, 2,
                          np.where(df['tenure'] < 48, 3, 4)))
            df['loyalty_score'] += tenure_score
        
        if 'Contract' in df.columns:
            contract_score = np.where(df['Contract'] == 'Month-to-month', 1,
                            np.where(df['Contract'] == 'One year', 2, 3))
            df['loyalty_score'] += contract_score
        
        if 'TechSupport' in df.columns:
            tech_score = (df['TechSupport'] == 'Yes').astype(int) * 2
            df['loyalty_score'] += tech_score
        
        max_score = 4 + 3 + 2
        if max_score > 0:
            df['loyalty_score'] = (df['loyalty_score'] / max_score * 10).round(1)
        else:
            df['loyalty_score'] = 0
        
        self.logger.info("Loyalty Score feature created")
        return df
    
    def create_customer_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Customer Segment based on value and risk."""
        self.logger.info("Creating Customer Segment feature...")
        df = df.copy()
        
        # Ensure CLV exists (without calling other methods that might loop)
        if 'clv_estimate' not in df.columns:
            df = self.create_customer_lifetime_value(df)
        
        # Ensure contract risk score exists (without calling other methods)
        if 'contract_risk_score' not in df.columns:
            if 'Contract' in df.columns:
                risk_score_mapping = {
                    'Month-to-month': 3,
                    'One year': 2,
                    'Two year': 1
                }
                df['contract_risk_score'] = df['Contract'].map(risk_score_mapping).fillna(1)
                df['contract_risk_level'] = df['Contract'].map({
                    'Month-to-month': 'High Risk',
                    'One year': 'Medium Risk',
                    'Two year': 'Low Risk'
                }).fillna('Unknown')
            else:
                df['contract_risk_score'] = 1
                df['contract_risk_level'] = 'Unknown'
        
        # Create segments
        df['customer_segment'] = 'Standard'
        
        clv_high = df['clv_estimate'] > df['clv_estimate'].quantile(0.75)
        high_risk = df['contract_risk_score'] == 3
        
        df.loc[clv_high & high_risk, 'customer_segment'] = 'High Value - High Risk'
        df.loc[clv_high & ~high_risk, 'customer_segment'] = 'High Value - Low Risk'
        df.loc[~clv_high & high_risk, 'customer_segment'] = 'Low Value - High Risk'
        df.loc[~clv_high & ~high_risk, 'customer_segment'] = 'Low Value - Low Risk'
        
        self.logger.info("Customer Segment feature created")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the complete feature engineering pipeline."""
        self.logger.info("Starting feature engineering pipeline...")
        df_engineered = df.copy()
        
        # Ensure all numeric columns are properly typed
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            if col in df_engineered.columns:
                df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
                if df_engineered[col].isnull().any():
                    df_engineered[col].fillna(df_engineered[col].median() if not df_engineered[col].isnull().all() else 0, inplace=True)
        
        # Create all features (order matters for dependencies)
        df_engineered = self.create_average_monthly_spend(df_engineered)
        df_engineered = self.create_customer_lifetime_value(df_engineered)
        df_engineered = self.create_tenure_groups(df_engineered)
        df_engineered = self.create_payment_risk_score(df_engineered)
        df_engineered = self.create_contract_risk_level(df_engineered)
        df_engineered = self.create_service_diversity_index(df_engineered)
        df_engineered = self.create_engagement_score(df_engineered)
        df_engineered = self.create_loyalty_score(df_engineered)
        df_engineered = self.create_customer_segment(df_engineered)
        
        new_features = [col for col in df_engineered.columns if col not in df.columns]
        self.logger.info(f"Feature engineering complete. Added {len(new_features)} features")
        self.logger.info(f"Final data shape: {df_engineered.shape}")
        
        return df_engineered


if __name__ == "__main__":
    # Test the feature engineering pipeline
    print("=" * 60)
    print("🧪 Testing Feature Engineering")
    print("=" * 60)
    
    # Create a small test DataFrame
    test_data = {
        'customerID': ['TEST-001', 'TEST-002'],
        'gender': ['Female', 'Male'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'Yes'],
        'tenure': [24, 48],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['No', 'No phone service'],
        'InternetService': ['Fiber optic', 'DSL'],
        'OnlineSecurity': ['No', 'Yes'],
        'OnlineBackup': ['Yes', 'No'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['No', 'Yes'],
        'StreamingTV': ['Yes', 'No'],
        'StreamingMovies': ['Yes', 'No'],
        'Contract': ['Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Credit card (automatic)'],
        'MonthlyCharges': [75.5, 95.0],
        'TotalCharges': [1500.0, 4560.0],
        'Churn': ['No', 'No']
    }
    
    df = pd.DataFrame(test_data)
    
    print("\n📊 Original data shape:", df.shape)
    
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df)
    
    print(f"\n✅ Engineered data shape: {df_engineered.shape}")
    print("\n📊 New features created:")
    new_features = [col for col in df_engineered.columns if col not in df.columns]
    for feature in new_features:
        print(f"   - {feature}: {df_engineered[feature].dtype}")
    
    print("\n📈 Sample of engineered data:")
    print(df_engineered[['customerID', 'avg_monthly_spend', 'clv_estimate', 'engagement_score', 'customer_segment']])