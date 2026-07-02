"""
Data Generator Module - Creates synthetic customer data for churn prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import random
import sys
import os

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import LoggerMixin


class DataGenerator(LoggerMixin):
    """
    Generates synthetic customer churn data with realistic patterns.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.logger.info(f"DataGenerator initialized with seed: {seed}")
    
    def generate_data(
        self, 
        n_samples: int = 7043, 
        save_path: Optional[str] = None,
        churn_rate: float = 0.26
    ) -> pd.DataFrame:
        """
        Generate synthetic customer churn data.
        
        Args:
            n_samples: Number of records to generate
            save_path: Path to save CSV file (optional)
            churn_rate: Target churn rate (0-1)
            
        Returns:
            DataFrame with synthetic data
        """
        self.logger.info(f"Generating {n_samples} synthetic records...")
        
        # Generate base data
        df = self._generate_base_data(n_samples)
        
        # Add churn patterns
        df = self._add_churn_patterns(df, churn_rate)
        
        self.logger.info(f"Generated {len(df)} records with {len(df.columns)} columns")
        self.logger.info(f"Churn distribution: {df['Churn'].value_counts().to_dict()}")
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            self.logger.info(f"Data saved to {save_path}")
        
        return df
    
    def _generate_base_data(self, n: int) -> pd.DataFrame:
        """Generate base customer data with realistic distributions."""
        
        # Contract types with realistic distribution
        contract_types = ['Month-to-month', 'One year', 'Two year']
        contract_weights = [0.55, 0.25, 0.20]
        
        # Internet service types
        internet_types = ['DSL', 'Fiber optic', 'No']
        internet_weights = [0.40, 0.40, 0.20]
        
        # Payment methods
        payment_methods = [
            'Electronic check', 
            'Mailed check', 
            'Bank transfer (automatic)', 
            'Credit card (automatic)'
        ]
        payment_weights = [0.35, 0.25, 0.20, 0.20]
        
        data = {
            'customerID': [f'CUST-{i:04d}' for i in range(n)],
            'gender': np.random.choice(['Female', 'Male'], n, p=[0.5, 0.5]),
            'SeniorCitizen': np.random.choice([0, 1], n, p=[0.80, 0.20]),
            'Partner': np.random.choice(['Yes', 'No'], n, p=[0.50, 0.50]),
            'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70]),
            'tenure': np.random.randint(1, 72, n),
            'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.40, 0.45, 0.15]),
            'InternetService': np.random.choice(internet_types, n, p=internet_weights),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.30, 0.50, 0.20]),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.35, 0.45, 0.20]),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.30, 0.50, 0.20]),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.25, 0.55, 0.20]),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.35, 0.45, 0.20]),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.35, 0.45, 0.20]),
            'Contract': np.random.choice(contract_types, n, p=contract_weights),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.60, 0.40]),
            'PaymentMethod': np.random.choice(payment_methods, n, p=payment_weights),
            'MonthlyCharges': np.random.uniform(20, 120, n).round(2),
            'TotalCharges': np.random.uniform(100, 5000, n).round(2),
        }
        
        return pd.DataFrame(data)
    
    def _add_churn_patterns(self, df: pd.DataFrame, target_rate: float) -> pd.DataFrame:
        """Add realistic churn patterns based on customer features."""
        
        # Calculate churn probability based on features
        churn_prob = np.zeros(len(df))
        
        # Contract type impact (month-to-month = high churn)
        churn_prob += 0.4 * (df['Contract'] == 'Month-to-month').astype(int)
        
        # Tenure impact (new customers churn more)
        churn_prob += 0.25 * (df['tenure'] < 12).astype(int)
        churn_prob += 0.15 * (df['tenure'] < 6).astype(int)
        
        # Tech support impact
        churn_prob += 0.15 * (df['TechSupport'] == 'No').astype(int)
        
        # Online security impact
        churn_prob += 0.10 * (df['OnlineSecurity'] == 'No').astype(int)
        
        # Payment method impact
        churn_prob += 0.10 * (df['PaymentMethod'] == 'Electronic check').astype(int)
        
        # Senior citizen impact
        churn_prob += 0.05 * (df['SeniorCitizen'] == 1).astype(int)
        
        # Normalize and add randomness
        churn_prob = churn_prob / churn_prob.max() * 0.8
        churn_prob *= np.random.uniform(0.5, 1.0, len(df))
        
        # Apply target rate
        threshold = np.percentile(churn_prob, (1 - target_rate) * 100)
        churn = (churn_prob > threshold).astype(int)
        
        df['Churn'] = ['Yes' if c == 1 else 'No' for c in churn]
        
        return df
    
    def generate_sample_data(self, n_samples: int = 10) -> pd.DataFrame:
        """
        Generate small sample data for testing.
        
        Args:
            n_samples: Number of sample records
            
        Returns:
            Sample DataFrame
        """
        return self.generate_data(n_samples=n_samples)
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the generated data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'churn_rate': df['Churn'].value_counts(normalize=True).to_dict(),
            'contract_distribution': df['Contract'].value_counts().to_dict(),
            'gender_distribution': df['gender'].value_counts().to_dict(),
            'avg_tenure': df['tenure'].mean(),
            'avg_monthly_charges': df['MonthlyCharges'].mean(),
            'avg_total_charges': df['TotalCharges'].mean(),
            'features': list(df.columns)
        }
        return stats


if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Test the data generator
    print("=" * 60)
    print("🧪 Testing Data Generator")
    print("=" * 60)
    
    generator = DataGenerator()
    df = generator.generate_data(n_samples=100)
    
    print(f"\n✅ Generated {len(df)} records")
    print(f"📊 Columns: {len(df.columns)}")
    print(f"📈 Churn distribution: {df['Churn'].value_counts().to_dict()}")
    
    print("\n📋 First 5 rows:")
    print(df.head())
    
    print("\n📊 Statistics:")
    stats = generator.get_data_statistics(df)
    for key, value in stats.items():
        if key != 'features':
            print(f"   {key}: {value}")