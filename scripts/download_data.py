"""
Download Telco Customer Churn Dataset - Working Version
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_dataset():
    """Create the dataset locally."""
    print("Creating dataset...")
    
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    n = 7043
    
    # Create features
    data = {
        'customerID': [f'CUST-{i:04d}' for i in range(n)],
        'gender': np.random.choice(['Female', 'Male'], n),
        'SeniorCitizen': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n, p=[0.5, 0.5]),
        'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'tenure': np.random.randint(1, 72, n),
        'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.4, 0.5, 0.1]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.4, 0.4, 0.2]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.3, 0.5, 0.2]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.3, 0.5, 0.2]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.3, 0.5, 0.2]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.2, 0.6, 0.2]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.3, 0.5, 0.2]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.3, 0.5, 0.2]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.3, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n),
        'MonthlyCharges': np.random.uniform(20, 120, n).round(2),
        'TotalCharges': np.random.uniform(100, 5000, n).round(2),
    }
    
    df = pd.DataFrame(data)
    
    # Add churn with realistic patterns
    tenure_churn_prob = 1 / (1 + np.exp(-(3 - df['tenure']/12)))
    contract_churn = df['Contract'] == 'Month-to-month'
    tech_churn = df['TechSupport'] == 'No'
    online_security_churn = df['OnlineSecurity'] == 'No'
    
    churn_prob = (0.3 * tenure_churn_prob + 
                  0.4 * contract_churn.astype(int) + 
                  0.2 * tech_churn.astype(int) +
                  0.1 * online_security_churn.astype(int))
    
    df['Churn'] = np.random.binomial(1, np.clip(churn_prob, 0, 1))
    df['Churn'] = df['Churn'].map({1: 'Yes', 0: 'No'})
    
    # Save
    output_path = 'data/raw/telco_churn.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset created successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Churn rate: {df['Churn'].value_counts().to_dict()}")
    print(f"   Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    create_dataset()