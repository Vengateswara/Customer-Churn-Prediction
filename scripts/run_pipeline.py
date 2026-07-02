"""
Complete Data Pipeline Runner
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import DataIngestor
from src.data.cleaning import DataCleaner
from src.features.engineering import FeatureEngineer

def run_pipeline():
    """Execute the complete data pipeline."""
    print("=" * 60)
    print("🚀 Starting Customer Churn Pipeline")
    print("=" * 60)
    
    # Step 1: Ingest data
    print("\n📥 Step 1: Data Ingestion")
    ingestor = DataIngestor()
    df = ingestor.ingest_and_validate()
    print(f"   ✅ Loaded {len(df)} records")
    
    # Step 2: Clean data
    print("\n🧹 Step 2: Data Cleaning")
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean_data(df)
    print(f"   ✅ Cleaned {len(df_cleaned)} records")
    
    # Step 3: Feature Engineering
    print("\n🔧 Step 3: Feature Engineering")
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df_cleaned)
    print(f"   ✅ Added {len([col for col in df_engineered.columns if col not in df_cleaned.columns])} new features")
    
    # Step 4: Save processed data
    print("\n💾 Step 4: Saving Processed Data")
    df_engineered.to_csv('data/processed/churn_with_features.csv', index=False)
    print("   ✅ Saved to data/processed/churn_with_features.csv")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    
    return df_engineered

if __name__ == "__main__":
    run_pipeline()