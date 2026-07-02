"""
Model Training Runner
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import DataPreprocessor
from src.models.train import ModelTrainer

def train_models():
    """Train all models."""
    print("=" * 60)
    print("🤖 Starting Model Training")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading engineered data...")
    df = pd.read_csv('data/processed/churn_with_features.csv')
    print(f"   ✅ Loaded {len(df)} records with {len(df.columns)} features")
    
    # Preprocess
    print("\n🔧 Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.preprocess_data(df)
    print(f"   ✅ Training set: {len(X_train)} records")
    print(f"   ✅ Test set: {len(X_test)} records")
    
    # Train models
    print("\n🎯 Training models...")
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Show results
    print("\n📊 Model Performance Summary:")
    print("-" * 60)
    print(f"{'Model':<15} {'ROC-AUC':<10} {'F1 Score':<10} {'Accuracy':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"{name:<15} {metrics['roc_auc']:.4f}    {metrics['f1']:.4f}     {metrics['accuracy']:.4f}")
        else:
            print(f"{name:<15} ❌ Error")
    
    # Save models
    print("\n💾 Saving models...")
    trainer.save_models()
    
    print("\n" + "=" * 60)
    print(f"✅ Training completed! Best model: {trainer.best_model_name}")
    print("=" * 60)
    
    return trainer

if __name__ == "__main__":
    train_models()