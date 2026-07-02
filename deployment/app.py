"""
Flask API for customer churn prediction with File Upload & Data Generation.
OPTIMIZED VERSION - Fast startup with lazy loading
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import joblib
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, List
import tempfile
import traceback

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.utils.exceptions import ModelPredictionError
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.data.generator import DataGenerator
from src.utils.file_handler import FileHandler
from werkzeug.utils import secure_filename

# Setup logging
logger = setup_logger('churn_api', log_file='logs/api.log')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "saved_models", "preprocessor.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "saved_models", "label_encoder.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "saved_models", "feature_importance.json")

# Initialize File Handler & Data Generator
file_handler = FileHandler()
data_generator = DataGenerator()

# Global variables for loaded models (lazy loaded)
model = None
preprocessor = None
label_encoder = None
feature_importance = None
_models_loaded = False


def load_models():
    """Load all required models and artifacts (lazy loading)."""
    global model, preprocessor, label_encoder, feature_importance, _models_loaded
    
    if _models_loaded:
        return
    
    try:
        start_time = time.time()
        logger.info("Loading models...")
        
        # Load model
        if Path(MODEL_PATH).exists():
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
        else:
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        # Load preprocessor
        if Path(PREPROCESSOR_PATH).exists():
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info("Preprocessor loaded")
        else:
            logger.error(f"Preprocessor file not found: {PREPROCESSOR_PATH}")
            raise FileNotFoundError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")
        
        # Load label encoder
        if Path(LABEL_ENCODER_PATH).exists():
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
            logger.info("Label encoder loaded")
        else:
            logger.warning(f"Label encoder not found at {LABEL_ENCODER_PATH}")
        
        # Load feature importance (optional)
        try:
            if Path(FEATURE_IMPORTANCE_PATH).exists():
                with open(FEATURE_IMPORTANCE_PATH, 'r') as f:
                    feature_importance = json.load(f)
                logger.info("Feature importance loaded")
            else:
                logger.info("Feature importance file not found, continuing without it")
        except Exception as e:
            logger.warning(f"Could not load feature importance: {str(e)}")
        
        _models_loaded = True
        logger.info(f"All models loaded in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def ensure_models_loaded():
    """Ensure models are loaded before prediction."""
    if not _models_loaded:
        load_models()


# ============================================
# PREPROCESSING FUNCTIONS
# ============================================

def preprocess_input(data: Dict[str, Any]) -> np.ndarray:
    """Preprocess input data for prediction (for SINGLE customer)."""
    try:
        ensure_models_loaded()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    df[col].fillna(0, inplace=True)
        
        # Engineer features with explicit config path
        engineer = FeatureEngineer(config_path=CONFIG_PATH)
        df_engineered = engineer.engineer_features(df)
        
        # Remove target column if present
        if 'Churn' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['Churn'])
        
        # Transform using preprocessor
        X_transformed = preprocessor.transform(df_engineered)
        
        logger.info(f"Input preprocessed: {X_transformed.shape}")
        return X_transformed
        
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        logger.error(traceback.format_exc())
        raise ModelPredictionError(f"Input preprocessing failed: {str(e)}")


def preprocess_batch(df: pd.DataFrame) -> np.ndarray:
    """Preprocess BATCH of data for prediction (OPTIMIZED)."""
    try:
        ensure_models_loaded()
        logger.info(f"Batch preprocessing {len(df)} rows...")
        
        engineer = FeatureEngineer(config_path=CONFIG_PATH)
        df_engineered = engineer.engineer_features(df)
        
        if 'Churn' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['Churn'])
        
        X_transformed = preprocessor.transform(df_engineered)
        logger.info(f"Batch preprocessed: {X_transformed.shape}")
        return X_transformed
        
    except Exception as e:
        logger.error(f"Error in batch preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        raise ModelPredictionError(f"Batch preprocessing failed: {str(e)}")


# ============================================
# PREDICTION FUNCTIONS
# ============================================

def get_predictions_probabilities(X: np.ndarray) -> Dict[str, Any]:
    """
    Get predictions and probabilities from the model.
    
    Args:
        X: Preprocessed feature array (single row)
        
    Returns:
        Dictionary with predictions and probabilities
    """
    try:
        ensure_models_loaded()
        
        # Get prediction
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        
        # Convert to readable format
        if label_encoder:
            try:
                prediction_label = label_encoder.inverse_transform(prediction)[0]
            except:
                prediction_label = "Churn" if prediction[0] == 1 else "No Churn"
        else:
            prediction_label = "Churn" if prediction[0] == 1 else "No Churn"
        
        # Get churn probability (class 1)
        if probability.shape[1] > 1:
            churn_probability = float(probability[0][1])
        else:
            churn_probability = float(probability[0][0])
        
        # If probability > 0.5 but label says "No Churn", override
        if churn_probability > 0.5 and prediction_label == "No Churn":
            prediction_label = "Churn"
        elif churn_probability <= 0.5 and prediction_label == "Churn":
            prediction_label = "No Churn"
        
        return {
            'prediction': prediction_label,
            'churn_probability': churn_probability,
            'probability': probability.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise ModelPredictionError(f"Prediction failed: {str(e)}")


def get_predictions_batch(X: np.ndarray) -> List[Dict]:
    """Get predictions for BATCH of customers (OPTIMIZED)."""
    try:
        ensure_models_loaded()
        logger.info(f"Batch predicting {X.shape[0]} rows...")
        
        # Get all predictions at once
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        results = []
        for i in range(len(predictions)):
            # Get prediction value
            pred_val = int(predictions[i])
            
            # Get probability (class 1 is churn)
            if probabilities.shape[1] > 1:
                churn_probability = float(probabilities[i][1])
            else:
                churn_probability = float(probabilities[i][0])
            
            # Convert to label using label encoder if available
            if label_encoder is not None:
                try:
                    prediction_label = label_encoder.inverse_transform([pred_val])[0]
                except:
                    prediction_label = "Churn" if pred_val == 1 else "No Churn"
            else:
                prediction_label = "Churn" if pred_val == 1 else "No Churn"
            
            # Double-check: if probability > 0.5 but label says "No Churn", override
            if churn_probability > 0.5 and prediction_label == "No Churn":
                prediction_label = "Churn"
            elif churn_probability <= 0.5 and prediction_label == "Churn":
                prediction_label = "No Churn"
            
            results.append({
                'prediction': prediction_label,
                'churn_probability': churn_probability
            })
        
        logger.info(f"Batch predictions completed for {len(results)} rows")
        churn_count = sum(1 for r in results if r.get('prediction') == 'Churn')
        logger.info(f"Churn predictions: {churn_count} out of {len(results)}")
        return results
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise ModelPredictionError(f"Batch prediction failed: {str(e)}")


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_risk_factors(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify key risk factors for churn."""
    risk_factors = []
    
    try:
        if data.get('Contract') == 'Month-to-month':
            risk_factors.append({
                'feature': 'contract_type',
                'importance': 0.35,
                'description': 'Month-to-month contracts have 3x higher churn rate'
            })
        
        tenure = data.get('tenure', 0)
        try:
            tenure = int(tenure) if tenure else 0
        except:
            tenure = 0
            
        if tenure < 6 and tenure > 0:
            risk_factors.append({
                'feature': 'tenure',
                'importance': 0.25,
                'description': f'New customer with only {tenure} months tenure'
            })
        elif tenure < 12 and tenure > 0:
            risk_factors.append({
                'feature': 'tenure',
                'importance': 0.20,
                'description': f'Customer with {tenure} months tenure (higher than average risk)'
            })
        
        if data.get('TechSupport') == 'No' and data.get('InternetService') != 'No':
            risk_factors.append({
                'feature': 'tech_support',
                'importance': 0.20,
                'description': 'No tech support subscription'
            })
        
        if data.get('PaymentMethod') == 'Electronic check':
            risk_factors.append({
                'feature': 'payment_method',
                'importance': 0.15,
                'description': 'Electronic check payment method (highest risk)'
            })
        
        if data.get('OnlineSecurity') == 'No' and data.get('InternetService') != 'No':
            risk_factors.append({
                'feature': 'online_security',
                'importance': 0.10,
                'description': 'No online security subscription'
            })
        
        risk_factors.sort(key=lambda x: x['importance'], reverse=True)
        return risk_factors[:5]
        
    except Exception as e:
        logger.error(f"Error getting risk factors: {str(e)}")
        return []


def get_recommendations(data: Dict[str, Any], risk_factors: List[Dict]) -> List[str]:
    """Generate retention recommendations based on risk factors."""
    recommendations = []
    
    try:
        if any(f['feature'] == 'contract_type' for f in risk_factors):
            recommendations.append("Offer 10% discount to convert to annual contract")
            recommendations.append("Highlight benefits of longer-term contracts")
        
        if any(f['feature'] == 'tenure' for f in risk_factors):
            recommendations.append("Send welcome and onboarding support")
            recommendations.append("Offer loyalty bonus for reaching 1-year milestone")
        
        if any(f['feature'] == 'tech_support' for f in risk_factors):
            recommendations.append("Offer free tech support trial for 3 months")
            recommendations.append("Promote tech support value proposition")
        
        if any(f['feature'] == 'payment_method' for f in risk_factors):
            recommendations.append("Suggest automatic payment methods")
            recommendations.append("Offer payment method change incentive")
        
        if len(recommendations) < 3:
            recommendations.append("Send personalized engagement emails")
            recommendations.append("Offer bundle discounts on additional services")
        
        return recommendations[:5]
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return ["Contact customer for retention"]


def calculate_confidence(probability: float) -> float:
    """
    Calculate confidence score based on probability.
    Higher confidence when probability is far from 0.5.
    """
    try:
        # Confidence is highest when probability is near 0 or 1
        confidence = 1 - abs(probability - 0.5) * 2
        return round(max(0.5, min(0.95, confidence)), 2)
    except:
        return 0.75  # Default confidence


def calculate_business_impact(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate business impact of churn.
    """
    try:
        monthly_charges = data.get('MonthlyCharges', 0)
        tenure = data.get('tenure', 0)
        
        # Ensure numeric values
        try:
            monthly_charges = float(monthly_charges) if monthly_charges else 0
            tenure = int(tenure) if tenure else 0
        except:
            monthly_charges = 0
            tenure = 0
        
        # Estimate remaining lifetime value
        avg_customer_lifetime = 36  # months
        remaining_months = max(0, avg_customer_lifetime - tenure)
        estimated_revenue_loss = monthly_charges * remaining_months * 0.7
        
        # Retention cost (typically 20% of revenue)
        retention_cost = estimated_revenue_loss * 0.2
        
        # Avoid division by zero
        if retention_cost <= 0:
            roi = "0%"
            net_savings = 0
        else:
            roi = f"{((estimated_revenue_loss - retention_cost) / retention_cost * 100):.0f}%"
            net_savings = estimated_revenue_loss - retention_cost
        
        return {
            'estimated_revenue_loss': f"${estimated_revenue_loss:,.2f}",
            'retention_cost': f"${retention_cost:,.2f}",
            'net_savings': f"${net_savings:,.2f}",
            'roi': roi
        }
        
    except Exception as e:
        logger.error(f"Error calculating business impact: {str(e)}")
        return {
            'estimated_revenue_loss': '$0.00',
            'retention_cost': '$0.00',
            'net_savings': '$0.00',
            'roi': '0%'
        }


def get_risk_level(probability: float) -> str:
    """Determine risk level based on churn probability."""
    try:
        if probability >= 0.7:
            return "Critical"
        elif probability >= 0.5:
            return "High"
        elif probability >= 0.3:
            return "Medium"
        else:
            return "Low"
    except:
        return "Medium"


def process_batch_predictions(df: pd.DataFrame) -> List[Dict]:
    """Process batch predictions for uploaded data (OPTIMIZED)."""
    results = []
    
    try:
        logger.info(f"Processing {len(df)} rows...")
        X = preprocess_batch(df)
        pred_results = get_predictions_batch(X)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                customer_id = row.get('customerID', 'unknown')
                pred = pred_results[idx]
                
                # Get prediction
                churn_prob = pred.get('churn_probability', 0)
                prediction_label = pred.get('prediction', 'No Churn')
                
                # If prediction is not set, derive from probability
                if prediction_label == 'Unknown' or not prediction_label:
                    prediction_label = 'Churn' if churn_prob > 0.5 else 'No Churn'
                
                # Get risk factors
                customer_data = row.to_dict()
                risk_factors = get_risk_factors(customer_data)
                
                results.append({
                    'customer_id': customer_id,
                    'churn_probability': churn_prob,
                    'prediction': prediction_label,
                    'risk_level': get_risk_level(churn_prob),
                    'risk_factors': risk_factors[:3]
                })
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                results.append({
                    'customer_id': row.get('customerID', 'unknown'),
                    'churn_probability': 0.5,
                    'prediction': 'Unknown',
                    'risk_level': 'Unknown',
                    'risk_factors': []
                })
        
        # Log sample predictions for debugging
        if results:
            logger.info(f"Sample predictions: {results[:3]}")
        
        churn_count = sum(1 for p in results if p.get('prediction') == 'Churn')
        logger.info(f"Churn count: {churn_count} out of {len(results)}")
        return results
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# ============================================
# API ROUTES
# ============================================

@app.route('/')
def home():
    """Home page."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - FAST (doesn't load models)."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': _models_loaded,
        'models_loading': False
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Make churn prediction for a single customer."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        customer_id = data.get('customerID', 'unknown')
        logger.info(f"Received prediction request for customer: {customer_id}")
        
        # Log the incoming data for debugging
        logger.info(f"Data received: {data}")
        
        # Preprocess input
        X = preprocess_input(data)
        
        # Make prediction
        prediction_result = get_predictions_probabilities(X)
        
        # Get risk factors
        risk_factors = get_risk_factors(data)
        
        # Get recommendations
        recommendations = get_recommendations(data, risk_factors)
        
        # Calculate additional metrics with error handling
        churn_prob = prediction_result.get('churn_probability', 0.5)
        
        try:
            confidence = calculate_confidence(churn_prob)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            confidence = 0.75
        
        try:
            risk_level = get_risk_level(churn_prob)
        except Exception as e:
            logger.error(f"Error calculating risk level: {str(e)}")
            risk_level = "Medium"
        
        try:
            business_impact = calculate_business_impact(data)
        except Exception as e:
            logger.error(f"Error calculating business impact: {str(e)}")
            business_impact = {
                'estimated_revenue_loss': '$0.00',
                'retention_cost': '$0.00',
                'net_savings': '$0.00',
                'roi': '0%'
            }
        
        # Prepare response
        response = {
            'customer_id': customer_id,
            'prediction': prediction_result.get('prediction', 'Unknown'),
            'churn_probability': churn_prob,
            'confidence_score': confidence,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': [
                {
                    'action': rec,
                    'priority': 'High' if i < 2 else 'Medium' if i < 4 else 'Low',
                    'estimated_impact': f"Reduce churn by {25 - i*5}%"
                } for i, rec in enumerate(recommendations)
            ],
            'business_impact': business_impact,
            'customer_segment': data.get('customer_segment', 'Standard'),
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0.0'
        }
        
        logger.info(f"Prediction completed for customer: {customer_id}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'customer_id': data.get('customerID', 'unknown') if data else 'unknown'
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Make churn predictions for multiple customers."""
    try:
        data = request.json
        if not data or 'customers' not in data:
            return jsonify({'error': 'No customer data provided'}), 400
        
        customers = data['customers']
        logger.info(f"Received batch prediction request for {len(customers)} customers")
        
        # Convert to DataFrame
        df = pd.DataFrame(customers)
        results = process_batch_predictions(df)
        
        response = {
            'total_customers': len(customers),
            'predictions': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction completed for {len(customers)} customers")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the deployed model."""
    try:
        ensure_models_loaded()
        info = {
            'model_type': type(model).__name__ if model else 'Not loaded',
            'feature_importance': feature_importance if feature_importance else 'Not available',
            'supported_features': [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
            ],
            'model_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================
# FILE UPLOAD & DATA GENERATION ROUTES
# ============================================

@app.route('/api/preview', methods=['POST'])
def preview_file():
    """Preview uploaded file."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'}), 400
        
        file = request.files['file']
        is_valid, message = file_handler.validate_file(file)
        if not is_valid:
            return jsonify({'success': False, 'message': message}), 400
        
        df, metadata = file_handler.read_file(file)
        is_valid, message = file_handler.validate_dataframe(df)
        if not is_valid:
            return jsonify({'success': False, 'message': message}), 400
        
        preview = df.head(10).to_dict('records')
        if 'Churn' in df.columns:
            churn_rate = df['Churn'].value_counts(normalize=True).get('Yes', 0)
            metadata['churn_rate'] = f"{churn_rate:.1%}"
        
        return jsonify({
            'success': True,
            'preview': preview,
            'metadata': metadata,
            'data': df.to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Error previewing file: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process file for predictions (OPTIMIZED)."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'}), 400
        
        file = request.files['file']
        is_valid, message = file_handler.validate_file(file)
        if not is_valid:
            return jsonify({'success': False, 'message': message}), 400
        
        start_time = time.time()
        df, metadata = file_handler.read_file(file)
        logger.info(f"File read in {time.time() - start_time:.2f}s")
        
        filepath = file_handler.save_uploaded_file(file)
        
        # Process ALL rows at once
        start_time = time.time()
        predictions = process_batch_predictions(df)
        logger.info(f"Batch processing completed in {time.time() - start_time:.2f}s")
        
        # Calculate churn count
        churn_count = sum(1 for p in predictions if p.get('prediction') == 'Churn')
        
        return jsonify({
            'success': True,
            'message': f'File processed successfully',
            'total_customers': len(df),
            'churn_count': churn_count,
            'predictions': predictions,
            'filepath': filepath
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def generate_data():
    """Generate sample data."""
    try:
        data = request.json
        n_samples = data.get('n_samples', 1000)
        
        df = data_generator.generate_data(n_samples=n_samples)
        preview = df.head(10).to_dict('records')
        stats = data_generator.get_data_statistics(df)
        
        return jsonify({
            'success': True,
            'message': f'Generated {n_samples} sample records',
            'preview': preview,
            'metadata': {
                'rows': len(df),
                'columns': len(df.columns),
                'churn_rate': f"{stats['churn_rate'].get('Yes', 0):.1%}",
                'format': 'Sample Data'
            },
            'data': df.to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/download-sample', methods=['GET'])
def download_sample():
    """Download sample data template."""
    try:
        df = data_generator.generate_sample_data(n_samples=100)
        temp_dir = tempfile.mkdtemp()
        filepath = os.path.join(temp_dir, 'sample_churn_data.csv')
        df.to_csv(filepath, index=False)
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name='sample_churn_data.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error downloading sample: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/predict_batch_json', methods=['POST'])
def predict_batch_json():
    """Process batch predictions from JSON data."""
    try:
        data = request.json
        if not data or 'customers' not in data:
            return jsonify({'success': False, 'message': 'No customer data provided'}), 400
        
        customers = data['customers']
        logger.info(f"Received batch prediction request for {len(customers)} customers")
        
        # Convert to DataFrame
        df = pd.DataFrame(customers)
        predictions = process_batch_predictions(df)
        
        return jsonify({
            'success': True,
            'message': f'Processed {len(customers)} customers',
            'total_customers': len(customers),
            'churn_count': sum(1 for p in predictions if p.get('prediction') == 'Churn'),
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # FAST STARTUP - Models load on first request
    logger.info("=" * 60)
    logger.info("🌲 Customer Churn Prediction API Starting...")
    logger.info("=" * 60)
    logger.info(f"📁 Base Directory: {BASE_DIR}")
    logger.info(f"📁 Model Path: {MODEL_PATH}")
    logger.info(f"📁 Preprocessor Path: {PREPROCESSOR_PATH}")
    logger.info("")
    logger.info("⏳ Models will load on FIRST prediction request")
    logger.info("⚡ Startup time: 3-5 seconds")
    logger.info("=" * 60)
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 5000))
    debug = os.getenv('API_DEBUG', 'False').lower() == 'true'
    
    app.run(host=host, port=port, debug=debug)