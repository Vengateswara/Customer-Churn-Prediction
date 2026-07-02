"""
Unit tests for the Flask API.
"""

import pytest
import json
import sys
import os

# Add src and deployment to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.app import app


class TestAPI:
    """Test Flask API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_predict_endpoint(self, client):
        """Test prediction endpoint."""
        test_data = {
            'customerID': 'TEST-001',
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 24,
            'MonthlyCharges': 75.5,
            'TotalCharges': 1500.0,
            'Contract': 'Month-to-month',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'TechSupport': 'No',
            'PaymentMethod': 'Electronic check',
            'PaperlessBilling': 'Yes',
            'PhoneService': 'Yes'
        }
        
        response = client.post(
            '/predict',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'prediction' in data
        assert 'churn_probability' in data
        assert 'risk_factors' in data
        assert 'recommendations' in data
    
    def test_predict_missing_data(self, client):
        """Test prediction with missing data."""
        test_data = {
            'customerID': 'TEST-002'
        }
        
        response = client.post(
            '/predict',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        # Should fail due to missing required fields
        assert response.status_code in [400, 500]
    
    def test_batch_predict(self, client):
        """Test batch prediction endpoint."""
        test_data = {
            'customers': [
                {
                    'customerID': 'TEST-001',
                    'gender': 'Female',
                    'SeniorCitizen': 0,
                    'Partner': 'Yes',
                    'Dependents': 'No',
                    'tenure': 24,
                    'MonthlyCharges': 75.5,
                    'TotalCharges': 1500.0,
                    'Contract': 'Month-to-month',
                    'InternetService': 'Fiber optic',
                    'OnlineSecurity': 'No',
                    'TechSupport': 'No',
                    'PaymentMethod': 'Electronic check',
                    'PaperlessBilling': 'Yes',
                    'PhoneService': 'Yes'
                }
            ]
        }
        
        response = client.post(
            '/predict_batch',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'total_customers' in data
        assert data['total_customers'] == 1
        assert 'predictions' in data