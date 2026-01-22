"""
Flask Backend API for Rheumatoid Arthritis Prediction
Provides REST API endpoints for ML model predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load model and preprocessors
MODEL_DIR = '../model'
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = 'model'  # Try relative path from project root

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'ra_random_forest_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.joblib'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
    
    # Load feature importance
    feature_importance = pd.read_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'))
    top_features = feature_importance.head(5).to_dict('records')
    
    # Load metrics
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    
    print("✓ Model and preprocessors loaded successfully!")
    print(f"✓ Model accuracy: {metrics['accuracy']:.4f}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please train the model first by running: python notebooks/train_model.py")
    model = None

# Log file for predictions
PREDICTIONS_LOG = 'predictions_log.csv'

def log_prediction(patient_data, prediction_result):
    """Log prediction to CSV file"""
    try:
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'age': patient_data['Age'],
            'gender': patient_data['Gender'],
            'prediction': 'RA Positive' if prediction_result['prediction'] == 1 else 'No RA',
            'probability': prediction_result['probability']['RA_Positive'],
            'risk_level': prediction_result['risk_level']
        }
        
        df = pd.DataFrame([log_entry])
        
        # Append to existing file or create new one
        if os.path.exists(PREDICTIONS_LOG):
            df.to_csv(PREDICTIONS_LOG, mode='a', header=False, index=False)
        else:
            df.to_csv(PREDICTIONS_LOG, mode='w', header=True, index=False)
            
    except Exception as e:
        print(f"Error logging prediction: {e}")

def validate_input(data):
    """Validate input data"""
    required_fields = [
        'Age', 'Gender', 'Morning_Stiffness_Duration', 'Joint_Pain_Score',
        'Swollen_Joint_Count', 'Rheumatoid_Factor', 'Anti_CCP', 'ESR',
        'CRP', 'Fatigue_Score', 'Family_History', 'Smoking_Status'
    ]
    
    errors = []
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing field: {field}")
    
    if errors:
        return False, errors
    
    # Validate ranges
    try:
        age = float(data['Age'])
        if not (18 <= age <= 120):
            errors.append("Age must be between 18 and 120")
        
        morning_stiffness = float(data['Morning_Stiffness_Duration'])
        if not (0 <= morning_stiffness <= 300):
            errors.append("Morning stiffness duration must be between 0 and 300 minutes")
        
        joint_pain = float(data['Joint_Pain_Score'])
        if not (0 <= joint_pain <= 10):
            errors.append("Joint pain score must be between 0 and 10")
        
        swollen_joints = float(data['Swollen_Joint_Count'])
        if not (0 <= swollen_joints <= 28):
            errors.append("Swollen joint count must be between 0 and 28")
        
        fatigue = float(data['Fatigue_Score'])
        if not (0 <= fatigue <= 10):
            errors.append("Fatigue score must be between 0 and 10")
        
        # Validate categorical fields
        if data['Gender'] not in ['Male', 'Female']:
            errors.append("Gender must be 'Male' or 'Female'")
        
        if data['Family_History'] not in ['Yes', 'No']:
            errors.append("Family history must be 'Yes' or 'No'")
        
        if data['Smoking_Status'] not in ['Yes', 'No']:
            errors.append("Smoking status must be 'Yes' or 'No'")
            
    except ValueError as e:
        errors.append(f"Invalid numeric value: {str(e)}")
    
    return len(errors) == 0, errors

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

def get_explanation(patient_data, prediction, probability, top_features_list):
    """Generate explanation for the prediction"""
    risk_level = get_risk_level(probability)
    
    explanation = {
        'risk_level': risk_level,
        'key_factors': [],
        'recommendations': []
    }
    
    # Analyze key factors
    if float(patient_data['Rheumatoid_Factor']) > 20:
        explanation['key_factors'].append("Elevated Rheumatoid Factor detected")
    
    if float(patient_data['Anti_CCP']) > 20:
        explanation['key_factors'].append("Elevated Anti-CCP antibodies detected")
    
    if float(patient_data['ESR']) > 20:
        explanation['key_factors'].append("Elevated ESR (inflammation marker)")
    
    if float(patient_data['CRP']) > 5:
        explanation['key_factors'].append("Elevated CRP (inflammation marker)")
    
    if float(patient_data['Morning_Stiffness_Duration']) > 30:
        explanation['key_factors'].append("Prolonged morning stiffness")
    
    if float(patient_data['Swollen_Joint_Count']) > 3:
        explanation['key_factors'].append("Multiple swollen joints")
    
    if float(patient_data['Joint_Pain_Score']) > 5:
        explanation['key_factors'].append("High joint pain score")
    
    # Add recommendations
    if prediction == 1:
        explanation['recommendations'] = [
            "Consult a rheumatologist for comprehensive evaluation",
            "Consider additional diagnostic tests",
            "Discuss early treatment options to prevent joint damage",
            "Monitor symptoms regularly"
        ]
    else:
        if risk_level == "Medium":
            explanation['recommendations'] = [
                "Continue monitoring symptoms",
                "Maintain a healthy lifestyle",
                "Follow up if symptoms worsen",
                "Consider regular check-ups"
            ]
        else:
            explanation['recommendations'] = [
                "Maintain healthy lifestyle habits",
                "Stay physically active",
                "Report any new symptoms to your doctor"
            ]
    
    # Add top important features
    explanation['top_features'] = top_features_list
    
    return explanation

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Rheumatoid Arthritis Prediction API',
        'version': '1.0',
        'status': 'active' if model is not None else 'model not loaded',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Make prediction (POST)',
            '/model-info': 'Model information',
            '/metrics': 'Model performance metrics'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict RA diagnosis for a patient
    
    Expected JSON input:
    {
        "Age": 45,
        "Gender": "Female",
        "Morning_Stiffness_Duration": 60,
        "Joint_Pain_Score": 7,
        "Swollen_Joint_Count": 8,
        "Rheumatoid_Factor": 85.5,
        "Anti_CCP": 120.3,
        "ESR": 45.2,
        "CRP": 15.8,
        "Fatigue_Score": 8,
        "Family_History": "Yes",
        "Smoking_Status": "No"
    }
    """
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        is_valid, errors = validate_input(data)
        if not is_valid:
            return jsonify({
                'error': 'Invalid input data',
                'details': errors
            }), 400
        
        # Prepare data for prediction
        patient_data = {}
        
        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in data:
                encoded_col = col + '_Encoded'
                patient_data[encoded_col] = le.transform([data[col]])[0]
        
        # Add numeric features
        numeric_features = [
            'Age', 'Morning_Stiffness_Duration', 'Joint_Pain_Score',
            'Swollen_Joint_Count', 'Rheumatoid_Factor', 'Anti_CCP',
            'ESR', 'CRP', 'Fatigue_Score'
        ]
        
        for feature in numeric_features:
            patient_data[feature] = float(data[feature])
        
        # Create DataFrame with correct feature order
        df_input = pd.DataFrame([patient_data])
        df_input = df_input[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(df_input)
        
        # Make prediction
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Prepare result
        result = {
            'prediction': prediction,
            'diagnosis': 'RA Positive' if prediction == 1 else 'No RA',
            'probability': {
                'No_RA': float(probabilities[0]),
                'RA_Positive': float(probabilities[1])
            },
            'risk_level': get_risk_level(probabilities[1]),
            'confidence': float(max(probabilities))
        }
        
        # Add explanation
        explanation = get_explanation(data, prediction, probabilities[1], top_features)
        result['explanation'] = explanation
        
        # Log prediction
        log_prediction(data, result)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/model-info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'Random Forest Classifier',
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'features': feature_names,
        'n_features': len(feature_names),
        'top_features': top_features
    })

@app.route('/metrics')
def get_metrics():
    """Get model performance metrics"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(metrics)

@app.route('/sample-input')
def sample_input():
    """Get sample input format"""
    return jsonify({
        'sample_positive': {
            'Age': 52,
            'Gender': 'Female',
            'Morning_Stiffness_Duration': 90,
            'Joint_Pain_Score': 8,
            'Swollen_Joint_Count': 12,
            'Rheumatoid_Factor': 95.5,
            'Anti_CCP': 150.3,
            'ESR': 55.2,
            'CRP': 25.8,
            'Fatigue_Score': 9,
            'Family_History': 'Yes',
            'Smoking_Status': 'Yes'
        },
        'sample_negative': {
            'Age': 35,
            'Gender': 'Male',
            'Morning_Stiffness_Duration': 10,
            'Joint_Pain_Score': 2,
            'Swollen_Joint_Count': 0,
            'Rheumatoid_Factor': 8.5,
            'Anti_CCP': 5.3,
            'ESR': 10.2,
            'CRP': 2.1,
            'Fatigue_Score': 3,
            'Family_History': 'No',
            'Smoking_Status': 'No'
        }
    })

if __name__ == '__main__':
    print("=" * 70)
    print("RHEUMATOID ARTHRITIS PREDICTION API")
    print("=" * 70)
    print("Starting Flask server...")
    print("API will be available at: http://127.0.0.1:5001")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
