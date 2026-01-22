"""
Test script for RA Prediction API
"""

import requests
import json

API_URL = "http://127.0.0.1:5001"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_metrics():
    """Test metrics endpoint"""
    print("\n" + "="*60)
    print("Testing /metrics endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/metrics")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_prediction_positive():
    """Test prediction with RA positive sample"""
    print("\n" + "="*60)
    print("Testing /predict endpoint - RA Positive Sample")
    print("="*60)
    
    patient_data = {
        "Age": 52,
        "Gender": "Female",
        "Morning_Stiffness_Duration": 90,
        "Joint_Pain_Score": 8,
        "Swollen_Joint_Count": 12,
        "Rheumatoid_Factor": 95.5,
        "Anti_CCP": 150.3,
        "ESR": 55.2,
        "CRP": 25.8,
        "Fatigue_Score": 9,
        "Family_History": "Yes",
        "Smoking_Status": "Yes"
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=patient_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"\nResponse:")
    result = response.json()
    print(f"  Diagnosis: {result['diagnosis']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"  RA Probability: {result['probability']['RA_Positive']*100:.1f}%")

def test_prediction_negative():
    """Test prediction with RA negative sample"""
    print("\n" + "="*60)
    print("Testing /predict endpoint - RA Negative Sample")
    print("="*60)
    
    patient_data = {
        "Age": 35,
        "Gender": "Male",
        "Morning_Stiffness_Duration": 10,
        "Joint_Pain_Score": 2,
        "Swollen_Joint_Count": 0,
        "Rheumatoid_Factor": 8.5,
        "Anti_CCP": 5.3,
        "ESR": 10.2,
        "CRP": 2.1,
        "Fatigue_Score": 3,
        "Family_History": "No",
        "Smoking_Status": "No"
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=patient_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"\nResponse:")
    result = response.json()
    print(f"  Diagnosis: {result['diagnosis']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"  No RA Probability: {result['probability']['No_RA']*100:.1f}%")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RA PREDICTION API - TEST SUITE")
    print("="*60)
    
    try:
        test_health()
        test_metrics()
        test_prediction_positive()
        test_prediction_negative()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
