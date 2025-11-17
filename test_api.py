#!/usr/bin/env python3
"""
Test script for the Diabetes Prediction API
===========================================

This script demonstrates how to use the Flask web API for diabetes prediction.
"""

import requests
import json

def test_api_prediction():
    """Test the API with sample data."""
    
    # API endpoint
    url = "http://localhost:5000/api/predict"
    
    # Sample test cases
    test_cases = [
        {
            "name": "Low Risk Patient",
            "data": {
                "Pregnancies": 1,
                "Glucose": 85,
                "BloodPressure": 66,
                "SkinThickness": 29,
                "Insulin": 0,
                "BMI": 26.6,
                "DiabetesPedigreeFunction": 0.351,
                "Age": 31
            }
        },
        {
            "name": "High Risk Patient",
            "data": {
                "Pregnancies": 8,
                "Glucose": 183,
                "BloodPressure": 64,
                "SkinThickness": 0,
                "Insulin": 0,
                "BMI": 23.3,
                "DiabetesPedigreeFunction": 0.672,
                "Age": 32
            }
        },
        {
            "name": "Moderate Risk Patient",
            "data": {
                "Pregnancies": 2,
                "Glucose": 120,
                "BloodPressure": 70,
                "SkinThickness": 30,
                "Insulin": 100,
                "BMI": 28.5,
                "DiabetesPedigreeFunction": 0.425,
                "Age": 45
            }
        }
    ]
    
    print("🧪 TESTING DIABETES PREDICTION API")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Make API request
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, 
                                   data=json.dumps(test_case['data']), 
                                   headers=headers, 
                                   timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display input data
                print("Input Data:")
                for key, value in test_case['data'].items():
                    print(f"  {key}: {value}")
                
                # Display prediction results
                print(f"\n🎯 Prediction Results:")
                prediction = "DIABETES RISK" if result['prediction'] == 1 else "NO DIABETES"
                print(f"  Prediction: {prediction}")
                print(f"  Confidence: {result['confidence']:.1%}")
                print(f"  No Diabetes Probability: {result['probability_no_diabetes']:.1%}")
                print(f"  Diabetes Probability: {result['probability_diabetes']:.1%}")
                
                # Risk assessment
                if result['confidence'] > 0.8:
                    confidence_level = "HIGH"
                elif result['confidence'] > 0.6:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"
                
                print(f"  Confidence Level: {confidence_level}")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the Flask app is running on localhost:5000")
        except requests.exceptions.Timeout:
            print("❌ Timeout Error: Request took too long")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ API testing completed!")

def display_usage_examples():
    """Display usage examples for the web application."""
    print(f"\n📚 USAGE EXAMPLES")
    print("=" * 60)
    
    print("🌐 Web Interface:")
    print("   Access: http://localhost:5000")
    print("   - Fill out the form with patient health data")
    print("   - Get instant diabetes risk prediction")
    print("   - View detailed probability analysis")
    
    print(f"\n🔌 API Interface:")
    print("   Endpoint: POST http://localhost:5000/api/predict")
    print("   Content-Type: application/json")
    
    print(f"\n📝 Sample API Request:")
    sample_request = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 30,
        "Insulin": 100,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.425,
        "Age": 45
    }
    print(json.dumps(sample_request, indent=2))
    
    print(f"\n📊 Sample API Response:")
    sample_response = {
        "prediction": 0,
        "probability_no_diabetes": 0.73,
        "probability_diabetes": 0.27,
        "confidence": 0.73
    }
    print(json.dumps(sample_response, indent=2))

if __name__ == "__main__":
    print("🚀 DIABETES PREDICTION API TESTER")
    print("=" * 60)
    
    # Display usage information
    display_usage_examples()
    
    # Test the API
    input("\nPress Enter to test the API (make sure Flask app is running)...")
    test_api_prediction()
    
    print(f"\n🎉 Testing completed! You can now use the web application at:")
    print("   🌐 http://localhost:5000")
    print("   📖 http://localhost:5000/about")
