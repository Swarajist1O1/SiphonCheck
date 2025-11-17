#!/usr/bin/env python3
"""
Simple test script to debug the diabetes prediction model
"""

import pandas as pd
import joblib
import numpy as np
import os

def test_model():
    print("🔍 Testing Diabetes Prediction Model")
    print("=" * 50)
    
    # Check if files exist
    files_to_check = [
        'diabetes_model.pkl',
        'diabetes_scaler.pkl', 
        'diabetes_noise_synthetic.csv',
        'diabetes_cleaned_engineered.csv',
        'diabetes.csv'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
    
    # Load model
    try:
        print("\n📂 Loading model...")
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('diabetes_scaler.pkl')
        print("✅ Model and scaler loaded successfully")
        
        # Load data to get feature names
        if os.path.exists('diabetes_noise_synthetic.csv'):
            df = pd.read_csv('diabetes_noise_synthetic.csv')
            print(f"📊 Using synthetic dataset: {df.shape}")
        elif os.path.exists('diabetes_cleaned_engineered.csv'):
            df = pd.read_csv('diabetes_cleaned_engineered.csv')
            print(f"📊 Using engineered dataset: {df.shape}")
        else:
            df = pd.read_csv('diabetes.csv')
            print(f"📊 Using original dataset: {df.shape}")
        
        feature_names = [col for col in df.columns if col != 'Outcome']
        print(f"🔍 Features ({len(feature_names)}): {feature_names[:5]}...")
        
        # Test prediction with sample data
        print("\n🧪 Testing prediction with sample data...")
        test_input = {
            'Pregnancies': 2,
            'Glucose': 120,
            'BloodPressure': 70,
            'SkinThickness': 20,
            'Insulin': 80,
            'BMI': 25,
            'DiabetesPedigreeFunction': 0.5,
            'Age': 30
        }
        
        # Create DataFrame
        df_test = pd.DataFrame([test_input])
        print(f"📊 Test input shape: {df_test.shape}")
        
        # Add missing features if needed
        if len(feature_names) > 8:
            print("🔧 Adding engineered features...")
            # Add basic engineered features
            missing_features = [f for f in feature_names if f not in df_test.columns]
            print(f"📋 Missing features: {len(missing_features)}")
            
            for feature in missing_features:
                if 'Category' in feature or 'Group' in feature or 'Risk' in feature:
                    df_test[feature] = 1  # Default category
                elif 'Score' in feature:
                    df_test[feature] = 0.5  # Default score
                elif 'Squared' in feature:
                    base_feature = feature.replace('_Squared', '')
                    if base_feature in df_test.columns:
                        df_test[feature] = df_test[base_feature] ** 2
                    else:
                        df_test[feature] = 100  # Default
                elif 'Interaction' in feature:
                    df_test[feature] = 25 * 30 / 100  # BMI * Age / 100
                elif 'Ratio' in feature:
                    df_test[feature] = 0.1  # Default ratio
                elif 'Log_' in feature:
                    df_test[feature] = 1.0  # Default log value
                else:
                    df_test[feature] = 0.5  # Generic default
        
        # Ensure correct order
        df_test = df_test[feature_names]
        print(f"✅ Final test shape: {df_test.shape}")
        
        # Scale and predict
        test_scaled = scaler.transform(df_test)
        prediction = model.predict(test_scaled)[0]
        probability = model.predict_proba(test_scaled)[0]
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
        print(f"   Probability No Diabetes: {probability[0]:.3f}")
        print(f"   Probability Diabetes: {probability[1]:.3f}")
        print(f"   Confidence: {max(probability):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model()
