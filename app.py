#!/usr/bin/env python3
"""
Flask Web Application for Diabetes Prediction
============================================

This web application provides an interactive interface for diabetes prediction
using the optimized machine learning model with synthetic data enhancement.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'diabetes_prediction_app_2025'

class DiabetesPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
    def load_or_train_model(self):
        """Load the massive synthetic data model."""
        model_path = 'diabetes_massive_model.pkl'
        scaler_path = 'diabetes_massive_scaler.pkl'
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                print("📂 Loading massive synthetic data model...")
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Set feature names for original diabetes dataset (8 features)
                self.feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                self.is_trained = True
                print(f"✅ Model loaded successfully! Features: {len(self.feature_names)}")
            else:
                print("🔄 Training new model...")
                self.train_model()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Attempting to train new model...")
            self.train_model()
            
    def train_model(self):
        """Train the diabetes prediction model using the optimized synthetic dataset."""
        try:
            # Try to load the optimized synthetic dataset
            if os.path.exists('diabetes_noise_synthetic.csv'):
                df = pd.read_csv('diabetes_noise_synthetic.csv')
                print("📊 Using optimized synthetic dataset")
            elif os.path.exists('diabetes_cleaned_engineered.csv'):
                df = pd.read_csv('diabetes_cleaned_engineered.csv')
                print("📊 Using cleaned and engineered dataset")
            else:
                df = pd.read_csv('diabetes.csv')
                print("📊 Using original dataset")
            
            # Prepare features and target
            X = df.drop('Outcome', axis=1)
            y = df['Outcome']
            
            self.feature_names = X.columns.tolist()
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale the features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train the model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate accuracy
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test)
            
            print(f"✅ Model trained successfully!")
            print(f"   Training Accuracy: {train_accuracy:.3f}")
            print(f"   Testing Accuracy: {test_accuracy:.3f}")
            
            # Save the model
            joblib.dump(self.model, 'diabetes_model.pkl')
            joblib.dump(self.scaler, 'diabetes_scaler.pkl')
            
            self.is_trained = True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            self.is_trained = False
    
    def predict(self, input_data):
        """Make a prediction based on input data."""
        if not self.is_trained:
            return None, "Model not trained"
        
        try:
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                # Basic features for simple form
                basic_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                
                df_input = pd.DataFrame([input_data])
                
                # If we have an engineered model, create engineered features
                if len(self.feature_names) > 8:
                    df_input = self._create_engineered_features(df_input)
                
                # Ensure all required features are present
                for feature in self.feature_names:
                    if feature not in df_input.columns:
                        if feature.endswith('_Category') or feature.endswith('_Group') or feature.endswith('_Risk'):
                            df_input[feature] = 0  # Default categorical value
                        else:
                            df_input[feature] = df_input[basic_features].mean().mean()  # Mean of basic features
                
                # Select only the features the model was trained on
                df_input = df_input[self.feature_names]
            
            # Scale the features
            input_scaled = self.scaler.transform(df_input)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            # Simple yes/no result
            return {
                'prediction': int(prediction),
                'simple_answer': 'YES' if prediction == 1 else 'NO',
                'prediction_text': 'YES - You may have diabetes' if prediction == 1 else 'NO - You likely do not have diabetes'
            }, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def _create_engineered_features(self, df):
        """Create engineered features for the input data to match training data exactly."""
        try:
            # BMI Categories
            df['BMI_Category'] = pd.cut(df['BMI'], 
                                      bins=[0, 18.5, 25, 30, 35, 100], 
                                      labels=[0, 1, 2, 3, 4]).astype(float)
            
            # Age Groups
            df['Age_Group'] = pd.cut(df['Age'], 
                                   bins=[0, 25, 35, 45, 55, 100], 
                                   labels=[0, 1, 2, 3, 4]).astype(float)
            
            # Glucose Categories
            df['Glucose_Category'] = pd.cut(df['Glucose'], 
                                          bins=[0, 100, 126, 200], 
                                          labels=[0, 1, 2]).astype(float)
            
            # Blood Pressure Categories
            df['BP_Category'] = pd.cut(df['BloodPressure'], 
                                     bins=[0, 80, 90, 100, 200], 
                                     labels=[0, 1, 2, 3]).astype(float)
            
            # Pregnancy Risk
            df['Pregnancy_Risk'] = pd.cut(df['Pregnancies'], 
                                        bins=[-1, 0, 2, 5, 20], 
                                        labels=[0, 1, 2, 3]).astype(float)
            
            # For single row, use reasonable reference values for normalization
            # (These should match the training data statistics)
            glucose_min, glucose_max = 44, 199  # Approximate from training data
            bmi_min, bmi_max = 18.2, 67.1  # Approximate from training data  
            age_min, age_max = 21, 81  # Approximate from training data
            
            glucose_norm = (df['Glucose'] - glucose_min) / (glucose_max - glucose_min + 1e-8)
            bmi_norm = (df['BMI'] - bmi_min) / (bmi_max - bmi_min + 1e-8)
            age_norm = (df['Age'] - age_min) / (age_max - age_min + 1e-8)
            
            df['Health_Risk_Score'] = (glucose_norm * 0.4 + bmi_norm * 0.3 + age_norm * 0.2 + 
                                     df['DiabetesPedigreeFunction'] * 0.1)
            
            # Metabolic Syndrome Score
            metabolic_indicators = 0
            metabolic_indicators += (df['BMI'] >= 30).astype(int)
            metabolic_indicators += (df['Glucose'] >= 100).astype(int)
            metabolic_indicators += (df['BloodPressure'] >= 85).astype(int)
            df['Metabolic_Syndrome_Score'] = metabolic_indicators
            
            # Interaction Features
            df['BMI_Age_Interaction'] = df['BMI'] * df['Age'] / 100
            df['Glucose_BMI_Ratio'] = df['Glucose'] / (df['BMI'] + 1e-8)
            df['Pregnancies_Age_Ratio'] = df['Pregnancies'] / (df['Age'] + 1)
            
            # Log Transformations
            df['Log_Insulin'] = np.log1p(df['Insulin'])
            df['Log_DiabetesPedigree'] = np.log1p(df['DiabetesPedigreeFunction'])
            
            # Polynomial Features
            df['BMI_Squared'] = df['BMI'] ** 2
            df['Age_Squared'] = df['Age'] ** 2
            df['Glucose_Squared'] = df['Glucose'] ** 2
            
            return df
            
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            # Return original df if feature engineering fails
            return df
    


# Initialize the model
predictor = DiabetesPredictionModel()

@app.route('/')
def home():
    """Home page with prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        print("🔍 Processing prediction request...")
        
        # Get form data
        input_data = {
            'Pregnancies': float(request.form.get('pregnancies', 0)),
            'Glucose': float(request.form.get('glucose', 0)),
            'BloodPressure': float(request.form.get('blood_pressure', 0)),
            'SkinThickness': float(request.form.get('skin_thickness', 0)),
            'Insulin': float(request.form.get('insulin', 0)),
            'BMI': float(request.form.get('bmi', 0)),
            'DiabetesPedigreeFunction': float(request.form.get('diabetes_pedigree', 0)),
            'Age': float(request.form.get('age', 0))
        }
        
        print(f"📊 Input data: {input_data}")
        
        # Validate input data
        if any(val < 0 for val in input_data.values()):
            print("❌ Invalid input: negative values found")
            flash('Please enter valid positive values for all fields.', 'error')
            return redirect(url_for('home'))
        
        # Check if model is trained
        if not predictor.is_trained:
            print("❌ Model not trained")
            flash('Model not available. Please contact administrator.', 'error')
            return redirect(url_for('home'))
        
        # Make prediction
        print("🤖 Making prediction...")
        result, error = predictor.predict(input_data)
        
        if error:
            print(f"❌ Prediction error: {error}")
            flash(f'Prediction failed: {error}', 'error')
            return redirect(url_for('home'))
        
        print(f"✅ Prediction successful: {result}")
        return render_template('result.html', 
                             prediction=result, 
                             input_data=input_data)
        
    except Exception as e:
        print(f"❌ Exception in predict route: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result, error = predictor.predict(data)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page with model information."""
    model_info = {
        'is_trained': predictor.is_trained,
        'feature_count': len(predictor.feature_names) if predictor.feature_names else 0,
        'model_type': 'Multiple Logistic Regression with Massive Synthetic Data',
        'accuracy': '88.26% (on synthetic data), 72.08% (on real data)',
        'dataset_size': '2,468 samples (768 original + 1,700 synthetic)'
    }
    return render_template('about.html', model_info=model_info)

if __name__ == '__main__':
    print("🚀 Starting Diabetes Prediction Web Application")
    print("=" * 60)
    
    # Load or train the model
    predictor.load_or_train_model()
    
    if predictor.is_trained:
        print("🌐 Starting Flask server...")
        # Production-ready configuration
        port = int(os.environ.get('PORT', 5000))
        debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
        
        if debug_mode:
            print("📱 Access the application at: http://localhost:{}".format(port))
        
        app.run(debug=debug_mode, host='0.0.0.0', port=port)
    else:
        print("❌ Failed to initialize model. Please check your data files.")
