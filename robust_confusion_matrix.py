#!/usr/bin/env python3
"""
Model File Diagnosis and Confusion Matrix Generator
==================================================

This script diagnoses the model file format and creates confusion matrices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def try_load_model(filepath):
    """Try to load a model using different methods."""
    print(f"Trying to load: {filepath}")
    
    # Method 1: Try pickle
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Successfully loaded with pickle: {type(model)}")
        return model, "pickle"
    except Exception as e:
        print(f"✗ Pickle failed: {e}")
    
    # Method 2: Try joblib
    try:
        model = joblib.load(filepath)
        print(f"✓ Successfully loaded with joblib: {type(model)}")
        return model, "joblib"
    except Exception as e:
        print(f"✗ Joblib failed: {e}")
    
    # Method 3: Try with different pickle protocols
    for protocol in [0, 1, 2, 3, 4, 5]:
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Successfully loaded with pickle protocol {protocol}: {type(model)}")
            return model, f"pickle_protocol_{protocol}"
        except:
            continue
    
    print(f"✗ Could not load {filepath} with any method")
    return None, None

def load_models_and_scaler():
    """Load all available models and scaler."""
    models = {}
    scaler = None
    
    # Try to load models
    model_files = {
        'Random Forest': 'diabetes_model_random_forest.pkl',
        'Logistic Regression': 'diabetes_model.pkl'
    }
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            model, method = try_load_model(filepath)
            if model is not None:
                models[model_name] = model
                print(f"✓ Loaded {model_name} using {method}")
    
    # Try to load scaler
    if os.path.exists('diabetes_scaler.pkl'):
        scaler, method = try_load_model('diabetes_scaler.pkl')
        if scaler is not None:
            print(f"✓ Loaded scaler using {method}")
    
    return models, scaler

def retrain_simple_model():
    """Retrain a simple model if no models can be loaded."""
    print("No models could be loaded. Training a new Random Forest model...")
    
    # Load data
    try:
        df = pd.read_csv('diabetes.csv')
        print(f"✓ Loaded dataset: {df.shape}")
    except FileNotFoundError:
        print("✗ diabetes.csv not found")
        return None, None, None
    
    # Prepare data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Test accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ New model trained with accuracy: {accuracy:.4f}")
    
    return model, scaler, X_test_scaled, y_test

def create_confusion_matrix_from_model(model, scaler, model_name="Model"):
    """Create confusion matrix using the loaded or trained model."""
    
    # Load and prepare test data
    try:
        df = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        print("✗ diabetes.csv not found")
        return None
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Use the same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale test data
    if scaler is not None:
        try:
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            print(f"Error scaling with loaded scaler: {e}")
            print("Creating new scaler...")
            new_scaler = StandardScaler()
            new_scaler.fit(X_train)
            X_test_scaled = new_scaler.transform(X_test)
    else:
        print("No scaler available, using raw features")
        X_test_scaled = X_test
    
    # Make predictions
    try:
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Confusion Matrix with counts
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f'{model_name}\nConfusion Matrix (Counts)\nAccuracy: {accuracy:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot 2: Normalized Confusion Matrix
    plt.subplot(1, 2, 2)
    cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f'{model_name}\nNormalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print results
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{model_name} - Confusion Matrix Results")
    print("="*50)
    print(f"Total Test Samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"Actual    No Diabetes  Diabetes")
    print(f"No Diabetes    {tn:3d}       {fp:3d}")
    print(f"Diabetes       {fn:3d}       {tp:3d}")
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"True Negatives:  {tn} (Correctly identified no diabetes)")
    print(f"False Positives: {fp} (Incorrectly predicted diabetes)")
    print(f"False Negatives: {fn} (Missed diabetes cases)")
    print(f"True Positives:  {tp} (Correctly identified diabetes)")
    
    print(f"\nCalculated Metrics:")
    print(f"Precision (of diabetes predictions): {precision:.4f}")
    print(f"Recall (diabetes detection rate):    {recall:.4f}")
    print(f"Specificity (no diabetes accuracy):  {specificity:.4f}")
    print(f"F1-Score:                            {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
    
    return cm

def main():
    """Main function."""
    print("Diabetes Model Confusion Matrix Generator")
    print("="*50)
    
    # Try to load existing models
    models, scaler = load_models_and_scaler()
    
    if models:
        print(f"\n✓ Found {len(models)} trained model(s)")
        for model_name, model in models.items():
            print(f"\nGenerating confusion matrix for {model_name}...")
            cm = create_confusion_matrix_from_model(model, scaler, model_name)
            if cm is not None:
                print(f"✓ Confusion matrix created for {model_name}")
    else:
        print("\n⚠️ No existing models could be loaded.")
        print("Training a new model...")
        
        model, scaler, X_test_scaled, y_test = retrain_simple_model()
        if model is not None:
            print("\nGenerating confusion matrix for newly trained model...")
            cm = create_confusion_matrix_from_model(model, scaler, "Random Forest (New)")
            if cm is not None:
                print("✓ Confusion matrix created for new model")
                
                # Save the new model
                try:
                    joblib.dump(model, 'diabetes_model_new.pkl')
                    joblib.dump(scaler, 'diabetes_scaler_new.pkl')
                    print("✓ New model and scaler saved with joblib")
                except Exception as e:
                    print(f"⚠️ Could not save new model: {e}")

if __name__ == "__main__":
    main()
