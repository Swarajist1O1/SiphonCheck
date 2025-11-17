#!/usr/bin/env python3
"""
Quick Confusion Matrix Generator
===============================

A simple script to quickly generate confusion matrices for your trained diabetes models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_model_and_scaler():
    """Load the trained model and scaler."""
    try:
        # Try loading Random Forest first (usually performs better)
        with open('diabetes_model_random_forest.pkl', 'rb') as f:
            model = pickle.load(f)
        model_name = "Random Forest"
        print("✓ Loaded Random Forest model")
    except FileNotFoundError:
        try:
            # Fallback to regular model
            with open('diabetes_model.pkl', 'rb') as f:
                model = pickle.load(f)
            model_name = "Logistic Regression"
            print("✓ Loaded Logistic Regression model")
        except FileNotFoundError:
            print("❌ No trained models found. Please train a model first.")
            return None, None, None
    
    # Load scaler
    try:
        with open('diabetes_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Loaded scaler")
    except FileNotFoundError:
        print("❌ Scaler not found. Creating new scaler.")
        scaler = StandardScaler()
    
    return model, scaler, model_name

def prepare_test_data():
    """Prepare test data for evaluation."""
    # Load the original dataset
    try:
        df = pd.read_csv('diabetes.csv')
        print(f"✓ Loaded dataset with shape: {df.shape}")
    except FileNotFoundError:
        print("❌ diabetes.csv not found.")
        return None, None
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split the data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_test, y_test

def create_confusion_matrix(model, scaler, X_test, y_test, model_name):
    """Create and display confusion matrix."""
    
    # Scale the test data
    if hasattr(scaler, 'n_features_in_'):
        X_test_scaled = scaler.transform(X_test)
    else:
        # If scaler isn't fitted, we need to fit it first
        # Load training data to fit scaler
        df = pd.read_csv('diabetes.csv')
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler.fit(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create confusion matrix
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
    plt.savefig(f'quick_confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed results
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{model_name} - Confusion Matrix Results")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"Actual    No Diabetes  Diabetes")
    print(f"No Diabetes    {tn:3d}       {fp:3d}")
    print(f"Diabetes       {fn:3d}       {tp:3d}")
    
    print(f"\nDetailed Metrics:")
    print(f"True Negatives:  {tn} (Correctly predicted no diabetes)")
    print(f"False Positives: {fp} (Incorrectly predicted diabetes)")
    print(f"False Negatives: {fn} (Missed diabetes cases)")
    print(f"True Positives:  {tp} (Correctly predicted diabetes)")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nCalculated Metrics:")
    print(f"Precision (of diabetes predictions): {precision:.4f}")
    print(f"Recall (diabetes detection rate):    {recall:.4f}")
    print(f"Specificity (no diabetes accuracy):  {specificity:.4f}")
    print(f"F1-Score:                            {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
    
    return cm, accuracy

def main():
    """Main function to generate quick confusion matrix."""
    print("Quick Confusion Matrix Generator")
    print("="*40)
    
    # Load model and scaler
    model, scaler, model_name = load_model_and_scaler()
    if model is None:
        return
    
    # Prepare test data
    X_test, y_test = prepare_test_data()
    if X_test is None:
        return
    
    # Create confusion matrix
    cm, accuracy = create_confusion_matrix(model, scaler, X_test, y_test, model_name)
    
    print(f"\n✓ Confusion matrix saved as 'quick_confusion_matrix_{model_name.lower().replace(' ', '_')}.png'")

if __name__ == "__main__":
    main()
