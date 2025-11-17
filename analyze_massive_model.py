#!/usr/bin/env python3
"""
Generate Feature Importance and Confusion Matrix for Massive Model
Analyze the performance and characteristics of the massive synthetic data model
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

def analyze_massive_model():
    """Analyze the massive model with feature importance and confusion matrix"""
    print("="*70)
    print("MASSIVE MODEL ANALYSIS - FEATURE IMPORTANCE & CONFUSION MATRIX")
    print("="*70)
    
    # Load the massive model and scaler
    print("Loading massive model...")
    try:
        model = joblib.load('diabetes_massive_model.pkl')
        scaler = joblib.load('diabetes_massive_scaler.pkl')
        print("✅ Model and scaler loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load the massive dataset
    print("\nLoading massive synthetic dataset...")
    df_massive = pd.read_csv('diabetes_massive_synthetic.csv')
    print(f"Dataset shape: {df_massive.shape}")
    print(f"Class distribution:\n{df_massive['Outcome'].value_counts()}")
    
    # Prepare data
    X = df_massive.drop('Outcome', axis=1)
    y = df_massive['Outcome']
    feature_names = X.columns.tolist()
    
    print(f"\nFeatures: {feature_names}")
    
    # Split data (same way as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Feature Importance (Coefficients for Logistic Regression)
    print(f"\n" + "="*50)
    print("FEATURE IMPORTANCE (LOGISTIC REGRESSION COEFFICIENTS)")
    print("="*50)
    
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    
    print(f"Intercept (β₀): {intercept:.4f}")
    print(f"\nFeature Coefficients:")
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(feature_importance.to_string(index=False, float_format='%.4f'))
    
    # Plot Feature Importance
    plt.figure(figsize=(12, 8))
    colors = ['red' if coef < 0 else 'green' for coef in feature_importance['Coefficient']]
    bars = plt.bar(range(len(feature_importance)), feature_importance['Coefficient'], color=colors, alpha=0.7)
    
    plt.title('Massive Model - Feature Coefficients (Importance)', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    plt.xticks(range(len(feature_importance)), feature_importance['Feature'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, coef) in enumerate(zip(bars, feature_importance['Coefficient'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.05),
                f'{coef:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('massive_model_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Feature importance plot saved as: massive_model_feature_importance.png")
    
    # Confusion Matrix
    print(f"\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"Actual    Non-Diabetic  Diabetic")
    print(f"Non-Diabetic    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"Diabetic        {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nDetailed Metrics:")
    print(f"True Negatives (TN):  {tn:4d} - Correctly predicted non-diabetic")
    print(f"False Positives (FP): {fp:4d} - Incorrectly predicted diabetic")
    print(f"False Negatives (FN): {fn:4d} - Incorrectly predicted non-diabetic")
    print(f"True Positives (TP):  {tp:4d} - Correctly predicted diabetic")
    
    # Performance metrics
    sensitivity = tp / (tp + fn)  # Recall for diabetic
    specificity = tn / (tn + fp)  # Recall for non-diabetic
    precision_diabetic = tp / (tp + fp)
    precision_non_diabetic = tn / (tn + fn)
    
    print(f"\nPerformance Metrics:")
    print(f"Sensitivity (Diabetic Recall):     {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"Specificity (Non-Diabetic Recall): {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"Precision (Diabetic):              {precision_diabetic:.4f} ({precision_diabetic*100:.2f}%)")
    print(f"Precision (Non-Diabetic):          {precision_non_diabetic:.4f} ({precision_non_diabetic*100:.2f}%)")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Massive Model - Confusion Matrix\n(Test Set Performance)', fontsize=16, fontweight='bold')
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    
    # Add percentage annotations
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='darkred')
    
    plt.tight_layout()
    plt.savefig('massive_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Confusion matrix plot saved as: massive_model_confusion_matrix.png")
    
    # Classification Report
    print(f"\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))
    
    # Mathematical Model
    print(f"\n" + "="*50)
    print("MATHEMATICAL MODEL")
    print("="*50)
    print(f"Logistic Regression Equation:")
    print(f"P(Diabetes=1|X) = 1 / (1 + e^(-z))")
    print(f"\nwhere z = {intercept:.4f}", end="")
    for feature, coef in zip(feature_names, coefficients):
        sign = "+" if coef >= 0 else ""
        print(f" {sign}{coef:.4f}×{feature}", end="")
    print()
    
    # Test on Original Data
    print(f"\n" + "="*50)
    print("PERFORMANCE ON ORIGINAL DATA")
    print("="*50)
    
    # Load original dataset
    df_original = pd.read_csv('diabetes_cleaned_only.csv')
    X_orig = df_original.drop('Outcome', axis=1)
    y_orig = df_original['Outcome']
    
    # Split original data the same way
    _, X_orig_test, _, y_orig_test = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
    )
    
    # Scale and predict
    X_orig_test_scaled = scaler.transform(X_orig_test)
    y_orig_pred = model.predict(X_orig_test_scaled)
    orig_accuracy = accuracy_score(y_orig_test, y_orig_pred)
    
    print(f"Accuracy on Original Test Data: {orig_accuracy:.4f} ({orig_accuracy*100:.2f}%)")
    print(f"Accuracy on Synthetic Test Data: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Difference: {accuracy - orig_accuracy:.4f} ({((accuracy - orig_accuracy)/orig_accuracy)*100:.2f}% relative)")
    
    # Summary
    print(f"\n" + "="*70)
    print("MASSIVE MODEL SUMMARY")
    print("="*70)
    print(f"Model Type: Multiple Logistic Regression")
    print(f"Training Dataset: {df_massive.shape[0]:,} samples (768 original + {df_massive.shape[0]-768:,} synthetic)")
    print(f"Features: {len(feature_names)}")
    print(f"Test Accuracy (Synthetic Data): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Test Accuracy (Original Data): {orig_accuracy:.4f} ({orig_accuracy*100:.2f}%)")
    print(f"Most Important Feature: {feature_importance.iloc[0]['Feature']} (coef: {feature_importance.iloc[0]['Coefficient']:.4f})")
    print(f"Files Generated:")
    print(f"  - massive_model_feature_importance.png")
    print(f"  - massive_model_confusion_matrix.png")

if __name__ == "__main__":
    analyze_massive_model()
