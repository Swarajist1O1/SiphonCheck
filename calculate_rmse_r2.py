#!/usr/bin/env python3
"""
Calculate RMSE and R-squared for the diabetes prediction model
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

def calculate_regression_metrics():
    print("DIABETES MODEL - RMSE AND R-SQUARED ANALYSIS")
    print("=" * 60)
    
    # Load the data
    df = pd.read_csv('diabetes.csv')
    
    # Preprocess data (same as in training)
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_processed = df.copy()
    
    for col in columns_with_zeros:
        if col in df_processed.columns:
            median_val = df_processed[df_processed[col] != 0][col].median()
            df_processed[col] = df_processed[col].replace(0, median_val)
    
    # Split data
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load the trained model and scaler
    try:
        model = joblib.load('diabetes_model_random_forest.pkl')
        scaler = joblib.load('diabetes_scaler.pkl')
        print("✓ Loaded trained Random Forest model and scaler")
    except FileNotFoundError:
        print("✗ Model files not found. Please run the training script first.")
        return
    
    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions and probabilities
    y_pred_binary = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\nTest set size: {len(y_test)} samples")
    print(f"Actual diabetes cases: {y_test.sum()}")
    print(f"Predicted diabetes cases: {y_pred_binary.sum()}")
    
    # Calculate RMSE and R² for probabilities vs actual binary outcomes
    print(f"\n1. CLASSIFICATION MODEL METRICS (Probabilities vs Binary Outcomes):")
    print("-" * 60)
    
    mse_prob = mean_squared_error(y_test, y_pred_proba)
    rmse_prob = math.sqrt(mse_prob)
    r2_prob = r2_score(y_test, y_pred_proba)
    mae_prob = mean_absolute_error(y_test, y_pred_proba)
    
    print(f"RMSE: {rmse_prob:.6f}")
    print(f"R²:   {r2_prob:.6f}")
    print(f"MSE:  {mse_prob:.6f}")
    print(f"MAE:  {mae_prob:.6f}")
    
    # Calculate RMSE and R² for binary predictions vs actual binary outcomes
    print(f"\n2. CLASSIFICATION MODEL METRICS (Binary Predictions vs Binary Outcomes):")
    print("-" * 60)
    
    mse_binary = mean_squared_error(y_test, y_pred_binary)
    rmse_binary = math.sqrt(mse_binary)
    r2_binary = r2_score(y_test, y_pred_binary)
    mae_binary = mean_absolute_error(y_test, y_pred_binary)
    
    print(f"RMSE: {rmse_binary:.6f}")
    print(f"R²:   {r2_binary:.6f}")
    print(f"MSE:  {mse_binary:.6f}")
    print(f"MAE:  {mae_binary:.6f}")
    
    # For comparison, train a linear regression model
    from sklearn.linear_model import LinearRegression
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_predictions = lr_model.predict(X_test_scaled)
    lr_predictions_clipped = np.clip(lr_predictions, 0, 1)
    
    print(f"\n3. LINEAR REGRESSION MODEL (for comparison):")
    print("-" * 60)
    
    mse_lr = mean_squared_error(y_test, lr_predictions_clipped)
    rmse_lr = math.sqrt(mse_lr)
    r2_lr = r2_score(y_test, lr_predictions_clipped)
    mae_lr = mean_absolute_error(y_test, lr_predictions_clipped)
    
    print(f"RMSE: {rmse_lr:.6f}")
    print(f"R²:   {r2_lr:.6f}")
    print(f"MSE:  {mse_lr:.6f}")
    print(f"MAE:  {mae_lr:.6f}")
    
    # Summary and interpretation
    print(f"\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    print(f"• Lower RMSE = Better predictions (closer to actual values)")
    print(f"• Higher R² = Better model (explains more variance)")
    print(f"• R² can be negative for poor models")
    print(f"")
    print(f"Random Forest (probabilities): RMSE={rmse_prob:.4f}, R²={r2_prob:.4f}")
    print(f"Random Forest (binary):        RMSE={rmse_binary:.4f}, R²={r2_binary:.4f}")
    print(f"Linear Regression:             RMSE={rmse_lr:.4f}, R²={r2_lr:.4f}")
    print(f"")
    print(f"🎯 Best RMSE: {min(rmse_prob, rmse_binary, rmse_lr):.4f}")
    print(f"🎯 Best R²:   {max(r2_prob, r2_binary, r2_lr):.4f}")
    
    print(f"\n" + "="*60)
    print("IMPORTANT NOTE:")
    print("="*60)
    print("For classification problems like diabetes prediction:")
    print("• Use Accuracy, Precision, Recall, F1-score, ROC AUC")
    print("• RMSE and R² are more meaningful for regression problems")
    print("• Here we calculate them for completeness and comparison")
    
    return {
        'rmse_prob': rmse_prob,
        'r2_prob': r2_prob,
        'rmse_binary': rmse_binary,
        'r2_binary': r2_binary,
        'rmse_lr': rmse_lr,
        'r2_lr': r2_lr
    }

if __name__ == "__main__":
    metrics = calculate_regression_metrics()
