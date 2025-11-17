#!/usr/bin/env python3
"""
Simple Logistic Regression Model for Diabetes Prediction
Using the original diabetes.csv dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data(file_path):
    """Load the diabetes dataset and perform basic exploration"""
    print("Loading diabetes dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print("\nDataset info:")
    print(df.info())
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nTarget variable distribution:")
    print(df['Outcome'].value_counts())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

def preprocess_data(df):
    """Preprocess the data for logistic regression"""
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    print(f"Features: {list(X.columns)}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Check for missing values
    print(f"\nMissing values in features: {X.isnull().sum().sum()}")
    print(f"Missing values in target: {y.isnull().sum()}")
    
    return X, y

def train_logistic_model(X, y):
    """Train a logistic regression model"""
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale the features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix (Test Set):")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
    plt.title('Confusion Matrix - Logistic Regression')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('logistic_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance (coefficients)
    feature_names = X.columns
    coefficients = model.coef_[0]
    
    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=True)
    
    plt.barh(importance_df['Feature'], importance_df['Coefficient'])
    plt.title('Logistic Regression Coefficients (Feature Importance)')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig('logistic_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nFeature Coefficients:")
    for feature, coef in zip(feature_names, coefficients):
        print(f"{feature}: {coef:.4f}")
    
    return model, scaler, X_test, y_test, y_test_pred

def save_model(model, scaler, filename_prefix='diabetes_logistic'):
    """Save the trained model and scaler"""
    model_filename = f"{filename_prefix}_model.pkl"
    scaler_filename = f"{filename_prefix}_scaler.pkl"
    
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    print(f"\nModel saved as: {model_filename}")
    print(f"Scaler saved as: {scaler_filename}")
    
    return model_filename, scaler_filename

def main():
    """Main function to run the complete pipeline"""
    print("="*60)
    print("DIABETES PREDICTION - LOGISTIC REGRESSION MODEL")
    print("="*60)
    
    # Load and explore data
    df = load_and_explore_data('diabetes.csv')
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Train model
    model, scaler, X_test, y_test, y_test_pred = train_logistic_model(X, y)
    
    # Save model
    save_model(model, scaler)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = main()
