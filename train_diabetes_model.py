#!/usr/bin/env python3
"""
Diabetes Dataset Training Script
===============================

This script trains a machine learning model to predict diabetes based on various health metrics.
The dataset contains 8 features and 1 target variable (Outcome).

Features:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)

Target:
- Outcome: Class variable (0 or 1) where 1 indicates diabetes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictor:
    def __init__(self, data_path='diabetes.csv'):
        """Initialize the diabetes predictor with data path."""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        
    def load_data(self):
        """Load the diabetes dataset."""
        print("Loading diabetes dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully! Shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        print("\nClass Distribution:")
        print(self.df['Outcome'].value_counts())
        print(f"Diabetes prevalence: {self.df['Outcome'].mean():.2%}")
        
        # Check for missing values
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        # Check for zero values (which might be missing in this dataset)
        print("\nZero values (potential missing data):")
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in self.df.columns:
                zero_count = (self.df[col] == 0).sum()
                print(f"{col}: {zero_count} ({zero_count/len(self.df)*100:.1f}%)")
    
    def visualize_data(self):
        """Create visualizations for data exploration."""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribution of target variable
        axes[0, 0].pie(self.df['Outcome'].value_counts(), labels=['No Diabetes', 'Diabetes'], 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Distribution of Diabetes Outcome')
        
        # Correlation heatmap
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Feature Correlation Matrix')
        
        # Age distribution by outcome
        axes[1, 0].hist([self.df[self.df['Outcome']==0]['Age'], 
                         self.df[self.df['Outcome']==1]['Age']], 
                        bins=20, alpha=0.7, label=['No Diabetes', 'Diabetes'])
        axes[1, 0].set_xlabel('Age')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Age Distribution by Diabetes Outcome')
        axes[1, 0].legend()
        
        # BMI vs Glucose scatter plot
        scatter = axes[1, 1].scatter(self.df['BMI'], self.df['Glucose'], 
                                   c=self.df['Outcome'], cmap='viridis', alpha=0.6)
        axes[1, 1].set_xlabel('BMI')
        axes[1, 1].set_ylabel('Glucose')
        axes[1, 1].set_title('BMI vs Glucose (colored by Outcome)')
        plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('diabetes_eda.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as 'diabetes_eda.png'")
    
    def preprocess_data(self):
        """Preprocess the data for training."""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Handle zero values (replace with median for non-zero values)
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in self.df.columns:
                zero_count = (self.df[col] == 0).sum()
                if zero_count > 0:
                    median_val = self.df[self.df[col] != 0][col].median()
                    self.df[col] = self.df[col].replace(0, median_val)
                    print(f"Replaced {zero_count} zero values in {col} with median: {median_val:.2f}")
        
        # Separate features and target
        X = self.df.drop('Outcome', axis=1)
        y = self.df['Outcome']
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled using StandardScaler")
    
    def train_models(self):
        """Train multiple models and compare their performance."""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for all models
            model.fit(self.X_train_scaled, self.y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
        
        self.models = results
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name} (ROC AUC: {results[best_model_name]['roc_auc']:.4f})")
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model."""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
        
        elif self.best_model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = LogisticRegression(random_state=42)
        
        elif self.best_model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            }
            model = SVC(random_state=42, probability=True)
        
        print(f"Tuning {self.best_model_name}...")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.best_model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    def evaluate_model(self):
        """Evaluate the final model."""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Predictions
        y_pred = self.best_model.predict(self.X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"Final Model: {self.best_model_name}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nEvaluation plots saved as 'model_evaluation.png'")
    
    def feature_importance(self):
        """Display feature importance if available."""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE")
        print("="*50)
        
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = self.df.drop('Outcome', axis=1).columns
            importances = self.best_model.feature_importances_
            
            # Create feature importance dataframe
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Feature Importance:")
            print(feature_imp)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(feature_imp['feature'], feature_imp['importance'])
            plt.xlabel('Importance')
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("\nFeature importance plot saved as 'feature_importance.png'")
        
        elif hasattr(self.best_model, 'coef_'):
            feature_names = self.df.drop('Outcome', axis=1).columns
            coefficients = self.best_model.coef_[0]
            
            # Create coefficient dataframe
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
            print("Feature Coefficients:")
            print(coef_df)
            
            # Plot coefficients
            plt.figure(figsize=(10, 6))
            colors = ['red' if c < 0 else 'blue' for c in coef_df['coefficient']]
            plt.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
            plt.xlabel('Coefficient Value')
            plt.title(f'Feature Coefficients - {self.best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_coefficients.png', dpi=300, bbox_inches='tight')
            print("\nFeature coefficients plot saved as 'feature_coefficients.png'")
    
    def save_model(self):
        """Save the trained model."""
        import joblib
        
        model_filename = f'diabetes_model_{self.best_model_name.lower().replace(" ", "_")}.pkl'
        scaler_filename = 'diabetes_scaler.pkl'
        
        joblib.dump(self.best_model, model_filename)
        joblib.dump(self.scaler, scaler_filename)
        
        print(f"\nModel saved as: {model_filename}")
        print(f"Scaler saved as: {scaler_filename}")
    
    def predict_sample(self, sample_data):
        """Make a prediction on new data."""
        sample_scaled = self.scaler.transform([sample_data])
        prediction = self.best_model.predict(sample_scaled)[0]
        probability = self.best_model.predict_proba(sample_scaled)[0][1]
        
        return prediction, probability
    
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline."""
        print("DIABETES PREDICTION MODEL TRAINING")
        print("="*70)
        
        # Load and explore data
        self.load_data()
        self.explore_data()
        self.visualize_data()
        
        # Preprocess and train
        self.preprocess_data()
        self.train_models()
        self.hyperparameter_tuning()
        
        # Evaluate and save
        self.evaluate_model()
        self.feature_importance()
        self.save_model()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        
        # Example prediction
        print("\nExample prediction:")
        sample = [6, 148, 72, 35, 0, 33.6, 0.627, 50]  # Sample from the dataset
        prediction, probability = self.predict_sample(sample)
        print(f"Sample: {sample}")
        print(f"Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
        print(f"Probability: {probability:.4f}")


def main():
    """Main function to run the diabetes prediction pipeline."""
    # Initialize the predictor
    predictor = DiabetesPredictor()
    
    # Run the complete pipeline
    predictor.run_complete_pipeline()


if __name__ == "__main__":
    main()
