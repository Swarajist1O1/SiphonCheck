#!/usr/bin/env python3
"""
Confusion Matrix Analysis for Diabetes Prediction Models
========================================================

This script creates detailed confusion matrices for all trained models in the project.
It loads the saved models and generates comprehensive visualizations and metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

class ConfusionMatrixAnalyzer:
    def __init__(self):
        """Initialize the confusion matrix analyzer."""
        self.models = {}
        self.scalers = {}
        self.datasets = {}
        self.test_results = {}
        
    def load_models(self):
        """Load all available trained models."""
        model_files = {
            'Logistic Regression': 'diabetes_model.pkl',
            'Random Forest': 'diabetes_model_random_forest.pkl'
        }
        
        print("Loading trained models...")
        for model_name, filename in model_files.items():
            try:
                with open(filename, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"✓ Loaded {model_name} from {filename}")
            except FileNotFoundError:
                print(f"⚠️ Model file {filename} not found")
        
        # Load scaler
        try:
            with open('diabetes_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("✓ Loaded scaler")
        except FileNotFoundError:
            print("⚠️ Scaler file not found, will create new one")
            self.scaler = StandardScaler()
    
    def load_datasets(self):
        """Load all available datasets."""
        dataset_files = {
            'Original': 'diabetes.csv',
            'Cleaned & Engineered': 'diabetes_cleaned_engineered.csv',
            'SMOTE Synthetic': 'diabetes_smote_synthetic.csv',
            'Noise Augmented': 'diabetes_noise_synthetic.csv'
        }
        
        print("\nLoading datasets...")
        for dataset_name, filename in dataset_files.items():
            try:
                df = pd.read_csv(filename)
                self.datasets[dataset_name] = df
                print(f"✓ Loaded {dataset_name}: {df.shape}")
            except FileNotFoundError:
                print(f"⚠️ Dataset {filename} not found")
    
    def prepare_data(self, dataset_name='Original'):
        """Prepare data for model evaluation."""
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not available. Using Original dataset.")
            dataset_name = 'Original'
        
        df = self.datasets[dataset_name].copy()
        
        # Separate features and target
        if 'Outcome' in df.columns:
            X = df.drop('Outcome', axis=1)
            y = df['Outcome']
        else:
            # Assume last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        if hasattr(self.scaler, 'n_features_in_'):
            # Scaler is already fitted
            X_test_scaled = self.scaler.transform(X_test)
        else:
            # Fit and transform
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        return X_test_scaled, y_test
    
    def generate_confusion_matrices(self, dataset_name='Original'):
        """Generate confusion matrices for all models."""
        print(f"\nGenerating confusion matrices for {dataset_name} dataset...")
        
        if not self.models:
            print("No models loaded. Please run load_models() first.")
            return
        
        X_test, y_test = self.prepare_data(dataset_name)
        
        # Calculate number of subplots needed
        n_models = len(self.models)
        if n_models == 0:
            print("No models available for evaluation.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        model_results = {}
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                if y_pred_proba is not None:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                else:
                    roc_auc = None
                
                # Store results
                model_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Create confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Plot confusion matrix with counts
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           ax=axes[0, idx], cbar=False)
                axes[0, idx].set_title(f'{model_name}\nConfusion Matrix (Counts)')
                axes[0, idx].set_xlabel('Predicted')
                axes[0, idx].set_ylabel('Actual')
                
                # Plot normalized confusion matrix
                cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                           ax=axes[1, idx], cbar=False)
                axes[1, idx].set_title(f'{model_name}\nConfusion Matrix (Normalized)')
                axes[1, idx].set_xlabel('Predicted')
                axes[1, idx].set_ylabel('Actual')
                
                # Print detailed metrics
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")
                if roc_auc:
                    print(f"ROC AUC: {roc_auc:.4f}")
                
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                print(f"\nConfusion Matrix:")
                print(cm)
                print(f"True Negatives: {cm[0,0]}")
                print(f"False Positives: {cm[0,1]}")
                print(f"False Negatives: {cm[1,0]}")
                print(f"True Positives: {cm[1,1]}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                # Create empty plot
                axes[0, idx].text(0.5, 0.5, f'Error:\n{model_name}', 
                                ha='center', va='center', transform=axes[0, idx].transAxes)
                axes[1, idx].text(0.5, 0.5, f'Error:\n{model_name}', 
                                ha='center', va='center', transform=axes[1, idx].transAxes)
        
        plt.tight_layout()
        plt.savefig(f'confusion_matrices_{dataset_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store results for this dataset
        self.test_results[dataset_name] = model_results
        
        return model_results
    
    def create_detailed_confusion_matrix(self, model_name, dataset_name='Original'):
        """Create a detailed confusion matrix for a specific model."""
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        X_test, y_test = self.prepare_data(dataset_name)
        model = self.models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Create detailed confusion matrix plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Standard confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
        disp.plot(ax=axes[0], cmap='Blues', values_format='d')
        axes[0].set_title(f'{model_name} - Confusion Matrix\nDataset: {dataset_name}')
        
        # Normalized confusion matrix
        cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['No Diabetes', 'Diabetes'])
        disp_norm.plot(ax=axes[1], cmap='Blues', values_format='.2f')
        axes[1].set_title(f'{model_name} - Normalized Confusion Matrix\nDataset: {dataset_name}')
        
        plt.tight_layout()
        plt.savefig(f'detailed_confusion_matrix_{model_name.lower().replace(" ", "_")}_{dataset_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed analysis
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n{model_name} - Detailed Confusion Matrix Analysis")
        print("="*60)
        print(f"Dataset: {dataset_name}")
        print(f"Total Samples: {len(y_test)}")
        print(f"\nConfusion Matrix Values:")
        print(f"True Negatives (TN):  {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives (TP):  {tp}")
        
        print(f"\nDerived Metrics:")
        print(f"Accuracy:  {(tp + tn) / (tp + tn + fp + fn):.4f}")
        print(f"Precision: {tp / (tp + fp):.4f}")
        print(f"Recall:    {tp / (tp + fn):.4f}")
        print(f"Specificity: {tn / (tn + fp):.4f}")
        print(f"F1-Score:  {2 * tp / (2 * tp + fp + fn):.4f}")
        
        return cm
    
    def compare_all_models(self):
        """Compare confusion matrices across all datasets and models."""
        if not self.test_results:
            print("No test results available. Run generate_confusion_matrices() first.")
            return
        
        # Create comparison summary
        comparison_data = []
        
        for dataset_name, results in self.test_results.items():
            for model_name, metrics in results.items():
                comparison_data.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'ROC AUC': metrics['roc_auc'] if metrics['roc_auc'] else 'N/A'
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\nModel Comparison Summary:")
        print("="*80)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison
        comparison_df.to_csv('model_confusion_matrix_comparison.csv', index=False)
        print(f"\nComparison saved to 'model_confusion_matrix_comparison.csv'")
        
        return comparison_df

def main():
    """Main function to run confusion matrix analysis."""
    print("Diabetes Model Confusion Matrix Analysis")
    print("="*50)
    
    analyzer = ConfusionMatrixAnalyzer()
    
    # Load models and datasets
    analyzer.load_models()
    analyzer.load_datasets()
    
    if not analyzer.models:
        print("No models found. Please ensure model files exist.")
        return
    
    if not analyzer.datasets:
        print("No datasets found. Please ensure data files exist.")
        return
    
    # Generate confusion matrices for each available dataset
    for dataset_name in analyzer.datasets.keys():
        print(f"\n{'='*60}")
        print(f"ANALYZING DATASET: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            analyzer.generate_confusion_matrices(dataset_name)
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {str(e)}")
    
    # Create detailed analysis for best performing model
    if analyzer.test_results:
        # Find best model (highest accuracy on original dataset)
        if 'Original' in analyzer.test_results:
            best_model = max(analyzer.test_results['Original'].items(), 
                           key=lambda x: x[1]['accuracy'])
            print(f"\nCreating detailed analysis for best model: {best_model[0]}")
            analyzer.create_detailed_confusion_matrix(best_model[0], 'Original')
    
    # Generate comparison
    analyzer.compare_all_models()

if __name__ == "__main__":
    main()
