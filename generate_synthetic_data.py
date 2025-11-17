#!/usr/bin/env python3
"""
Synthetic Data Generation and Model Testing for Diabetes Dataset
==============================================================

This script generates synthetic data to improve model accuracy and tests the results.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def install_required_packages():
    """Install required packages for synthetic data generation."""
    try:
        from imblearn.over_sampling import SMOTE
        print("✓ imbalanced-learn is already installed")
        return True
    except ImportError:
        print("Installing imbalanced-learn...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'imbalanced-learn'])
            print("✓ imbalanced-learn installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install imbalanced-learn: {e}")
            return False

def load_datasets():
    """Load original and engineered datasets."""
    print("📊 LOADING DATASETS")
    print("=" * 50)
    
    # Load original dataset
    df_original = pd.read_csv('diabetes.csv')
    print(f"Original dataset: {df_original.shape}")
    
    # Load engineered dataset
    try:
        df_engineered = pd.read_csv('diabetes_cleaned_engineered.csv')
        print(f"Engineered dataset: {df_engineered.shape}")
        return df_original, df_engineered
    except FileNotFoundError:
        print("❌ Engineered dataset not found. Please run the data engineering pipeline first.")
        return df_original, None

def generate_smote_data(df):
    """Generate synthetic data using SMOTE."""
    print("\n🔬 GENERATING SYNTHETIC DATA USING SMOTE")
    print("=" * 50)
    
    try:
        from imblearn.over_sampling import SMOTE
        
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        print(f"Original class distribution:")
        print(f"  No Diabetes (0): {(y == 0).sum()} samples")
        print(f"  Diabetes (1): {(y == 1).sum()} samples")
        print(f"  Imbalance ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1")
        
        # Apply SMOTE
        minority_class_size = (y == 1).sum()
        k_neighbors = min(5, minority_class_size - 1)
        
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Create new dataframe
        df_synthetic = pd.DataFrame(X_resampled, columns=X.columns)
        df_synthetic['Outcome'] = y_resampled
        
        print(f"\nAfter SMOTE:")
        print(f"  No Diabetes (0): {(y_resampled == 0).sum()} samples")
        print(f"  Diabetes (1): {(y_resampled == 1).sum()} samples")
        print(f"  Total samples: {len(df_synthetic)} (original: {len(df)})")
        print(f"  Synthetic samples added: {len(df_synthetic) - len(df)}")
        
        return df_synthetic
        
    except Exception as e:
        print(f"❌ SMOTE generation failed: {e}")
        return df

def generate_noise_augmentation(df, augmentation_factor=1.5):
    """Generate synthetic data using noise augmentation."""
    print(f"\n🎛️ GENERATING NOISE-AUGMENTED DATA")
    print("=" * 50)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    target_size = int(len(df) * augmentation_factor)
    additional_samples = target_size - len(df)
    
    print(f"Target dataset size: {target_size}")
    print(f"Additional samples to generate: {additional_samples}")
    
    # Generate additional samples using noise injection
    new_samples = []
    new_labels = []
    
    for _ in range(additional_samples):
        # Randomly select a sample as template
        idx = np.random.randint(0, len(df))
        template = X.iloc[idx].copy()
        label = y.iloc[idx]
        
        # Add noise to numerical features
        for col in X.select_dtypes(include=[np.number]).columns:
            # Skip categorical features
            if col not in ['BMI_Category', 'Age_Group', 'Glucose_Category', 'BP_Category', 'Pregnancy_Risk']:
                # Add noise proportional to feature standard deviation
                noise_std = X[col].std() * 0.1  # 10% of std
                noise = np.random.normal(0, noise_std)
                template[col] = template[col] + noise
                
                # Ensure values stay within reasonable bounds
                template[col] = max(0, template[col])  # Non-negative values
                template[col] = min(template[col], X[col].max() * 1.2)  # Upper bound
        
        new_samples.append(template.values)
        new_labels.append(label)
    
    # Create new dataframe with synthetic samples
    new_df_X = pd.DataFrame(new_samples, columns=X.columns)
    new_df_y = pd.Series(new_labels)
    
    # Combine original and synthetic data
    X_combined = pd.concat([X, new_df_X], ignore_index=True)
    y_combined = pd.concat([y, new_df_y], ignore_index=True)
    
    df_augmented = X_combined.copy()
    df_augmented['Outcome'] = y_combined
    
    print(f"✅ Noise augmentation completed:")
    print(f"  Final dataset size: {len(df_augmented)}")
    print(f"  Augmentation: +{((len(df_augmented) - len(df)) / len(df) * 100):.1f}%")
    
    return df_augmented

def evaluate_model_performance(df_original, df_synthetic, dataset_name="Synthetic"):
    """Evaluate and compare model performance."""
    print(f"\n📊 EVALUATING MODEL PERFORMANCE - {dataset_name}")
    print("=" * 60)
    
    results = {}
    
    datasets = [
        ("Original", df_original),
        (dataset_name, df_synthetic)
    ]
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    for dataset_label, df in datasets:
        print(f"\n--- {dataset_label} Dataset ---")
        
        # Prepare data
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        dataset_results = {}
        
        for model_name, model in models.items():
            # Train and evaluate model
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))
            r2 = r2_score(y_test, y_pred_proba)
            
            dataset_results[model_name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"{model_name}:")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"  ROC-AUC:  {roc_auc:.4f}")
            print(f"  RMSE:     {rmse:.4f}")
            print(f"  R²:       {r2:.4f}")
        
        results[dataset_label] = dataset_results
    
    # Calculate improvements
    print(f"\n📈 IMPROVEMENT SUMMARY:")
    print("-" * 40)
    
    for model_name in models.keys():
        orig_acc = results["Original"][model_name]['accuracy']
        synth_acc = results[dataset_name][model_name]['accuracy']
        
        orig_auc = results["Original"][model_name]['roc_auc']
        synth_auc = results[dataset_name][model_name]['roc_auc']
        
        acc_improvement = ((synth_acc - orig_acc) / orig_acc) * 100
        auc_improvement = ((synth_auc - orig_auc) / orig_auc) * 100
        
        print(f"{model_name}:")
        print(f"  Accuracy improvement: {acc_improvement:+.2f}%")
        print(f"  ROC-AUC improvement:  {auc_improvement:+.2f}%")
    
    return results

def create_comparison_visualization(results):
    """Create visualization comparing model performance."""
    print(f"\n📊 CREATING PERFORMANCE COMPARISON PLOTS")
    print("=" * 50)
    
    # Extract data for plotting
    models = list(results['Original'].keys())
    datasets = list(results.keys())
    
    metrics = ['accuracy', 'roc_auc']
    metric_names = ['Accuracy', 'ROC-AUC']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Performance: Original vs Synthetic Data', fontsize=16)
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        x = np.arange(len(models))
        width = 0.35
        
        original_scores = [results['Original'][model][metric] for model in models]
        synthetic_scores = [results[datasets[1]][model][metric] for model in models]
        
        bars1 = ax.bar(x - width/2, original_scores, width, label='Original', alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x + width/2, synthetic_scores, width, label='Synthetic', alpha=0.8, color='lightblue')
        
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('synthetic_data_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison plot saved as 'synthetic_data_comparison.png'")
    plt.show()

def main():
    """Main function to run synthetic data generation and evaluation."""
    print("🎯 SYNTHETIC DATA GENERATION FOR DIABETES PREDICTION")
    print("=" * 65)
    
    # Check and install required packages
    if not install_required_packages():
        print("❌ Cannot proceed without required packages")
        return
    
    # Load datasets
    df_original, df_engineered = load_datasets()
    
    if df_engineered is None:
        print("Using original dataset for synthetic data generation...")
        df_base = df_original
    else:
        print("Using engineered dataset for synthetic data generation...")
        df_base = df_engineered
    
    # Test SMOTE synthetic data generation
    print("\n" + "="*65)
    print("🧬 TESTING SMOTE SYNTHETIC DATA GENERATION")
    print("="*65)
    
    df_smote = generate_smote_data(df_base)
    
    if len(df_smote) > len(df_base):
        # Save SMOTE dataset
        df_smote.to_csv('diabetes_smote_synthetic.csv', index=False)
        print(f"✓ SMOTE synthetic dataset saved as 'diabetes_smote_synthetic.csv'")
        
        # Evaluate SMOTE performance
        smote_results = evaluate_model_performance(df_base, df_smote, "SMOTE Synthetic")
    
    # Test noise augmentation
    print("\n" + "="*65)
    print("🎛️ TESTING NOISE AUGMENTATION")
    print("="*65)
    
    df_noise = generate_noise_augmentation(df_base, augmentation_factor=1.8)
    
    # Save noise-augmented dataset
    df_noise.to_csv('diabetes_noise_synthetic.csv', index=False)
    print(f"✓ Noise-augmented dataset saved as 'diabetes_noise_synthetic.csv'")
    
    # Evaluate noise augmentation performance
    noise_results = evaluate_model_performance(df_base, df_noise, "Noise Augmented")
    
    # Create comparison visualizations
    if len(df_smote) > len(df_base):
        create_comparison_visualization(smote_results)
    
    # Final summary
    print(f"\n🎉 SYNTHETIC DATA GENERATION COMPLETED!")
    print("=" * 50)
    print(f"📁 Files generated:")
    if len(df_smote) > len(df_base):
        print(f"   • diabetes_smote_synthetic.csv")
    print(f"   • diabetes_noise_synthetic.csv")
    print(f"   • synthetic_data_comparison.png")
    
    print(f"\n📊 Best performing approach:")
    if len(df_smote) > len(df_base):
        # Compare best accuracy from both methods
        best_smote_acc = max([smote_results['SMOTE Synthetic'][model]['accuracy'] for model in smote_results['SMOTE Synthetic']])
        best_noise_acc = max([noise_results['Noise Augmented'][model]['accuracy'] for model in noise_results['Noise Augmented']])
        
        if best_smote_acc > best_noise_acc:
            print(f"   🏆 SMOTE Synthetic Data (Best accuracy: {best_smote_acc:.4f})")
        else:
            print(f"   🏆 Noise Augmented Data (Best accuracy: {best_noise_acc:.4f})")
    else:
        best_noise_acc = max([noise_results['Noise Augmented'][model]['accuracy'] for model in noise_results['Noise Augmented']])
        print(f"   🏆 Noise Augmented Data (Best accuracy: {best_noise_acc:.4f})")

if __name__ == "__main__":
    main()
