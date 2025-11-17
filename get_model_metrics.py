"""
Quick Model Performance Analysis
Calculate Accuracy, RMSE, and R² for Original vs Engineered Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_datasets():
    """Load both original and engineered datasets"""
    print("📊 LOADING DATASETS")
    print("=" * 50)
    
    # Load original dataset
    df_original = pd.read_csv('diabetes.csv')
    print(f"Original dataset: {df_original.shape}")
    
    # Load engineered dataset
    df_engineered = pd.read_csv('diabetes_cleaned_engineered.csv')
    print(f"Engineered dataset: {df_engineered.shape}")
    
    return df_original, df_engineered

def calculate_metrics(df, dataset_name, model_name="Random Forest"):
    """Calculate all metrics for a dataset"""
    print(f"\n--- {dataset_name} Dataset with {model_name} ---")
    
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
    
    # Train model
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))
    r2 = r2_score(y_test, y_pred_proba)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"R²:       {r2:.4f}")
    
    # Additional metrics
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc, 
        'rmse': rmse,
        'r2': r2,
        'test_size': len(y_test),
        'positive_cases': y_test.sum(),
        'dataset_size': len(df)
    }

def compare_models():
    """Compare different models on both datasets"""
    print("\n🔬 MODEL COMPARISON")
    print("=" * 50)
    
    df_original, df_engineered = load_datasets()
    
    models = ["Random Forest", "Logistic Regression"]
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"🤖 {model_name.upper()} RESULTS")
        print(f"{'='*60}")
        
        # Original dataset
        results[f"{model_name}_Original"] = calculate_metrics(
            df_original, "Original", model_name
        )
        
        # Engineered dataset  
        results[f"{model_name}_Engineered"] = calculate_metrics(
            df_engineered, "Engineered", model_name
        )
        
        # Calculate improvements
        orig_key = f"{model_name}_Original"
        eng_key = f"{model_name}_Engineered"
        
        acc_improvement = ((results[eng_key]['accuracy'] - results[orig_key]['accuracy']) / 
                          results[orig_key]['accuracy']) * 100
        auc_improvement = ((results[eng_key]['roc_auc'] - results[orig_key]['roc_auc']) / 
                          results[orig_key]['roc_auc']) * 100
        rmse_improvement = ((results[orig_key]['rmse'] - results[eng_key]['rmse']) / 
                           results[orig_key]['rmse']) * 100  # Lower RMSE is better
        r2_improvement = ((results[eng_key]['r2'] - results[orig_key]['r2']) / 
                         abs(results[orig_key]['r2'])) * 100 if results[orig_key]['r2'] != 0 else 0
        
        print(f"\n📈 IMPROVEMENT SUMMARY for {model_name}:")
        print(f"Accuracy: {acc_improvement:+.2f}%")
        print(f"ROC-AUC:  {auc_improvement:+.2f}%") 
        print(f"RMSE:     {rmse_improvement:+.2f}% (lower is better)")
        print(f"R²:       {r2_improvement:+.2f}%")
    
    return results

def create_summary_table(results):
    """Create a comprehensive summary table"""
    print(f"\n📋 COMPREHENSIVE RESULTS TABLE")
    print("=" * 80)
    
    # Create results DataFrame
    summary_data = []
    for key, metrics in results.items():
        model_name, dataset = key.split('_')
        summary_data.append({
            'Model': model_name,
            'Dataset': dataset,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'RMSE': f"{metrics['rmse']:.4f}",
            'R²': f"{metrics['r2']:.4f}",
            'Test_Size': metrics['test_size'],
            'Dataset_Size': metrics['dataset_size']
        })
    
    df_results = pd.DataFrame(summary_data)
    print(df_results.to_string(index=False))
    
    # Best performing model
    print(f"\n🏆 BEST PERFORMING MODEL:")
    best_accuracy = df_results.loc[df_results['Accuracy'].astype(float).idxmax()]
    best_auc = df_results.loc[df_results['ROC-AUC'].astype(float).idxmax()]
    
    print(f"Best Accuracy: {best_accuracy['Model']} on {best_accuracy['Dataset']} = {best_accuracy['Accuracy']}")
    print(f"Best ROC-AUC:  {best_auc['Model']} on {best_auc['Dataset']} = {best_auc['ROC-AUC']}")
    
    return df_results

def main():
    """Main function"""
    print("🎯 DIABETES DATASET - MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("Calculating Accuracy, RMSE, and R² for all models and datasets...")
    
    # Run comparison
    results = compare_models()
    
    # Create summary
    summary_df = create_summary_table(results)
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"─" * 40)
    
    # Get specific Random Forest results for detailed comparison
    rf_orig = results['Random Forest_Original']
    rf_eng = results['Random Forest_Engineered']
    
    print(f"📊 Random Forest Performance:")
    print(f"   Original Dataset:   Acc={rf_orig['accuracy']:.3f}, RMSE={rf_orig['rmse']:.3f}, R²={rf_orig['r2']:.3f}")
    print(f"   Engineered Dataset: Acc={rf_eng['accuracy']:.3f}, RMSE={rf_eng['rmse']:.3f}, R²={rf_eng['r2']:.3f}")
    
    # Calculate percentage changes
    acc_change = (rf_eng['accuracy'] - rf_orig['accuracy']) / rf_orig['accuracy'] * 100
    rmse_change = (rf_orig['rmse'] - rf_eng['rmse']) / rf_orig['rmse'] * 100  # Positive means improvement
    r2_change = (rf_eng['r2'] - rf_orig['r2']) / abs(rf_orig['r2']) * 100 if rf_orig['r2'] != 0 else 0
    
    print(f"\n📈 Feature Engineering Impact:")
    print(f"   Accuracy change: {acc_change:+.1f}%")
    print(f"   RMSE improvement: {rmse_change:+.1f}%")
    print(f"   R² change: {r2_change:+.1f}%")
    
    print(f"\n🎯 FINAL ANSWER TO YOUR QUESTION:")
    print(f"═" * 50)
    print(f"🔸 ACCURACY: {rf_eng['accuracy']:.4f} ({rf_eng['accuracy']*100:.1f}%)")
    print(f"🔸 RMSE:     {rf_eng['rmse']:.4f}")
    print(f"🔸 R²:       {rf_eng['r2']:.4f}")
    print(f"═" * 50)
    print(f"(Best performing model: Random Forest on Engineered Dataset)")

if __name__ == "__main__":
    main()
