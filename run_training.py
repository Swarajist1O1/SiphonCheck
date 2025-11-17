#!/usr/bin/env python3
"""
Simple script to run the diabetes model training
"""

from train_diabetes_model import DiabetesPredictor

def main():
    print("Starting Diabetes Prediction Model Training...")
    
    # Initialize and run the predictor
    predictor = DiabetesPredictor('diabetes.csv')
    predictor.run_complete_pipeline()
    
    print("\nTraining completed! Check the generated files:")
    print("- diabetes_eda.png: Exploratory data analysis plots")
    print("- model_evaluation.png: Model evaluation plots")
    print("- feature_importance.png or feature_coefficients.png: Feature analysis")
    print("- Trained model files (.pkl)")

if __name__ == "__main__":
    main()
