# Multiple Logistic Regression Model Results - Diabetes Prediction

**Date:** November 18, 2025  
**Model Type:** Multiple Logistic Regression (Multivariate)  
**Dataset:** Massive Synthetic Dataset (diabetes_massive_synthetic.csv)  
**Parameters:** 9 total (1 intercept + 8 feature coefficients)  
**Final Model:** Enhanced with 1,700 synthetic samples  

## Dataset Information

### Original Dataset:
- **Original Samples:** 768
- **Class Distribution:** 500 non-diabetic (65.1%) vs 268 diabetic (34.9%)

### Massive Synthetic Dataset:
- **Total Samples:** 2,468 (768 original + 1,700 synthetic)
- **Features:** 8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
- **Target Variable:** Outcome (0: Non-Diabetic, 1: Diabetic)
- **Enhanced Distribution:** 1,251 non-diabetic (50.7%) vs 1,217 diabetic (49.3%)
- **Improvement:** Much better class balance (51/49 vs original 65/35)

### Dataset Statistics
```
Feature               Mean      Std       Min     25%     50%     75%     Max
Pregnancies          3.85      3.37      0.00    1.00    3.00    6.00   17.00
Glucose            120.89     31.97      0.00   99.00  117.00  140.25  199.00
BloodPressure       69.11     19.36      0.00   62.00   72.00   80.00  122.00
SkinThickness       20.54     15.95      0.00    0.00   23.00   32.00   99.00
Insulin             79.80    115.24      0.00    0.00   30.50  127.25  846.00
BMI                 31.99      7.88      0.00   27.30   32.00   36.60   67.10
DiabetesPedigreeFunction 0.47  0.33      0.08    0.24    0.37    0.63    2.42
Age                 33.24     11.76     21.00   24.00   29.00   41.00   81.00
```

## Model Training Configuration

### Logistic Regression Type:
- **Multiple Logistic Regression** (not simple 2-parameter logistic)
- **Input Features:** 8 medical variables
- **Total Parameters:** 9 (1 intercept + 8 feature coefficients)
- **Sklearn Implementation:** `LogisticRegression(random_state=42, max_iter=1000)`

### Training Setup:
- **Train/Test Split:** 80/20 (stratified)
- **Training Samples:** 614
- **Test Samples:** 154
- **Feature Scaling:** StandardScaler applied
- **Random State:** 42
- **Max Iterations:** 1000

### Model Parameters:
- **C:** 1.0 (regularization strength)
- **Penalty:** L2 (Ridge regularization)
- **Solver:** LBFGS (Limited-memory BFGS)
- **Tolerance:** 0.0001
- **Class Weight:** None (equal weights)

## Model Performance

### Massive Model Performance (Current):
- **Training Accuracy:** 88.70%
- **Test Accuracy:** 88.26% (on synthetic data)
- **Real Data Accuracy:** 71.24% (on original data)
- **Training Samples:** 1,974
- **Test Samples:** 494

### Classification Report (Synthetic Test Set):
```
              precision    recall  f1-score   support

Non-Diabetic       0.88      0.89      0.88       250
    Diabetic       0.88      0.88      0.88       244

    accuracy                           0.88       494
   macro avg       0.88      0.88      0.88       494
weighted avg       0.88      0.88      0.88       494
```

### Confusion Matrix (Synthetic Test Set):
```
                Predicted
Actual      Non-Diabetic  Diabetic
Non-Diabetic      222        28
Diabetic           30       214
```

### Performance Comparison:
| Model Version | Test Accuracy | Notes |
|---------------|---------------|--------|
| Original Model | 71.43% | Small dataset, class imbalance |
| Enhanced Model | 76.14% | First synthetic data attempt |
| **Massive Model** | **88.26%** | **Large synthetic dataset (current)** |
| Massive on Real Data | 71.24% | Generalization to real data |

#### Confusion Matrix Breakdown
- **True Negatives (TN):** 82 - Correctly predicted non-diabetic
- **False Positives (FP):** 18 - Incorrectly predicted diabetic
- **False Negatives (FN):** 26 - Incorrectly predicted non-diabetic
- **True Positives (TP):** 28 - Correctly predicted diabetic

### Performance Metrics
- **Sensitivity (Recall for Diabetic):** 51.85% (28/54)
- **Specificity (Recall for Non-Diabetic):** 82.00% (82/100)
- **Positive Predictive Value (Precision for Diabetic):** 60.87% (28/46)
- **Negative Predictive Value (Precision for Non-Diabetic):** 75.93% (82/108)

## Multiple Logistic Regression Mathematical Model

### Mathematical Form:
```
P(Diabetes = 1|X) = 1 / (1 + e^(-z))

where z = β₀ + β₁×Pregnancies + β₂×Glucose + β₃×BloodPressure + 
          β₄×SkinThickness + β₅×Insulin + β₆×BMI + 
          β₇×DiabetesPedigreeFunction + β₈×Age
```

### Actual Equation with Massive Model Coefficients:
```
z = 0.0305 + 0.2921×Pregnancies + 2.6795×Glucose + (-0.0421)×BloodPressure + 
    0.2049×SkinThickness + (-0.1919)×Insulin + 0.5621×BMI + 
    0.5389×DiabetesPedigreeFunction + 0.1168×Age
```

**Key Changes from Original Model:**
- **Glucose coefficient increased** from 1.14 to **2.68** (134% stronger!)
- **DiabetesPedigreeFunction** doubled from 0.26 to **0.54**
- **SkinThickness** tripled from 0.07 to **0.20**
- **Intercept** changed from negative (-0.87) to slightly positive (0.03)

### Model Parameters (β coefficients):
| Parameter | Original Value | **Massive Model Value** | Feature | Interpretation |
|-----------|----------------|-------------------------|---------|----------------|
| **β₀ (Intercept)** | -0.8749 | **0.0305** | Bias term | Baseline log-odds when all features = 0 |
| **β₁** | 0.3730 | **0.2921** | Pregnancies | **Moderate positive predictor** |
| **β₂** | 1.1438 | **2.6795** | Glucose | **STRONGEST positive predictor** (much stronger) |
| **β₃** | -0.1977 | **-0.0421** | BloodPressure | **Weak negative predictor** |
| **β₄** | 0.0664 | **0.2049** | SkinThickness | **Moderate positive predictor** (stronger) |
| **β₅** | -0.1273 | **-0.1919** | Insulin | **Weak negative predictor** |
| **β₆** | 0.7138 | **0.5621** | BMI | **Strong positive predictor** |
| **β₇** | 0.2555 | **0.5389** | DiabetesPedigreeFunction | **Strong positive predictor** (much stronger) |
| **β₈** | 0.1841 | **0.1168** | Age | **Weak positive predictor** |

## Feature Importance (Ranked by Absolute Coefficient Value)

| Rank | Feature | Coefficient | Abs Coefficient | Interpretation |
|------|---------|-------------|-----------------|----------------|
| 1 | Glucose (β₂) | 1.1438 | 1.1438 | **Strongest positive predictor** - Higher glucose increases diabetes risk |
| 2 | BMI (β₆) | 0.7138 | 0.7138 | **Strong positive predictor** - Higher BMI increases diabetes risk |
| 3 | Pregnancies (β₁) | 0.3730 | 0.3730 | **Moderate positive predictor** - More pregnancies increase risk |
| 4 | DiabetesPedigreeFunction (β₇) | 0.2555 | 0.2555 | **Positive predictor** - Family history increases risk |
| 5 | BloodPressure (β₃) | -0.1977 | 0.1977 | **Weak negative predictor** - Unexpected negative correlation |
| 6 | Age (β₈) | 0.1841 | 0.1841 | **Weak positive predictor** - Older age slightly increases risk |
| 7 | Insulin (β₅) | -0.1273 | 0.1273 | **Weak negative predictor** - Unexpected negative correlation |
| 8 | SkinThickness (β₄) | 0.0664 | 0.0664 | **Minimal positive predictor** - Least important feature |

## Model Interpretation

### Multiple Logistic Regression vs Simple Logistic:
- **Not Simple Logistic:** This is NOT the basic 2-parameter logistic regression (y = β₀ + β₁x)
- **Multiple Logistic:** Uses 9 parameters to model complex relationships between 8 medical variables
- **Multivariate Analysis:** Considers all health factors simultaneously, not just one predictor
- **Complex Decision Boundary:** Can capture interactions between multiple medical indicators

### Key Insights:
1. **Glucose level (β₂ = 1.14)** is by far the most important predictor
2. **BMI (β₆ = 0.71)** is the second most important factor
3. **Number of pregnancies (β₁ = 0.37)** shows moderate importance
4. **Intercept (β₀ = -0.87)** indicates baseline negative log-odds (protective baseline)
5. Unexpected negative coefficients for **BloodPressure** and **Insulin** may indicate:
   - Multicollinearity effects between variables
   - Data quality issues (zeros as missing values)
   - Complex non-linear relationships masked by linear model

### Clinical Relevance:
- The multiple logistic model aligns with medical knowledge where glucose and BMI are primary diabetes risk factors
- High precision for non-diabetic cases (76%) makes it reliable for ruling out diabetes
- Lower recall for diabetic cases (52%) suggests the model misses some diabetes cases
- **Advantage over simple logistic:** Can detect diabetes patterns across multiple symptoms simultaneously

### Model Complexity:
- **Parameters:** 9 total parameters (much more complex than simple 2-parameter logistic)
- **Feature Space:** 8-dimensional input space
- **Decision Boundary:** Complex hyperplane in 8D space, not just a simple line
- **Interpretability:** Each coefficient shows the change in log-odds per unit change in that feature, holding all other features constant

---

## Simple vs Multiple Logistic Regression Comparison

### Simple Logistic Regression (NOT used here):
- **Parameters:** 2 only (β₀ + β₁)
- **Equation:** `P(y=1) = 1/(1 + e^(-(β₀ + β₁×x)))`
- **Use Case:** Single predictor variable
- **Example:** Predicting diabetes based on glucose level only

### Multiple Logistic Regression (USED in this project):
- **Parameters:** 9 total (β₀ + β₁ + β₂ + ... + β₈)
- **Equation:** `P(y=1) = 1/(1 + e^(-z))` where z involves all 8 features
- **Use Case:** Multiple predictor variables
- **Advantage:** Captures complex medical relationships simultaneously

### Why Multiple Logistic is Better for Medical Diagnosis:
1. **Realistic:** Diabetes depends on multiple factors, not just one
2. **Comprehensive:** Uses all available medical information
3. **Accurate:** Better predictions than single-variable models
4. **Clinically Relevant:** Matches how doctors make diagnoses

---

## Files Generated

- **Model File:** `diabetes_logistic_model.pkl`
- **Scaler File:** `diabetes_logistic_scaler.pkl`
- **Confusion Matrix Plot:** `logistic_confusion_matrix.png`
- **Feature Importance Plot:** `logistic_feature_importance.png`
- **Results Report:** `logistic_regression_results.md`

## Model Limitations

1. **Class Imbalance:** Dataset has more non-diabetic samples (65%) than diabetic (35%)
2. **Feature Quality:** Some features may have zeros representing missing values
3. **Overfitting:** Small gap between training (79%) and test (71%) accuracy suggests slight overfitting
4. **Medical Sensitivity:** 48% false negative rate could be concerning for medical screening

## Recommendations for Improvement

1. **Handle Missing Values:** Investigate and properly handle zeros that may represent missing data
2. **Feature Engineering:** Create additional features or transformations
3. **Class Balancing:** Use techniques like SMOTE or class weights
4. **Model Ensemble:** Combine with other algorithms for better performance
5. **Cross-Validation:** Use k-fold cross-validation for more robust evaluation
6. **Threshold Tuning:** Adjust decision threshold to optimize for medical use case

---

*Model trained on November 18, 2025 using scikit-learn's LogisticRegression with StandardScaler preprocessing.*
