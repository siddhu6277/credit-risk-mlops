# Credit Risk Prediction System using MLOps

## Project Overview

The Credit Risk Prediction System predicts whether a customer is likely to have good or bad credit risk based on customer and loan-related information.

The project combines Machine Learning with MLOps practices to make the model reproducible, trackable, and monitorable.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- MLflow
- DVC
- Evidently
- Git

## Machine Learning Models

The following models were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The final selected model is:

**Logistic Regression**

## Final Model Results

| Metric | Score |
|---|---:|
| Accuracy | 80.00% |
| Precision | 70.83% |
| Recall | 56.67% |
| F1 Score | 62.96% |
| ROC-AUC | 80.62% |

## MLOps Workflow

```text
Dataset
   ↓
Data Preparation
   ↓
EDA
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
MLflow Tracking
   ↓
DVC Version Control
   ↓
Evidently Data Drift Monitoring
   ↓
Streamlit Application