# Diabetes Prediction System

A machine learning project for predicting diabetes using the PIMA Indians Diabetes Dataset.

## Project Overview

This project uses machine learning classification algorithms to predict whether a person is likely to have diabetes based on medical attributes.

The project includes data preprocessing, feature scaling, training multiple machine learning models, comparing their accuracy, and saving the best-performing model.

## Dataset

**PIMA Indians Diabetes Dataset**

The dataset contains medical diagnostic measurements and a target variable (`Outcome`) indicating whether diabetes is present.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

## Machine Learning Models

The following classification algorithms are trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. XGBoost

The model with the highest accuracy is selected as the best model and saved as:

`diabetes_best_model.pkl`

## Project Workflow

1. Load the diabetes dataset.
2. Perform data preprocessing.
3. Split the data into training and testing sets.
4. Apply feature scaling using `StandardScaler`.
5. Train multiple classification models.
6. Compare model accuracy.
7. Generate a classification report for the best model.
8. Save the best-performing model for future use.

## Project Files

| File | Description |
|------|-------------|
| `task1.py` | Data loading, preprocessing, scaling and preparation |
| `task2.py` | Model training, evaluation and comparison |
| `diabetes.csv` | PIMA Indians Diabetes dataset |
| `processed_data.pkl` | Processed training and testing data |
| `scaler.pkl` | Saved feature scaler |
| `diabetes_best_model.pkl` | Saved best-performing machine learning model |

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/MahimaAnade/Diabetes-Prediction-System.git
