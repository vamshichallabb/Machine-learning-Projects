# Fraud Detection ML Project

This project uses machine learning to detect fraudulent transactions with a Random Forest classifier and SMOTE for handling class imbalance. It includes modular scripts for data preprocessing, model training, evaluation, and prediction.

## Project Structure
- `Fraud_Analysis_Dataset.csv`: Transaction dataset source.
- `fraud_detection.py`: Main script with the full pipeline (data loading, preprocessing, training, evaluation, visualization).
- `data_preprocessing.py`: Functions for loading and preprocessing data (feature engineering, scaling, encoding).
- `train_model.py`: Script to train the model with hyperparameter tuning and save it.
- `evaluate_model.py`: Script to evaluate the model’s performance (classification report, ROC AUC, PR AUC).
- `prediction_model.py`: Script to predict fraud on new transactions.
- `requirements.txt`: Python dependencies.

## Requirements
- Python 3.8+
- Libraries:
  ```bash
  pandas
  numpy
  scikit-learn
  imbalanced-learn
  matplotlib
  seaborn
  joblib
