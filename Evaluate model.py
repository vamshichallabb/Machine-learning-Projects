import joblib
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from data_preprocessing import load_data, preprocess_features

def evaluate_model():
    df = load_data("data/Fraud_Analysis_Dataset.csv")
    X, y, preprocessor = preprocess_features(df)
    model = joblib.load("models/fraud_detection_model.pkl")

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print("Classification Report:\n", classification_report(y, y_pred))
    print("ROC AUC:", roc_auc_score(y, y_proba))

    precision, recall, _ = precision_recall_curve(y, y_proba)
    pr_auc = auc(recall, precision)
    print("PR AUC:", pr_auc)

if __name__ == "__main__":
    evaluate_model()
