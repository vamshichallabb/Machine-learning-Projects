import joblib
import pandas as pd

def predict_new(data: dict):
    model = joblib.load("models/fraud_detection_model.pkl")
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    return prediction[0], probability[0]

if __name__ == "__main__":
    sample_transaction = {
        'type': 'TRANSFER',
        'amount': 10000.0,
        'oldbalanceOrg': 50000.0,
        'newbalanceOrig': 40000.0,
        'oldbalanceDest': 1000.0,
        'newbalanceDest': 11000.0,
        'errorOrig': 0,
        'errorDest': 0,
        'isMerchant': 0
    }

    pred, prob = predict_new(sample_transaction)
    print(f"Prediction: {'Fraud' if pred == 1 else 'Not Fraud'} (Prob: {prob:.2f})")
