
---


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(path: str):
    """Load dataset from CSV file"""
    return pd.read_csv(path)

def preprocess_features(df: pd.DataFrame):
    """Feature engineering and preprocessing pipeline"""

    df['errorOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
    df['errorDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['isMerchant'] = np.where(df['nameDest'].str.startswith("M"), 1, 0)

    features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest', 'errorOrig',
                'errorDest', 'isMerchant']
    X = df[features]
    y = df['isFraud']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = ['type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    return X, y, preprocessor
