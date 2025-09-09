import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data, preprocess_features

def train_model():
    df = load_data("data/Fraud_Analysis_Dataset.csv")
    X, y, preprocessor = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('smote', SMOTE(random_state=42)),
                               ('classifier', rf)])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    joblib.dump(grid.best_estimator_, "models/fraud_detection_model.pkl")
    print("✅ Model trained and saved at models/fraud_detection_model.pkl")

if __name__ == "__main__":
    train_model()
