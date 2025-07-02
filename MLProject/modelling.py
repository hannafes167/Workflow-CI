# modelling.py

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Load preprocessed data
X_train = pd.read_csv("heart-disease_preprocessing/X_train.csv").astype('float64')
X_test = pd.read_csv("heart-disease_preprocessing/X_test.csv").astype('float64')
y_train = pd.read_csv("heart-disease_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("heart-disease_preprocessing/y_test.csv").values.ravel()

# Setup MLflow (lokal tracking)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("HeartDisease_CI")

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100],
    'max_depth': [10],
    'min_samples_split': [2]
}

model = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1
)

model.fit(X_train, y_train)
best_model = model.best_estimator_

# Start MLflow run
with mlflow.start_run(run_name="ci_model_run"):
    mlflow.log_param("n_estimators", best_model.n_estimators)
    mlflow.log_param("max_depth", best_model.max_depth)
    mlflow.log_param("min_samples_split", best_model.min_samples_split)

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Simpan model sebagai artifact untuk CI
    joblib.dump(best_model, "model.pkl")
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    print("✅ Model retraining via CI selesai dan disimpan.")
