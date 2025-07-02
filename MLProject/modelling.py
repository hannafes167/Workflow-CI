# modelling_tuning.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Load preprocessed data
X_train = pd.read_csv("heart-disease_preprocessing/X_train.csv").astype('float64')
X_test = pd.read_csv("heart-disease_preprocessing/X_test.csv").astype('float64')
y_train = pd.read_csv("heart-disease_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("heart-disease_preprocessing/y_test.csv").values.ravel()

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("HeartDisease_Tuning")


# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Logging manual (tanpa start_run)
mlflow.log_param("n_estimators", best_model.n_estimators)
mlflow.log_param("max_depth", best_model.max_depth)
mlflow.log_param("min_samples_split", best_model.min_samples_split)

y_pred = best_model.predict(X_test)

mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
mlflow.log_metric("precision", precision_score(y_test, y_pred))
mlflow.log_metric("recall", recall_score(y_test, y_pred))
mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

mlflow.sklearn.log_model(best_model, "random_forest_model")

print("✅ Training dan tuning selesai.")
