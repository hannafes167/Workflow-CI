import pandas as pd
import mlflow
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# Load dataset hasil preprocessing
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# Atur eksperimen
mlflow.set_experiment("Heart Disease Grid Search")

# Parameter untuk tuning
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5]
}

# Model dasar
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)

# Mulai run manual logging
with mlflow.start_run(run_name="GridSearch_RF_ManualLogging"):

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Logging parameter terbaik
    for param, value in grid_search.best_params_.items():
        mlflow.log_param(param, value)

    # Logging metrik manual
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # Simpan laporan klasifikasi ke file
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # Simpan model ke file
    joblib.dump(best_model, "rf_best_model.joblib")
    mlflow.log_artifact("rf_best_model.joblib")

    # Cetak hasil evaluasi
    print("=== HASIL EVALUASI ===")
    print("Best Params:", grid_search.best_params_)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", report)
