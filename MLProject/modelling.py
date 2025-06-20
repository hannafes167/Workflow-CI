import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train = pd.read_csv("X_train.csv").astype('float64')
X_test = pd.read_csv("X_test.csv").astype('float64')
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

mlflow.set_tracking_uri("file:./mlruns")  
mlflow.set_experiment("HeartDisease_Basic")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="basic_autolog_run"):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

  
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
    mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
    mlflow.log_metric("test_f1", f1_score(y_test, y_pred))

    print("Training dan testing selesai.")
