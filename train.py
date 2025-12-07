import pickle
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")


# https://dagshub.com/RajeshB-0699/e2e-mlops2-exp-track.mlflow

# import dagshub
# dagshub.init(repo_owner='RajeshB-0699', repo_name='e2e-mlops2-exp-track', mlflow=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("IRIS MODEL TRAINING")
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    n_estimators = 100
    max_depth = None
    random_state = 42
    
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "iris_model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    mlflow.log_artifact(model_path)

    print(f"Model completed with accuracy : {accuracy}")
