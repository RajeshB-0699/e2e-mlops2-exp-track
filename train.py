# import pickle
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# import mlflow
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from dotenv import load_dotenv

# load_dotenv()

# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
# MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")


# # https://dagshub.com/RajeshB-0699/e2e-mlops2-exp-track.mlflow

# # import dagshub
# # dagshub.init(repo_owner='RajeshB-0699', repo_name='e2e-mlops2-exp-track', mlflow=True)

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment("IRIS MODEL TRAINING")
# iris = load_iris()
# X, y = iris.data, iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# with mlflow.start_run():
#     n_estimators = 100
#     max_depth = None
#     random_state = 42
    
#     mlflow.log_param("n_estimators", n_estimators)
#     mlflow.log_param("max_depth", max_depth)
#     mlflow.log_param("random_state", random_state)

#     model = RandomForestClassifier(
#         n_estimators=n_estimators,
#         random_state=random_state,
#         max_depth=max_depth
#     )

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average="macro")
#     recall = recall_score(y_test, y_pred, average="macro")

#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)

#     model_dir = "model"
#     os.makedirs(model_dir, exist_ok=True)
#     model_path = os.path.join(model_dir, "iris_model.pkl")

#     with open(model_path, "wb") as f:
#         pickle.dump(model, f)
    
#     mlflow.log_artifact(model_path)

#     print(f"Model completed with accuracy : {accuracy}")


import os
import pickle
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
from mlflow.tracking import MlflowClient

# ----------------------------
# 1️⃣ Load environment variables
# ----------------------------
# Create a .env file with:
# MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
# MLFLOW_TRACKING_USERNAME=<your-username>
# MLFLOW_TRACKING_PASSWORD=<your-personal-access-token>
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not all([MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD]):
    raise ValueError("Please set MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_PASSWORD in .env")

# ----------------------------
# 2️⃣ Set MLflow tracking URI and experiment
# ----------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment_name = "IRIS MODEL TRAINING"
mlflow.set_experiment(experiment_name)

# Optional: check existing runs on DagsHub
client = MlflowClient()
exp = client.get_experiment_by_name(experiment_name)
if exp:
    print(f"Experiment '{experiment_name}' exists. ID: {exp.experiment_id}")
    runs = client.search_runs(exp.experiment_id)
    print(f"Total existing runs in this experiment: {len(runs)}")
else:
    print(f"Experiment '{experiment_name}' does not exist. It will be created by MLflow.")

# ----------------------------
# 3️⃣ Load dataset
# ----------------------------
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 4️⃣ Train and log model
# ----------------------------
with mlflow.start_run():
    # Hyperparameters
    n_estimators = 100
    max_depth = None
    random_state = 42

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # Model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Save model artifact
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "iris_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact(model_path)

    print(f"Model completed with accuracy : {accuracy}")

# ----------------------------
# 5️⃣ List all runs in this experiment (CI + local)
# ----------------------------
print("\nAll runs in this experiment:")
for run in client.search_runs(exp.experiment_id):
    print(f"Run ID: {run.info.run_id}, Status: {run.info.status}, Accuracy: {run.data.metrics.get('accuracy')}")

