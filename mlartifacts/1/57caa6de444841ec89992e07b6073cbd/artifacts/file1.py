import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Use SQLite backend instead of filesystem


# ✅ Define experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Load Wine dataxXset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 5


mlflow.set_experiment('YT-MLOPS-Exp1')

# Each run will be logged separately under the same experiment
with mlflow.start_run(run_name="Run-1"):
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # ✅ Log metrics and params
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    plt.savefig("Confusion-matrix.png")
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # ✅ Add tags
    mlflow.set_tags({"Author": "SANJAYSRINIVAS B", "Project": "Wine Classification"})

    # ✅ Log model safely using Skops format
    mlflow.sklearn.log_model(
        rf,
        name="Random-Forest-Model",
        serialization_format="skops"
    )

    print("Accuracy:", accuracy)
