import mlflow
import optuna
import shap
import lime
import lime.lime_tabular
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow.sklearn
import matplotlib.pyplot as plt

# Set the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. MLflow setup
mlflow.set_experiment("Scikit-learn Classification")

def objective(trial):
    """Optuna objective function."""
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

    with mlflow.start_run(nested=True):
        # Create and train the model
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        })
        mlflow.log_metric("accuracy", accuracy)

        return 1.0 - accuracy # Optuna minimizes, so we minimize 1 - accuracy

# 2. Optuna hyperparameter tuning
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# 3. Train the best model and log with MLflow
best_params = study.best_params
with mlflow.start_run(run_name="Best Model") as run:
    rf_best = RandomForestClassifier(**best_params, random_state=42)
    rf_best.fit(X_train, y_train)
    y_pred = rf_best.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        rf_best,
        "model",
        input_example=X_train,
        registered_model_name="Scikit-learn-Iris-RF"
    )

    # 4. SHAP Explanations
    explainer_shap = shap.TreeExplainer(rf_best)
    shap_values = explainer_shap.shap_values(X_test)

    # Create and save SHAP summary plot
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, class_names=class_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig("shap_summary_plot.png")
    mlflow.log_artifact("shap_summary_plot.png")
    plt.close()

    # 5. LIME Explanations
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        verbose=True,
        mode="classification",
    )

    # Explain a single instance
    i = 25
    exp = explainer_lime.explain_instance(X_test[i], rf_best.predict_proba, num_features=5)
    exp.save_to_file("lime_explanation.html")
    mlflow.log_artifact("lime_explanation.html")

    print(f"Best trial accuracy: {1.0 - study.best_value}")
    print(f"Best parameters: {best_params}")
    print(f"Run ID: {run.info.run_id}")
    print("SHAP and LIME explanations saved as artifacts in MLflow.")
