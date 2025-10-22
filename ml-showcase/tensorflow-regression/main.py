import mlflow
import mlflow.tensorflow
import tensorflow as tf
import shap
import lime
import lime.lime_tabular
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Set the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Load and prepare dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. MLflow setup
mlflow.set_experiment("TensorFlow Regression")

with mlflow.start_run() as run:
    # 2. Build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=[X_train.shape[1]]),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # 3. TensorBoard callback
    log_dir = "logs/fit/" + run.info.run_id
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 4. Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard_callback],
        verbose=1,
    )

    # Log parameters and metrics
    mlflow.log_param("epochs", 20)
    mlflow.log_metric("final_val_loss", history.history["val_loss"][-1])
    mlflow.log_metric("final_val_mae", history.history["val_mae"][-1])

    # Log the model
    mlflow.tensorflow.log_model(
        model,
        "model",
        input_example=X_train,
        registered_model_name="TensorFlow-California-Housing-Regressor"
    )

    # 5. SHAP Explanations
    explainer_shap = shap.KernelExplainer(model.predict, X_train[:100])
    shap_values = explainer_shap.shap_values(X_test[:100])

    shap.summary_plot(shap_values, X_test[:100], feature_names=housing.feature_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig("shap_summary_plot.png")
    mlflow.log_artifact("shap_summary_plot.png")
    plt.close()

    # 6. LIME Explanations
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=housing.feature_names,
        class_names=["price"],
        verbose=True,
        mode="regression",
    )

    i = 25
    exp = explainer_lime.explain_instance(X_test[i], model.predict, num_features=5)
    exp.save_to_file("lime_explanation.html")
    mlflow.log_artifact("lime_explanation.html")

    # Log TensorBoard logs
    mlflow.log_artifacts(log_dir, artifact_path="tensorboard_logs")

    print(f"Run ID: {run.info.run_id}")
    print("Model, SHAP, LIME, and TensorBoard logs saved in MLflow.")
