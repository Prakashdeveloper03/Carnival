import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import shap
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
import os

# Set the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Load and prepare dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 1. MLflow setup
mlflow.set_experiment("TensorFlow Vision")

with mlflow.start_run() as run:
    # 2. Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 3. TensorBoard callback
    log_dir = "logs/fit/" + run.info.run_id
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 4. Train the model
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

    # Log parameters and metrics
    mlflow.log_param("epochs", 5)
    mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
    mlflow.tensorflow.log_model(
        model,
        "model",
        input_example=X_train,
        registered_model_name="TensorFlow-CIFAR10-CNN"
    )

    # 5. SHAP Explanations
    background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    explainer_shap = shap.DeepExplainer(model, background)
    shap_values = explainer_shap.shap_values(X_test[:5])
    
    shap.image_plot(shap_values, -X_test[:5], show=False)
    plt.savefig("shap_explanation.png")
    mlflow.log_artifact("shap_explanation.png")
    plt.close()

    # 6. LIME Explanations
    explainer_lime = lime_image.LimeImageExplainer()
    explanation = explainer_lime.explain_instance(X_test[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title("LIME Explanation")
    plt.savefig("lime_explanation.png")
    mlflow.log_artifact("lime_explanation.png")
    plt.close()

    # Log TensorBoard logs
    mlflow.log_artifacts(log_dir, artifact_path="tensorboard_logs")

    print(f"Run ID: {run.info.run_id}")
    print("Model, SHAP, LIME, and TensorBoard logs saved in MLflow.")
