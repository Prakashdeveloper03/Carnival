import mlflow
import mlflow.tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import shap
import lime
from lime import lime_text
import numpy as np
import os

# Set the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# 1. Load and prepare dataset
(ds_train, ds_test), ds_info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

train_texts = [text.numpy().decode('utf-8') for text, label in ds_train]
train_labels = [label.numpy() for text, label in ds_train]
test_texts = [text.numpy().decode('utf-8') for text, label in ds_test]
test_labels = [label.numpy() for text, label in ds_test]

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_texts)
X_train = tokenizer.texts_to_sequences(train_texts)
X_test = tokenizer.texts_to_sequences(test_texts)

X_train = pad_sequences(X_train, maxlen=256)
X_test = pad_sequences(X_test, maxlen=256)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# 2. MLflow setup
mlflow.set_experiment("TensorFlow NLP")

with mlflow.start_run() as run:
    # 3. Build the model
    model = Sequential([
        Embedding(10000, 128, input_length=256),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 4. TensorBoard callback
    log_dir = "logs/fit/" + run.info.run_id
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 5. Train the model
    history = model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

    mlflow.log_param("epochs", 2)
    mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
    mlflow.tensorflow.log_model(
        model,
        "model",
        input_example=X_train,
        registered_model_name="TensorFlow-IMDB-LSTM"
    )

    # 6. SHAP Explanations
    explainer = shap.KernelExplainer(lambda x: model.predict(x), X_train[:10])
    shap_values = explainer.shap_values(X_test[:1], nsamples=100)
    
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test[0], show=False, matplotlib=True)
    plt.savefig("shap_explanation.png")
    mlflow.log_artifact("shap_explanation.png")
    plt.close()

    # 7. LIME Explanations
    def predictor(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        pad_seqs = pad_sequences(seqs, maxlen=256)
        return model.predict(pad_seqs)

    explainer_lime = lime_text.LimeTextExplainer(class_names=['negative', 'positive'])
    exp = explainer_lime.explain_instance(test_texts[0], predictor, num_features=6)
    exp.save_to_file('lime_explanation.html')
    mlflow.log_artifact("lime_explanation.html")

    mlflow.log_artifacts(log_dir, artifact_path="tensorboard_logs")

    print(f"Run ID: {run.info.run_id}")
    print("Model, SHAP, LIME, and TensorBoard logs saved in MLflow.")
