import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import lime
import lime.lime_tabular
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Load and prepare dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 1. MLflow setup
mlflow.set_experiment("PyTorch Classification")

# 2. Build the model
class ClassificationNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

model = ClassificationNet(X_train.shape[1], len(iris.target_names))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

with mlflow.start_run() as run:
    # 3. Train the model
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # 4. Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        _, y_pred = torch.max(y_pred_tensor, 1)
        accuracy = accuracy_score(y_test, y_pred.numpy())

    mlflow.log_param("epochs", 100)
    mlflow.log_param("lr", 0.001)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.pytorch.log_model(
        model,
        "model",
        input_example=X_train_tensor,
        registered_model_name="PyTorch-Iris-Classifier"
    )

    # 5. SHAP Explanations
    explainer_shap = shap.KernelExplainer(lambda x: model(torch.from_numpy(x).float()).detach().numpy(), X_train[:100])
    shap_values = explainer_shap.shap_values(X_test[:10])

    shap.summary_plot(shap_values, X_test[:10], feature_names=iris.feature_names, class_names=iris.target_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig("shap_summary_plot.png")
    mlflow.log_artifact("shap_summary_plot.png")
    plt.close()

    # 6. LIME Explanations
    def predict_proba(x):
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(x).float()).numpy()

    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        verbose=True,
        mode="classification",
    )

    i = 10
    exp = explainer_lime.explain_instance(X_test[i], predict_proba, num_features=5)
    exp.save_to_file("lime_explanation.html")
    mlflow.log_artifact("lime_explanation.html")

    print(f"Run ID: {run.info.run_id}")
    print("Model, SHAP, and LIME explanations saved in MLflow.")
