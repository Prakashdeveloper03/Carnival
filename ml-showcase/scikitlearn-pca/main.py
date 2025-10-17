import mlflow
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import mlflow.sklearn
import matplotlib.pyplot as plt

# Set the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 1. MLflow setup
mlflow.set_experiment("Scikit-learn PCA")

# 2. PCA
n_components = 2
with mlflow.start_run(run_name="PCA") as run:
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    mlflow.log_param("n_components", n_components)
    mlflow.log_metric("explained_variance_ratio", np.sum(pca.explained_variance_ratio_))
    mlflow.sklearn.log_model(
        pca,
        "model",
        input_example=X,
        registered_model_name="Scikit-learn-Iris-PCA"
    )

    # 3. Visualize the PCA result
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Iris Dataset")
    plt.legend(handles=scatter.legend_elements()[0], labels=iris.target_names.tolist())
    plt.savefig("pca_result.png")
    mlflow.log_artifact("pca_result.png")
    plt.close()

    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_)}")
    print(f"Run ID: {run.info.run_id}")
    print("PCA result visualization saved as an artifact in MLflow.")
