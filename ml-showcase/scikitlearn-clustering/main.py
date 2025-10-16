import mlflow
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import mlflow.sklearn
import matplotlib.pyplot as plt

# Set the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Load dataset
iris = load_iris()
X = iris.data

# 1. MLflow setup
mlflow.set_experiment("Scikit-learn Clustering")

# 2. K-Means Clustering
n_clusters = 3
with mlflow.start_run(run_name="KMeans") as run:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette = silhouette_score(X, labels)

    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_metric("silhouette_score", silhouette)
    mlflow.sklearn.log_model(
        kmeans,
        "model",
        input_example=X,
        registered_model_name="Scikit-learn-Iris-KMeans"
    )

    # 3. Visualize the clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.title("K-Means Clustering of Iris Dataset")
    plt.savefig("kmeans_clusters.png")
    mlflow.log_artifact("kmeans_clusters.png")
    plt.close()

    print(f"Silhouette Score: {silhouette}")
    print(f"Run ID: {run.info.run_id}")
    print("Cluster visualization saved as an artifact in MLflow.")
