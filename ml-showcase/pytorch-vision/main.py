import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import shap
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt

# Set the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# 1. Load and prepare dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Build the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 3. MLflow setup
mlflow.set_experiment("PyTorch Vision")

with mlflow.start_run() as run:
    # 4. Train the model
    for epoch in range(2):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 5. Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    mlflow.log_param("epochs", 2)
    mlflow.log_metric("accuracy", accuracy)
    sample_input, _ = next(iter(trainloader))
    mlflow.pytorch.log_model(
        net,
        "model",
        input_example=sample_input,
        registered_model_name="PyTorch-CIFAR10-CNN"
    )

    # 6. SHAP Explanations
    batch = next(iter(testloader))
    images, _ = batch
    background = next(iter(trainloader))[0]
    explainer = shap.DeepExplainer(net, background)
    shap_values = explainer.shap_values(images)
    
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(images.numpy(), 1, -1), 1, 2)
    shap.image_plot(shap_numpy, -test_numpy, show=False)
    plt.savefig("shap_explanation.png")
    mlflow.log_artifact("shap_explanation.png")
    plt.close()

    # 7. LIME Explanations
    def predict_proba(images):
        net.eval()
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            return torch.softmax(net(images), dim=1).detach().numpy()

    explainer_lime = lime_image.LimeImageExplainer()
    explanation = explainer_lime.explain_instance(test_numpy[0], predict_proba, top_labels=5, hide_color=0, num_samples=1000)
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title("LIME Explanation")
    plt.savefig("lime_explanation.png")
    mlflow.log_artifact("lime_explanation.png")
    plt.close()

    print(f"Run ID: {run.info.run_id}")
    print("Model, SHAP, and LIME explanations saved in MLflow.")
