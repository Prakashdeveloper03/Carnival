import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import shap
import lime
from lime import lime_text
import numpy as np

# Set the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# 1. Load and prepare dataset
tokenizer = get_tokenizer('basic_english')
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator((tokenizer(item[1]) for item in train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def text_pipeline(x):
    return vocab(tokenizer(x))

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(1 if _label == 'pos' else 0)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iter = IMDB(split='train')
train_dataloader = torch.utils.data.DataLoader(list(train_iter), batch_size=8, shuffle=False, collate_fn=collate_batch)

# 2. Build the model
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

num_class = 2
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5.0)

# 3. MLflow setup
mlflow.set_experiment("PyTorch NLP")

with mlflow.start_run() as run:
    # 4. Train the model
    for epoch in range(2):
        for i, (label, text, offsets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            optimizer.step()

    # 5. Log model
    sample_input = next(iter(train_dataloader))
    mlflow.pytorch.log_model(
        model,
        "model",
        input_example=(sample_input[1], sample_input[2]),
        registered_model_name="PyTorch-IMDB-LSTM"
    )

    # 6. LIME Explanations
    test_texts = [item[1] for item in IMDB(split='test')]
    
    def predictor(texts):
        model.eval()
        predictions = []
        for text in texts:
            with torch.no_grad():
                text_tensor = torch.tensor(text_pipeline(text)).to(device)
                offsets = torch.tensor([0]).to(device)
                output = model(text_tensor, offsets)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()
                predictions.append(probabilities[0])
        return np.array(predictions)

    explainer = lime_text.LimeTextExplainer(class_names=['neg', 'pos'])
    exp = explainer.explain_instance(test_texts[0], predictor, num_features=6)
    exp.save_to_file('lime_explanation.html')
    mlflow.log_artifact("lime_explanation.html")

    print(f"Run ID: {run.info.run_id}")
    print("Model and LIME explanation saved in MLflow.")
    # SHAP is not easily applicable here due to the EmbeddingBag layer and text processing pipeline.
    print("SHAP is not included for this PyTorch NLP example due to complexity.")
