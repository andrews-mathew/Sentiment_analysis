import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from collections import Counter
import re

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load the dataset
dataset = load_dataset("amazon_polarity")

# Tokenizer
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

# Build vocab
counter = Counter()
for sample in dataset["train"]["content"][:50000]:  # subset to speed up vocab building
    counter.update(tokenize(sample))

vocab = {"<pad>": 0, "<unk>": 1}
for i, (word, freq) in enumerate(counter.items(), start=2):
    vocab[word] = i

def encode(text):
    return [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)]

# Collate function
def collate_batch(batch):
    texts, labels = [], []
    for sample in batch:
        encoded = torch.tensor(encode(sample["content"]), dtype=torch.int64)
        texts.append(encoded)
        labels.append(torch.tensor(sample["label"], dtype=torch.int64))
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    labels = torch.stack(labels)
    return texts, labels

# DataLoaders
train_dataloader = DataLoader(dataset["train"].select(range(500000)), batch_size=64, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(dataset["test"].select(range(200000)), batch_size=64, shuffle=False, collate_fn=collate_batch)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(self.dropout(hidden[-1]))

# Instantiate model
model = LSTMModel(vocab_size=len(vocab), embed_dim=100, hidden_dim=128, output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train function
def train(model, dataloader, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=running_loss / (progress.n + 1))

# Evaluate
def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {100 * correct / total:.2f}%")

# Run
train(model, train_dataloader, epochs=5)
evaluate(model, test_dataloader)

# Save model
torch.save(model.state_dict(), "amazon_lstm_model.pth")
print("Model saved as 'amazon_lstm_model.pth'")
