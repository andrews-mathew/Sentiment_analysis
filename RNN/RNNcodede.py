# ======= RNN Training with GloVe and Model Saving =======
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
from collections import Counter
from tqdm import tqdm

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 200
BATCH_SIZE = 64
EMBED_DIM = 100
HIDDEN_DIM = 128
EPOCHS = 25
CSV_PATH = "E:/SVMpredictionmodel/IMDB Dataset.csv/IMDB Dataset.csv"
GLOVE_PATH = "glove.6B.100d.txt"
MODEL_PATH = "sentiment_rnn_glove.pth"

# 1. Load and preprocess CSV
df = pd.read_csv(CSV_PATH)
df.dropna(inplace=True)
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
texts, labels = df['review'].values, df['sentiment'].values

# 2. Simple tokenizer
def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

tokenized_texts = [simple_tokenize(text) for text in texts]

# 3. Build vocab
word_counts = Counter(word for sentence in tokenized_texts for word in sentence)
vocab = {"<pad>": 0, "<unk>": 1}
for i, (word, _) in enumerate(word_counts.items(), start=2):
    vocab[word] = i

# 4. Load GloVe embeddings
def load_glove(glove_path, vocab, embed_dim):
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim))
    embeddings[0] = np.zeros(embed_dim)
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                idx = vocab[word]
                vec = np.array(parts[1:], dtype=np.float32)
                embeddings[idx] = vec
    return torch.tensor(embeddings, dtype=torch.float)

embedding_matrix = load_glove(GLOVE_PATH, vocab, EMBED_DIM)

# 5. Encode

def encode(sentence):
    ids = [vocab.get(word, vocab["<unk>"]) for word in sentence]
    return ids[:MAX_LEN] + [0] * (MAX_LEN - len(ids))

# 6. Dataset class
class IMDBDataset(Dataset):
    def __init__(self, tokenized_texts, labels):
        self.data = [(encode(tokens), label) for tokens, label in zip(tokenized_texts, labels)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(tokenized_texts, labels, test_size=0.2, random_state=42)
train_dataset = IMDBDataset(X_train, y_train)
test_dataset = IMDBDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 8. RNN Model class
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pretrained_embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, hn = self.rnn(x)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(self.dropout(hn)).squeeze()

# 9. Train
model = SentimentRNN(len(vocab), EMBED_DIM, HIDDEN_DIM, embedding_matrix).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for inputs, labels in progress:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix({"loss": loss.item()})
    print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss:.4f}")

# 10. Evaluate
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = torch.sigmoid(model(inputs))
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 11. Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab
}, MODEL_PATH)
print("✅ Model and vocab saved.")
