import torch
import torch.nn as nn
import re
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from datasets import load_dataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

# Rebuild vocab (same as training)
dataset = load_dataset("amazon_polarity")
counter = Counter()
for sample in dataset["train"]["content"][:50000]:
    counter.update(tokenize(sample))

vocab = {"<pad>": 0, "<unk>": 1}
for i, (word, _) in enumerate(counter.items(), start=2):
    vocab[word] = i

# Encode input text
def encode(text):
    return [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)]

# Model class (same as training)
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

# Load model
model = LSTMModel(vocab_size=len(vocab), embed_dim=100, hidden_dim=128, output_dim=2).to(device)
model.load_state_dict(torch.load("amazon_lstm_model.pth", map_location=device))
model.eval()

# Predict function
def predict_sentiment(text):
    with torch.no_grad():
        encoded = torch.tensor(encode(text), dtype=torch.int64)
        padded = pad_sequence([encoded], batch_first=True, padding_value=vocab["<pad>"]).to(device)
        output = model(padded)
        pred = torch.argmax(output, dim=1).item()
        return "Positive 😊" if pred == 1 else "Negative 😞"

# Example usage
text_input = input("Enter a product review: ")
print("Sentiment:", predict_sentiment(text_input))
