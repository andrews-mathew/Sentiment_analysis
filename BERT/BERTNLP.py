import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import numpy as np

# ✅ Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ✅ 1. Load Dataset
df = pd.read_csv('E:/SVMpredictionmodel/IMDB Dataset.csv/IMDB Dataset.csv')
df['label'] = df['sentiment'].apply(lambda x: 1 if x.lower() == 'positive' else 0)

# ✅ 2. Train-Test Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

# ✅ 3. Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ✅ 4. Custom Dataset Class
class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=200): #128
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ✅ 5. Create DataLoaders
train_dataset = IMDbDataset(train_texts, train_labels, tokenizer)
test_dataset = IMDbDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ✅ 6. Load BERT Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# ✅ 7. Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# ✅ 8. Training Loop
# ✅ 8. Training Loop with Progress Bar
epochs = 3
model.train()

for epoch in range(epochs):
    print(f"\n🔁 Epoch {epoch + 1}/{epochs}")
    running_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"✅ Epoch {epoch + 1} completed — Average Loss: {avg_loss:.4f}")


# ✅ 9. Evaluation with Progress Bar
model.eval()
predictions = []
true_labels = []

print("\n🔎 Evaluating on Test Set...")
with torch.no_grad():
    eval_bar = tqdm(test_loader, desc="Evaluating", leave=False)
    for batch in eval_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())


# ✅ 10. Report
print("Accuracy:", accuracy_score(true_labels, predictions))
print("Classification Report:\n", classification_report(true_labels, predictions))

# ✅ 11. Save Model and Tokenizer
model.save_pretrained("bert_imdb_model")
tokenizer.save_pretrained("bert_imdb_model")
print("✅ Model and tokenizer saved to 'bert_imdb_model'")
