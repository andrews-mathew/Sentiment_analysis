from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load
model = BertForSequenceClassification.from_pretrained("bert_imdb_model")
tokenizer = BertTokenizer.from_pretrained("bert_imdb_model")
model.eval()

text = "The movie was very bad!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()

print("Sentiment:", "Positive" if pred == 1 else "Negative")
