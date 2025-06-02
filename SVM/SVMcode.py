import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load IMDb dataset from local CSV
df = pd.read_csv("E:/SVMpredictionmodel/IMDB Dataset.csv/IMDB Dataset.csv")

# Optional: View basic info
print("Dataset loaded. Sample:")
print(df.head())

# Step 2: Clean and preprocess
# Convert sentiments to binary (positive:1, negative:0)
df['label'] = df['sentiment'].apply(lambda x: 1 if x.lower() == 'positive' else 0)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train SVM model
model = LinearSVC()
print("Training SVM...")
model.fit(X_train_vec, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_vec)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the model and vectorizer
joblib.dump(model, "svm_imdb_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nModel and vectorizer saved as 'svm_imdb_model.pkl' and 'tfidf_vectorizer.pkl'")
