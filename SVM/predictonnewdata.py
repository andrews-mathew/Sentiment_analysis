import joblib

# Load saved model and vectorizer
model = joblib.load("svm_imdb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Predict
sample_text = ["The movie was spectacular!"]
sample_vec = vectorizer.transform(sample_text)
prediction = model.predict(sample_vec)
print("Prediction:", "Positive" if prediction[0] == 1 else "Negative")
