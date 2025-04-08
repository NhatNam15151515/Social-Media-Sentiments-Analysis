import pickle

# Load mô hình Logistic Regression
with open('sentiment_LR.model', 'rb') as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer và LabelEncoder
with open('vectorizer_LR.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('label_encoder_LR.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Dự đoán thử 1 văn bản
sample_texts = [
    "I really love this product, it's amazing!",
    "This is the worst experience I've ever had.",
    "The product was delivered on time, as expected.",
    "This is bad",
    "Nothing ever goes the way I want it to.",
    "bad luck",
    "luck"
]
for text in sample_texts:
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    print(f"\nVăn bản: {text}")
    predicted_label = label_encoder.inverse_transform(prediction)
    print(f"Sentiment dự đoán: {predicted_label[0]}")

