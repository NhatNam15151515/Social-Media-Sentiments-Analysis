import pickle
import os
import numpy as np

# Load các thành phần cần thiết
model_path = '../../train/Logistic_Regression/sentiment.pkl'
vectorizer_path = '../../train/Logistic_Regression/vectorizer.pkl'
label_encoder_path = '../../train/Logistic_Regression/label_encoder.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Hàm dự đoán 1 câu
def predict_sentiment(text):
    X_input = vectorizer.transform([text])
    prediction = model.predict(X_input)
    predicted_label = label_encoder.inverse_transform(prediction)
    proba = model.predict_proba(X_input)[0]
    label_probs = {label_encoder.classes_[i]: round(proba[i]*100, 2) for i in range(len(proba))}

    result = {
        "text": text,
        "prediction": predicted_label[0],
        "probabilities": label_probs
    }
    return result

# ===== Danh sách mẫu test =====
sample_texts = [
    # Positive
    "The customer service here is truly excellent, the staff are very helpful and friendly.     # Positive",
    "I've been using this product for 6 months and I'm completely satisfied. Totally worth the money.     # Positive",
    "The app runs smoothly, has a beautiful interface, and is very easy to use. 10 out of 10! # Positive"     ,

    # Negative
    "I can't believe how bad this product is, I'm extremely disappointed. # Negative",
    "The app keeps crashing and it's basically unusable. # Negative",
    "Delivery was almost a week late and nobody answered the customer service hotline. # Negative",

    # Neutral
    "I received the item yesterday, haven’t had time to try it yet so I can't say much. # Neutral",
    "The product matches the description, packaging was okay, nothing special. # Neutral",
    "It’s alright I guess, not too good but not too bad either.     # Neutral",
]

# ===== Chạy dự đoán cho từng mẫu =====
print("KẾT QUẢ DỰ ĐOÁN:\n")
for text in sample_texts:
    res = predict_sentiment(text)
    print(f"Văn bản: {res['text']}")
    print(f"Dự đoán: {res['prediction']}")
    print(f"Xác suất:")
    for label, prob in res['probabilities'].items():
        print(f"   - {label}: {prob}%")
    print("-" * 50)