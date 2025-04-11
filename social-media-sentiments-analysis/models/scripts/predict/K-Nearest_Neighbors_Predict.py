import numpy as np
import joblib

# Load lại model, vectorizer và label encoder nếu cần
model = joblib.load('../../train/KNN/sentiment.pkl')
vectorizer = joblib.load('../../train/KNN/vectorizer.pkl')
label_encoder = joblib.load('../../train/KNN/label_encoder.pkl')

def predict_sentiment(text, model=model, vectorizer=vectorizer, encoder=label_encoder):
    # Chuyển đổi văn bản thành vector đặc trưng
    text_features = vectorizer.transform([text])

    # Dự đoán xác suất
    probabilities = model.predict_proba(text_features)[0]

    # Lấy nhãn có xác suất cao nhất
    prediction = np.argmax(probabilities)

    # Chuyển đổi nhãn số thành nhãn gốc
    predicted_sentiment = encoder.inverse_transform([prediction])[0]

    # Tạo dictionary cho xác suất mỗi lớp
    prob_dict = {encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}

    return {
        'sentiment': predicted_sentiment,
        'probabilities': prob_dict
    }

# Test
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

for text in sample_texts:
    result = predict_sentiment(text)
    print(f"\nVăn bản: {text}")
    print(f"Sentiment dự đoán: {result['sentiment']}")
    print("Xác suất:")
    for sentiment, prob in result['probabilities'].items():
        print(f"  - {sentiment}: {prob * 100:.2f}%")
