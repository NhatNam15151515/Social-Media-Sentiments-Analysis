import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load model, vectorizer và encoder
model = joblib.load('../../train/Support_Vector_Machine/sentiment.model')
vectorizer = joblib.load('../../train/Support_Vector_Machine/vectorizer.pkl')
label_encoder = joblib.load('../../train/Support_Vector_Machine/label_encoder.pkl')
# Hàm dự đoán cho mô hình SVM
def predict_sentiment_svm(text, model=model, vectorizer=vectorizer, encoder=label_encoder):
    # Chuyển văn bản thành vector đặc trưng
    text_features = vectorizer.transform([text])

    # Dự đoán xác suất
    probabilities = model.predict_proba(text_features)[0]

    # Lấy nhãn có xác suất cao nhất
    prediction = np.argmax(probabilities)

    # Chuyển đổi nhãn số thành nhãn gốc
    predicted_sentiment = encoder.inverse_transform([prediction])[0]

    # Tạo dictionary xác suất theo phần trăm
    prob_dict = {
        encoder.inverse_transform([i])[0]: round(prob * 100, 2)
        for i, prob in enumerate(probabilities)
    }

    return {
        'sentiment': predicted_sentiment,
        'probabilities': prob_dict
    }

# Test thử với mẫu
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
    result = predict_sentiment_svm(text)
    print(f"\nVăn bản: {text}")
    print(f"Sentiment dự đoán: {result['sentiment']}")
    print("Xác suất:")
    for sentiment, prob in result['probabilities'].items():
        print(f"  - {sentiment}: {prob:.2f}%")
