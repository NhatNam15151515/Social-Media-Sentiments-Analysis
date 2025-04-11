import numpy as np
import pickle

# Load model và các thành phần cần thiết
from sklearn.naive_bayes import MultinomialNB

# Đường dẫn tới mô hình và vectorizer đã lưu
model_path = '../../train/Naive_Bayes/sentiment.pkl'
vectorizer_path = '../../train/Naive_Bayes/vectorizer.pkl'
label_encoder_path = '../../train/Naive_Bayes/label_encoder.pkl'

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))
label_encoder = pickle.load(open(label_encoder_path, 'rb'))
# 10. Hàm dự đoán
def predict_sentiment(text, model=model, vectorizer=vectorizer, encoder=label_encoder):
    features = vectorizer.transform([text])
    probabilities = model.predict_proba(features)[0]
    prediction = np.argmax(probabilities)
    predicted_sentiment = encoder.inverse_transform([prediction])[0]
    prob_dict = {encoder.inverse_transform([i])[0]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}

    return {
        'sentiment': predicted_sentiment,
        'probabilities': prob_dict
    }

# 11. Test nhanh
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
    "Attending a virtual conference on AI.     # Neutral",
]

for text in sample_texts:
    result = predict_sentiment(text)
    print(f"\nVăn bản: {text}")
    print(f"Sentiment dự đoán: {result['sentiment']}")
    print("Xác suất:")
    for sentiment, prob in result['probabilities'].items():
        print(f"  - {sentiment}: {prob:.2f}%")
