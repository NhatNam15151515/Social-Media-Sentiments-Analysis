import pickle
import xgboost as xgb
import numpy as np
# Load các thành phần đã lưu
model = xgb.Booster()
model.load_model('../../train/XGBoost/sentiment.model')

vectorizer = pickle.load(open('../../train/XGBoost/vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('../../train/XGBoost/label_encoder.pkl', 'rb'))

def predict_sentiment(text, model=model, vectorizer=vectorizer, encoder=label_encoder):
    text_features = vectorizer.transform([text])
    dtext = xgb.DMatrix(text_features)
    probabilities = model.predict(dtext)[0]
    prediction = np.argmax(probabilities)
    predicted_sentiment = encoder.inverse_transform([prediction])[0]
    prob_dict = {encoder.inverse_transform([i])[0]: prob * 100 for i, prob in enumerate(probabilities)}

    return {
        'sentiment': predicted_sentiment,
        'probabilities': prob_dict
    }
# ===== Dữ liệu test mẫu =====
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

# ===== In kết quả =====
for text in sample_texts:
    result = predict_sentiment(text)  # hoặc _lr tùy model
    print(f"\nVăn bản: {text}")
    print(f"Sentiment dự đoán: {result['sentiment']}")
    print("Xác suất:")
    for sentiment, prob in result['probabilities'].items():
        print(f"  - {sentiment}: {prob:.2f}%")

