import pickle
import xgboost as xgb
import numpy as np
import re
import string
# Load các thành phần đã lưu
model = xgb.Booster()
model.load_model('../../train_archive/XGBoost/sentiment.model')

vectorizer = pickle.load(open('../../train_archive/XGBoost/vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('../../train_archive/XGBoost/label_encoder.pkl', 'rb'))

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
def preprocess_text(text, remove_stopwords=True):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs and emails
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\S+@\S+", '', text)

    # 3. Remove emojis and non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('utf-8')  # giữ lại ASCII thôi

    # 4. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 5. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 6. Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Remove stopwords (optional)
    # if remove_stopwords:
    #     text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])

    return text
# 11. Test nhanh
sample_texts = [
    # Positive
    "The customer service here is truly excellent, the staff are very helpful and friendly.",
    "I've been using this product for 6 months and I'm completely satisfied. Totally worth the money.",
    "The app runs smoothly, has a beautiful interface, and is very easy to use. 10 out of 10!"     ,

    # Negative
    "I can't believe how bad this product is, I'm extremely disappointed.",
    "The app keeps crashing and it's basically unusable. # Negative",
    "Delivery was almost a week late and nobody answered the customer service hotline.",

    # Neutral
    "The product matches the description, packaging was okay, nothing special.",
    "Attending a virtual conference on AI.",
    "Confusion surrounds me as I navigate through life's choices.",
]
sample_texts = [preprocess_text(text) for text in sample_texts]
# ===== In kết quả =====
for text in sample_texts:
    result = predict_sentiment(text)  # hoặc _lr tùy model
    print(f"\nVăn bản: {text}")
    print(f"Sentiment dự đoán: {result['sentiment']}")
    print("Xác suất:")
    for sentiment, prob in result['probabilities'].items():
        print(f"  - {sentiment}: {prob:.2f}%")

