import pickle
import xgboost as xgb
import numpy as np
# Load các thành phần đã lưu
model = xgb.Booster()
model.load_model('sentiment_XGB.model')

vectorizer = pickle.load(open('vectorizer_XGB.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder_XGB.pkl', 'rb'))

#Hàm dự đoán cho dữ liệu mới
def predict_sentiment(text, model=model, vectorizer=vectorizer, encoder=label_encoder):

    # Chuyển đổi văn bản thành vector đặc trưng
    text_features = vectorizer.transform([text])
    dtext = xgb.DMatrix(text_features)

    # Dự đoán xác suất
    probabilities = model.predict(dtext)[0]

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

# test
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
    result = predict_sentiment(text)
    print(f"\nVăn bản: {text}")
    print(f"Sentiment dự đoán: {result['sentiment']}")
    print("Xác suất:")
    for sentiment, prob in result['probabilities'].items():
        print(f"  - {sentiment}: {prob:.4f}")