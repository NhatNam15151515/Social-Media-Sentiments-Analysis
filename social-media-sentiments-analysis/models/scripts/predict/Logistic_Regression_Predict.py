import pickle
import re
import string

model_path = '../../train_archive/Logistic_Regression/sentiment.pkl'
vectorizer_path = '../../train_archive/Logistic_Regression/vectorizer.pkl'
label_encoder_path = '../../train_archive/Logistic_Regression/label_encoder.pkl'

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