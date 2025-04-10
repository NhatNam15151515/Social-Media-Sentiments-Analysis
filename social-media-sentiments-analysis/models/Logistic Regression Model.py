import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

# 1. Đọc dữ liệu
data = pd.read_csv('../data/external/sentimentgroups.csv')
data = data.dropna(subset=['Text', 'Sentiment'])
data = data[data['Sentiment'].isin(['positive', 'negative', 'neutral'])]

# 2. Mã hóa nhãn
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Sentiment'])

# 3. Trích xuất đặc trưng TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    max_features=1500,
    stop_words='english',
    ngram_range=(1, 1)
)
X = tfidf_vectorizer.fit_transform(data['Text'])

# 4. Chia tập train, validation, test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
)

print(f"Tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"Tập validation: {X_val.shape[0]} mẫu")
print(f"Tập kiểm tra: {X_test.shape[0]} mẫu")

# 5. Huấn luyện mô hình Logistic Regression
logreg = LogisticRegression(
    C=1.0,
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    random_state=42
)
logreg.fit(X_train, y_train)

# 6. Dự đoán và đánh giá
y_val_pred = logreg.predict(X_val)
print("\nKết quả trên tập validation:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

y_test_pred = logreg.predict(X_test)
print("\nKết quả trên tập test:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

# 7. Ma trận nhầm lẫn
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn - Logistic Regression')
plt.savefig('confusion_matrix_LR.png')
plt.show()

# 8. Lưu mô hình và các thành phần
pickle.dump(logreg, open('sentiment_LR.model', 'wb'))
pickle.dump(tfidf_vectorizer, open('vectorizer_LR.pkl', "wb"))
pickle.dump(label_encoder, open('label_encoder_LR.pkl', 'wb'))
