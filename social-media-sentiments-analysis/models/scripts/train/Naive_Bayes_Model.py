import pandas as pd
import numpy as np
import os
import re
import string
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Các từ có giá trị thấp
low_value_words = {'new', 'like', 'feeling', 'day', 'world'}

save_dir = '../../train_archive/Naive_Bayes'
os.makedirs(save_dir, exist_ok=True)

# 1. Load dữ liệu
data = pd.read_csv('../../../data/external/sentimentgroups.csv')
data = data.dropna(subset=['Text', 'Sentiment'])
data = data[data['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

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
    if remove_stopwords:
        text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])

    # 8. Remove low value word
    text = ' '.join([word for word in text.split() if word not in low_value_words])

    return text

# 2. Mã hóa nhãn
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Sentiment'])

# 3. Tiền xử lý + Vector hóa văn bản
data['Clean_Text'] = data['Text'].apply(preprocess_text)
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3) ) # Tạo vector hóa cho văn bản
X = vectorizer.fit_transform(data['Clean_Text'])
print("\n 20 dòng đầu tiên của dữ liệu sau khi tiền xử lý:")
print(data[[ 'Clean_Text', 'Sentiment']].head(20))

# 3. Chia dữ liệu train/val/test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, stratify=y_train_val, random_state=42)

# 4. Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Đánh giá
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

print("\nKết quả trên tập validation:")
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

print("\nKết quả trên tập test:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))
print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
# 6. Vẽ confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn Naive Bayes')
plt.tight_layout()
plt.savefig(f'{save_dir}/confusion_matrix.png')


# 7. Vẽ biểu đồ theo lớp
metrics_val = precision_recall_fscore_support(y_val, y_val_pred, average=None, labels=range(len(label_encoder.classes_)))
metrics_test = precision_recall_fscore_support(y_test, y_test_pred, average=None, labels=range(len(label_encoder.classes_)))

labels = label_encoder.classes_
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(1, 3, figsize=(20, 6))

ax[0].bar(x - width/2, metrics_val[0], width, label='Validation', color='skyblue')
ax[0].bar(x + width/2, metrics_test[0], width, label='Test', color='salmon')
ax[0].set_title('Precision theo lớp')
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].legend()
ax[0].grid(True)

ax[1].bar(x - width/2, metrics_val[1], width, label='Validation', color='skyblue')
ax[1].bar(x + width/2, metrics_test[1], width, label='Test', color='salmon')
ax[1].set_title('Recall theo lớp')
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].legend()
ax[1].grid(True)

ax[2].bar(x - width/2, metrics_val[2], width, label='Validation', color='skyblue')
ax[2].bar(x + width/2, metrics_test[2], width, label='Test', color='salmon')
ax[2].set_title('F1-score theo lớp')
ax[2].set_xticks(x)
ax[2].set_xticklabels(labels)
ax[2].legend()
ax[2].grid(True)

plt.suptitle('Biểu đồ Precision - Recall - F1 theo lớp (Naive Bayes)')
plt.tight_layout()
plt.savefig(f'{save_dir}/metrics_by_class.png')


# 8. Biểu đồ tổng hợp
avg_metrics = {
    'Accuracy': [accuracy_score(y_val, y_val_pred), accuracy_score(y_test, y_test_pred)],
    'Precision': [precision_score(y_val, y_val_pred, average='macro'), precision_score(y_test, y_test_pred, average='macro')],
    'Recall': [recall_score(y_val, y_val_pred, average='macro'), recall_score(y_test, y_test_pred, average='macro')],
    'F1-score': [f1_score(y_val, y_val_pred, average='macro'), f1_score(y_test, y_test_pred, average='macro')]
}

metrics_df = pd.DataFrame(avg_metrics, index=['Validation', 'Test'])
metrics_df.plot(kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black')
plt.title("So sánh các chỉ số trung bình giữa Validation và Test")
plt.ylabel("Giá trị")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{save_dir}/avg_metrics_comparison.png')


# 9. Lưu model và các thành phần
pickle.dump(model, open(f'{save_dir}/sentiment.pkl', 'wb'))
pickle.dump(vectorizer, open(f'{save_dir}/vectorizer.pkl', 'wb'))
pickle.dump(label_encoder, open(f'{save_dir}/label_encoder.pkl', 'wb'))
