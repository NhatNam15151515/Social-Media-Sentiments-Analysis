import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

save_dir = '../../train/Naive_Bayes'
os.makedirs(save_dir, exist_ok=True)

# 1. Load dữ liệu
data = pd.read_csv('../../../data/external/sentimentgroups.csv')
data = data.dropna(subset=['Text', 'Sentiment'])
data = data[data['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

# 2. Encode và vectorize
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Sentiment'])

vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 1))
X = vectorizer.fit_transform(data['Text'])

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

# 6. Vẽ confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn Naive Bayes')
plt.tight_layout()
plt.savefig(f'{save_dir}/confusion_matrix.png')
plt.show()

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
plt.show()

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
plt.show()

# 9. Lưu model và các thành phần
pickle.dump(model, open(f'{save_dir}/sentiment.pkl', 'wb'))
pickle.dump(vectorizer, open(f'{save_dir}/vectorizer.pkl', 'wb'))
pickle.dump(label_encoder, open(f'{save_dir}/label_encoder.pkl', 'wb'))
