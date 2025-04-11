import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

# Thư mục lưu
save_dir = '../../train/KNN'
os.makedirs(save_dir, exist_ok=True)

# 1. Đọc dữ liệu
data = pd.read_csv('../../../data/external/sentimentgroups.csv')
data = data.dropna(subset=['Text', 'Sentiment'])
data = data[data['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

# 2. Mã hóa nhãn
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Sentiment'])

# 3. Vector hóa TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    max_features=1500,
    stop_words='english',
    ngram_range=(1, 1)
)
X = tfidf_vectorizer.fit_transform(data['Text'])

# 4. Chia dữ liệu
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                  test_size=0.176, random_state=42, stratify=y_train_val)

# 5. Huấn luyện model KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 6. Đánh giá
y_val_pred = model.predict(X_val)
print("\nKết quả trên tập validation:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nBáo cáo phân loại:")
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

y_test_pred = model.predict(X_test)
print("\nKết quả trên tập test:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

# 7. Vẽ ma trận nhầm lẫn
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn KNN')
plt.savefig(f'{save_dir}/confusion_matrix.png')
plt.show()

# 8. Lưu model và vectorizer, label encoder
pickle.dump(model, open(f'{save_dir}/sentiment.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open(f'{save_dir}/vectorizer.pkl', 'wb'))
pickle.dump(label_encoder, open(f'{save_dir}/label_encoder.pkl', 'wb'))
# Dự đoán cho validation và test
y_val_pred_labels = model.predict(X_val)
y_test_pred_labels = model.predict(X_test)

# 10. Vẽ biểu đồ F1-score, Precision, Recall theo từng lớp
metrics_val = precision_recall_fscore_support(y_val, y_val_pred_labels, average=None, labels=range(len(label_encoder.classes_)))
metrics_test = precision_recall_fscore_support(y_test, y_test_pred_labels, average=None, labels=range(len(label_encoder.classes_)))

labels = label_encoder.classes_
x = np.arange(len(labels))  # vị trí nhãn
width = 0.35

fig, ax = plt.subplots(1, 3, figsize=(20, 6))

# Precision
ax[0].bar(x - width/2, metrics_val[0], width, label='Validation', color='skyblue')
ax[0].bar(x + width/2, metrics_test[0], width, label='Test', color='salmon')
ax[0].set_title('Precision theo lớp')
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].legend()
ax[0].grid(True)

# Recall
ax[1].bar(x - width/2, metrics_val[1], width, label='Validation', color='skyblue')
ax[1].bar(x + width/2, metrics_test[1], width, label='Test', color='salmon')
ax[1].set_title('Recall theo lớp')
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].legend()
ax[1].grid(True)

# F1-score
ax[2].bar(x - width/2, metrics_val[2], width, label='Validation', color='skyblue')
ax[2].bar(x + width/2, metrics_test[2], width, label='Test', color='salmon')
ax[2].set_title('F1-score theo lớp')
ax[2].set_xticks(x)
ax[2].set_xticklabels(labels)
ax[2].legend()
ax[2].grid(True)

plt.suptitle('Biểu đồ Precision - Recall - F1 theo lớp (KNN)')
plt.tight_layout()
plt.savefig(f'{save_dir}/metrics_by_class.png')
plt.show()

# 11. Vẽ biểu đồ tổng hợp các chỉ số trung bình
avg_metrics = {
    'Accuracy': [accuracy_score(y_val, y_val_pred_labels), accuracy_score(y_test, y_test_pred_labels)],
    'Precision': [precision_score(y_val, y_val_pred_labels, average='macro'), precision_score(y_test, y_test_pred_labels, average='macro')],
    'Recall': [recall_score(y_val, y_val_pred_labels, average='macro'), recall_score(y_test, y_test_pred_labels, average='macro')],
    'F1-score': [f1_score(y_val, y_val_pred_labels, average='macro'), f1_score(y_test, y_test_pred_labels, average='macro')]
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