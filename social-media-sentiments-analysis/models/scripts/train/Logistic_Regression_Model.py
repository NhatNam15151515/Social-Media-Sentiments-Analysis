import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

# Đường dẫn lưu model
save_dir = f'../../train/Logistic_Regression'
os.makedirs(save_dir, exist_ok=True)

# 1. Đọc dữ liệu
data = pd.read_csv('../../../data/external/sentimentgroups.csv')
print("Kích thước dữ liệu:", data.shape)
print(data.info())
print(data['Sentiment'].value_counts())

# 2. Tiền xử lý
data = data.dropna(subset=['Text', 'Sentiment'])
data = data[data['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Sentiment'])

tfidf_vectorizer = TfidfVectorizer(
    max_features=1500,
    stop_words='english',
    ngram_range=(1, 1)
)
X = tfidf_vectorizer.fit_transform(data['Text'])

# 3. Chia tập train / val / test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                  test_size=0.176, random_state=42, stratify=y_train_val)

print(f"Tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"Tập validation: {X_val.shape[0]} mẫu")
print(f"Tập kiểm tra: {X_test.shape[0]} mẫu")

# 4. Huấn luyện mô hình Logistic Regression
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

# 5. Đánh giá mô hình
def evaluate_model(model, X, y_true, label, save_conf_path):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    print(f"\nKết quả trên tập {label}:")
    print("Accuracy:", acc)
    print("\nBáo cáo phân loại:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title(f'Ma trận nhầm lẫn - {label}')
    plt.savefig(save_conf_path)
    plt.close()

    return y_pred

y_val_pred = evaluate_model(model, X_val, y_val, "Validation", f"{save_dir}/confusion_matrix_LR_val.png")
y_test_pred = evaluate_model(model, X_test, y_test, "Test", f"{save_dir}/confusion_matrix_LR_test.png")

# 6. Lưu model và các thành phần
pickle.dump(model, open(f"{save_dir}/sentiment.pkl", 'wb'))
pickle.dump(tfidf_vectorizer, open(f"{save_dir}/vectorizer.pkl", 'wb'))
pickle.dump(label_encoder, open(f"{save_dir}/label_encoder.pkl", 'wb'))

# 7. Vẽ biểu đồ các chỉ số Precision, Recall, F1 theo từng lớp
def plot_metrics_by_class(y_true_val, y_pred_val, y_true_test, y_pred_test):
    labels = label_encoder.classes_
    x = np.arange(len(labels))

    metrics_val = precision_recall_fscore_support(y_true_val, y_pred_val, average=None, labels=range(len(labels)))
    metrics_test = precision_recall_fscore_support(y_true_test, y_pred_test, average=None, labels=range(len(labels)))

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    width = 0.35

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

    plt.suptitle('Biểu đồ Precision - Recall - F1 theo lớp (Logistic Regression)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_by_class_LR.png')
    plt.show()

plot_metrics_by_class(y_val, y_val_pred, y_test, y_test_pred)

# 8. Biểu đồ tổng hợp chỉ số trung bình
avg_metrics = {
    'Accuracy': [accuracy_score(y_val, y_val_pred), accuracy_score(y_test, y_test_pred)],
    'Precision': [precision_score(y_val, y_val_pred, average='macro'), precision_score(y_test, y_test_pred, average='macro')],
    'Recall': [recall_score(y_val, y_val_pred, average='macro'), recall_score(y_test, y_test_pred, average='macro')],
    'F1-score': [f1_score(y_val, y_val_pred, average='macro'), f1_score(y_test, y_test_pred, average='macro')]
}

metrics_df = pd.DataFrame(avg_metrics, index=['Validation', 'Test'])
metrics_df.plot(kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black')
plt.title("So sánh các chỉ số trung bình giữa Validation và Test (Logistic Regression)")
plt.ylabel("Giá trị")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{save_dir}/avg_metrics_comparison_LR.png')
plt.show()
