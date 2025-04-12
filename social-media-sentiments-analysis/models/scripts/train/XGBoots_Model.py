import pandas as pd
import numpy as np
import os
import string
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# Các từ có giá trị thấp
low_value_words = {'new', 'like', 'feeling', 'day', 'world'}


save_dir = f'../../train_archive/XGBoost'
# 1. Đọc dữ liệu từ file CSV đã được xử lý sentiment
data = pd.read_csv('../../../data/external/sentimentgroups.csv')

# Kiểm tra dữ liệu
print("Kích thước dữ liệu:", data.shape)
print("\nThông tin dữ liệu:")
print(data.info())
print("\nỨng với mỗi nhóm sentiment có bao nhiêu mẫu:")
print(data['Sentiment'].value_counts())

# 2. Tiền xử lý dữ liệu
# Loại bỏ các hàng có giá trị null trong cột Text hoặc Sentiment
data = data.dropna(subset=['Text', 'Sentiment'])

# Loại bỏ các sentiment không thuộc 3 nhóm chính
data = data[data['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
# Tiền xử lý văn bản
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
# Mã hóa nhãn Sentiment thành số
data['Clean_Text'] = data['Text'].apply(preprocess_text)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Sentiment'])

# 3. Trích xuất đặc trưng - Giảm số lượng đặc trưng
# vectorizer = TfidfVectorizer(
#     max_features=1500,  # Giảm xuống từ 5000
#     stop_words='english',
#     ngram_range=(1, 1)   # Chỉ sử dụng unigram thay vì (1,2)
# )
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
X = vectorizer.fit_transform(data['Clean_Text'])

# 4. Chia dữ liệu thành tập train, validation và test (tỷ lệ 7:1.5:1.5)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                  test_size=0.176, random_state=42, stratify=y_train_val)

print(f"Tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"Tập validation: {X_val.shape[0]} mẫu")
print(f"Tập kiểm tra: {X_test.shape[0]} mẫu")

# 5. Chuyển dữ liệu sang DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# 6. Huấn luyện mô hình với tham số đã tối ưu
params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoder.classes_),
    'eval_metric': 'mlogloss',
    'colsample_bytree': 0.8,
    'learning_rate': 0.5,
    'max_depth': 10,
    'min_child_weight': 1,
    'n_estimators': 300,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'subsample': 0.8,
    'seed': 42
}

watchlist = [(dtrain, 'train'), (dval, 'validation')]
num_rounds = 150

# Lưu kết quả huấn luyện để vẽ biểu đồ sau này
evals_result = {}

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    evals=watchlist,
    early_stopping_rounds=10,
    evals_result=evals_result,
    verbose_eval=True
)

# 7. Đánh giá mô hình
y_val_pred = model.predict(dval)
y_val_pred_labels = np.argmax(y_val_pred, axis=1)
print("\nKết quả trên tập validation:")

print("\nBáo cáo phân loại:")
print(classification_report(y_val, y_val_pred_labels, target_names=label_encoder.classes_))

y_test_pred = model.predict(dtest)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)
print("\nKết quả trên tập test:")

print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_test_pred_labels, target_names=label_encoder.classes_))
print("Accuracy validation:", accuracy_score(y_val, y_val_pred_labels))
print("Accuracy test:", accuracy_score(y_test, y_test_pred_labels))
# 8. Tạo thư mục lưu model
os.makedirs(save_dir, exist_ok=True)

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_test_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn XGBoost')
plt.savefig(f'{save_dir}/confusion_matrix_XGB.png')

# 9. Lưu mô hình và các thành phần cần thiết
model.save_model(f'{save_dir}/sentiment.model')
pickle.dump(vectorizer, open(f'{save_dir}/vectorizer.pkl', 'wb'))
pickle.dump(label_encoder, open(f'{save_dir}/label_encoder.pkl', 'wb'))

# Vẽ biểu đồ train vs validation log loss
epochs = len(evals_result['train']['mlogloss'])
x_axis = range(epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, evals_result['train']['mlogloss'], label='Train')
plt.plot(x_axis, evals_result['validation']['mlogloss'], label='Validation')
plt.xlabel('Vòng lặp (Boosting round)')
plt.ylabel('Log Loss')
plt.title('Biểu đồ huấn luyện XGBoost - Log Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{save_dir}/training_curve.png')

# 10. Vẽ biểu đồ F1-score, Precision, Recall theo từng lớp
from sklearn.metrics import precision_recall_fscore_support

# Tính chỉ số cho tập validation và test
metrics_val = precision_recall_fscore_support(y_val, y_val_pred_labels, average=None, labels=range(len(label_encoder.classes_)))
metrics_test = precision_recall_fscore_support(y_test, y_test_pred_labels, average=None, labels=range(len(label_encoder.classes_)))

labels = label_encoder.classes_
x = np.arange(len(labels))  # vị trí nhãn

# Vẽ biểu đồ so sánh từng chỉ số theo lớp
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
width = 0.35

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

plt.suptitle('Biểu đồ Precision - Recall - F1 theo lớp (XGBoost)')
plt.tight_layout()
plt.savefig(f'{save_dir}/metrics_by_class.png')


# 11. Vẽ biểu đồ tổng hợp các chỉ số trung bình
from sklearn.metrics import precision_score, recall_score, f1_score

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

