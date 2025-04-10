import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
# 1. Đọc dữ liệu từ file CSV đã được xử lý sentiment
data = pd.read_csv('../data/external/sentimentgroups.csv')

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
data = data[data['Sentiment'].isin(['positive', 'negative', 'neutral'])]

# Mã hóa nhãn Sentiment thành số
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Sentiment'])

# 3. Trích xuất đặc trưng - Giảm số lượng đặc trưng
tfidf_vectorizer = TfidfVectorizer(
    max_features=1500,  # Giảm xuống từ 5000
    stop_words='english',
    ngram_range=(1, 1)   # Chỉ sử dụng unigram thay vì (1,2)
)
X = tfidf_vectorizer.fit_transform(data['Text'])
# 4. Chia dữ liệu thành tập train, validation và test (tỷ lệ 7:1.5:1.5)
# Đầu tiên chia thành train + val và test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Sau đó chia train + val thành train và val
# train = 7/(7+1.5) = 0.824 của tập train_val
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                test_size=0.176, random_state=42, stratify=y_train_val)

print(f"Tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"Tập validation: {X_val.shape[0]} mẫu")
print(f"Tập kiểm tra: {X_test.shape[0]} mẫu")

# 6. Huấn luyện mô hình với xgboost.train() thay vì XGBClassifier
# Chuyển đổi dữ liệu sang định dạng DMatrix của XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# 6. Huấn luyện mô hình với xgboost.train() thay vì XGBClassifier
import xgboost as xgb

# Chuyển đổi dữ liệu sang định dạng DMatrix của XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Tham số từ Grid Search đã thực hiện
params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoder.classes_),
    'eval_metric': 'mlogloss',
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'max_depth': 10,
    'min_child_weight': 1,
    'n_estimators': 300,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'subsample': 0.8,
    'seed': 42
}

# Tạo watchlist để theo dõi hiệu suất
watchlist = [(dtrain, 'train'), (dval, 'validation')]

# Huấn luyện mô hình với early stopping
num_rounds = 150 # Tương đương với n_estimators
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    evals=watchlist,
    early_stopping_rounds=10,
    verbose_eval=True
)

# 7. Đánh giá mô hình
# Dự đoán trên tập validation
y_val_pred = model.predict(dval)
y_val_pred_labels = np.argmax(y_val_pred, axis=1)
print("\nKết quả trên tập validation:")
print("Accuracy:", accuracy_score(y_val, y_val_pred_labels))
print("\nBáo cáo phân loại:")
print(classification_report(y_val, y_val_pred_labels, target_names=label_encoder.classes_))

# Dự đoán trên tập test
y_test_pred = model.predict(dtest)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)
print("\nKết quả trên tập test:")
print("Accuracy:", accuracy_score(y_test, y_test_pred_labels))
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_test_pred_labels, target_names=label_encoder.classes_))

# 8. Vẽ ma trận nhầm lẫn
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_test_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn XGBoost')
plt.savefig('confusion_matrix_XGB.png')

# 9. Lưu mô hình và các thành phần cần thiết
model.save_model('sentiment_XGB.model')
pickle.dump(tfidf_vectorizer, open('vectorizer_XGB.pkl', 'wb'))
pickle.dump(label_encoder, open('label_encoder_XGB.pkl', 'wb'))
# Lấy kết quả huấn luyện
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
plt.savefig('training_curve.png')
plt.show()
