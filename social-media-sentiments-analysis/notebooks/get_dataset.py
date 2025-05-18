import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Khởi tạo API (cần có file kaggle.json trong thư mục ~/.kaggle)
api = KaggleApi()
api.authenticate()

# Đường dẫn lưu file
data_saving_path = "../data/external"
file_name = "sentimentdataset.csv"
full_path = os.path.join(data_saving_path, file_name)

# Tạo thư mục nếu chưa tồn tại
os.makedirs(data_saving_path, exist_ok=True)

# Kiểm tra xem file đã tồn tại chưa
if not os.path.exists(full_path):
    print(f"Đang tải tập dữ liệu...")
    # Tải tập dữ liệu từ Kaggle
    api.dataset_download_files('kashishparmar02/social-media-sentiments-analysis-dataset', 
                              path=data_saving_path, unzip=True)
    print(f"Đã tải xong tập dữ liệu vào {data_saving_path}")
else:
    print(f"File đã tồn tại tại: {full_path}, không tải lại.")

# # Đọc dữ liệu
# try:
#     data = pd.read_csv(full_path)
#     print("Đã đọc dữ liệu thành công:")
#     print(data.head())
# except Exception as e:
#     print(f"Lỗi khi đọc file: {e}")