# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Đường dẫn tới file CSV cần tải
file_path = "sentimentdataset.csv"

# Tải dữ liệu vào DataFrame
data = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "kashishparmar02/social-media-sentiments-analysis-dataset",
    file_path
)

# Hiển thị 5 dòng đầu tiên
print(data.head())

# Đường dẫn lưu file
saving_path = r"..\data\external"
file_name = os.path.join(saving_path, "sentimentdataset.csv")

# Kiểm tra xem file đã tồn tại chưa
if not os.path.exists(file_name):
    # Lưu file nếu chưa tồn tại
    os.makedirs(saving_path, exist_ok=True)  # Tạo thư mục nếu chưa có
    data.to_csv(file_name, index=False)
    print(f"Bộ dữ liệu đã được lưu tại: {file_name}")
else:
    print(f"File đã tồn tại tại: {file_name}, không lưu lại.")
