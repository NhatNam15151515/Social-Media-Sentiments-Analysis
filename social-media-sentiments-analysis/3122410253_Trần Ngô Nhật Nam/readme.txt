📄 Tập tin báo cáo: 3122410253_Trần Ngô Nhật Nam.docx
🔍 Chủ đề: ứng dụng thuật toán svm phân loại cảm xúc dựa trên nội dung bài đăng mạng xã hội (Social Media Sentiment Analysis)

📁 Tệp dữ liệu sử dụng:

sentimentdataset.csv
→ Chứa tập dữ liệu gốc gồm 191 nhãn cảm xúc khác nhau được gán cho các bài đăng trên mạng xã hội.
Đây là dữ liệu thô, được sử dụng để phân tích, thống kê và tiền xử lý ban đầu.

Split_Data.ipynb
→ Notebook phụ trợ có nhiệm vụ chuyển đổi và gom nhóm các nhãn cảm xúc gốc trong sentimentdataset.csv thành 3 nhóm cảm xúc chính:

Tích cực (Positive)

Tiêu cực (Negative)

Trung lập (Neutral)

Việc gom nhóm này được thực hiện nhằm đơn giản hóa bài toán phân loại, giúp mô hình học máy hoạt động hiệu quả và ổn định hơn.

sentimentgroups.csv
→ Là phiên bản dữ liệu đã được chuẩn hóa sau khi xử lý bởi Split_Data.ipynb.
Chỉ bao gồm 3 nhãn cảm xúc chính như trên, và là dữ liệu đầu vào chính cho quá trình vector hóa và huấn luyện mô hình.

Social Media Sentiments Analysis - New Data.ipynb
→ Notebook chính, thực hiện toàn bộ quy trình khai phá dữ liệu: từ tiền xử lý văn bản, trực quan hóa dữ liệu, vector hóa TF-IDF,
huấn luyện và đánh giá các mô hình học máy như Support Vector Machine, XGBoost, Naive Bayes, Logistic Regression.