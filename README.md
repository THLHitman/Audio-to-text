# Vietnamese Speech-to-Text Web App

Ứng dụng web sử dụng mô hình Wav2Vec2 để chuyển đổi âm thanh tiếng Việt sang văn bản, hỗ trợ nhiều định dạng file và có giao diện thân thiện với người dùng.

## 📌 Tính năng

* ✅ Tải lên file âm thanh (MP3, WAV, FLAC, M4A, OGG, WMA)
* ✅ Tự động chuyển đổi âm thanh sang định dạng WAV 16kHz
* ✅ Phân đoạn âm thanh dài thành từng phần để xử lý
* ✅ Sử dụng mô hình Wav2Vec2 cho tiếng Việt để nhận dạng giọng nói
* ✅ Hiển thị và sao chép kết quả văn bản
* ✅ Kiểm tra tình trạng tải mô hình thời gian thực
* ✅ Giao diện đẹp bằng Tailwind CSS

## 📂 Cấu trúc dự án

```
├── app.py                   # Flask backend chính
├── index.html               # Giao diện web phía client
├── wav2vec2-vi/             # Thư mục chứa mô hình đã fine-tuned
│   ├── config.json
│   ├── model_handling.py    # Định nghĩa lớp model
│   ├── pytorch_model.bin
│   └── preprocessor_config.json
├── uploads/                 # Lưu file người dùng tải lên tạm thời
└── README.md
```

## 🚀 Cài đặt và chạy

### Yêu cầu

* Python 3.8+
* CUDA GPU (khuyến khích để tăng tốc inference)
* pip

### Cài đặt thư viện

pip install flask torchaudio transformers

### Tải model

Chạy file dow_wav2vec2-vi.ipynb

### Chạy server

python app.py

Ứng dụng sẽ chạy tại `http://localhost:5000`

## 💡 Cách sử dụng

1. Mở trình duyệt và truy cập địa chỉ `http://localhost:5000`
2. Tải lên file âm thanh tiếng Việt
3. Chờ quá trình nhận dạng hoàn tất
4. Xem và sao chép kết quả văn bản

## 🧠 Mô hình Wav2Vec2

Ứng dụng sử dụng mô hình Wav2Vec2 đã fine-tuned cho tiếng Việt. Mô hình được tải từ thư mục `./wav2vec2-vi` với kiến trúc CTC (Connectionist Temporal Classification) để chuyển đổi chuỗi âm thanh thành văn bản.

## 🔐 Giới hạn

* Dung lượng tối đa mỗi file: **100MB**
* File âm thanh được xử lý theo từng đoạn **15 giây** để tối ưu bộ nhớ GPU

## 📄 Bản quyền

Phát triển bởi \[Nhóm thực tập sinh Khoa CNTT], sử dụng các công cụ mã nguồn mở từ Hugging Face, PyTorch và Torchaudio.
