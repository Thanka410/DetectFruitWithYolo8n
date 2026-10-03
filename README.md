📌 **`fruit-detection-yolov8`** – phát hiện trái cây (táo, chuối, cam) bằng mô hình YOLOv8.

---
 🍎🍌🍊 Fruit Detection with YOLOv8

Chào mừng bạn đến với dự án Phát hiện Trái cây sử dụng YOLOv8!
Dự án này trình bày cách phát hiện và phân loại trái cây (táo, chuối, cam) theo thời gian thực bằng cách sử dụng mô hình học sâu đã được đào tạo.

---

 📌 Mục tiêu

- Phát hiện và phân loại các loại trái cây trong hình ảnh: **táo**, **chuối**, **cam**.
- Ứng dụng YOLOv8 (You Only Look Once phiên bản mới) cho việc nhận diện đối tượng nhanh và chính xác.
- Có thể áp dụng vào hệ thống kiểm tra trái cây trong dây chuyền sản xuất, nông nghiệp thông minh, hoặc hướng dẫn học AI.

---

 🎯 Ví dụ minh họa

Khi bạn đưa một ảnh chứa trái cây như bên dưới:

- Mô hình sẽ xử lý ảnh qua các lớp mạng nơ-ron.
- Xác định các vùng có chứa trái cây.
- Dự đoán loại trái cây trong mỗi vùng đó.
- Xuất ra độ tin cậy cho từng dự đoán (VD: 95% là quả táo).

<img width="1002" height="603" alt="image" src="https://github.com/user-attachments/assets/368e4239-3979-4369-9995-85ad3006b228" />


---

 🗂️ Cấu trúc thư mục

```

fruit-detection-yolov8/
├── dataset/               # Bộ dữ liệu huấn luyện
├── runs/                  # Kết quả sau khi huấn luyện (ảnh, mô hình, logs)
├── demo/                  # Hình ảnh hoặc video demo
├── detect.py              # Script chạy dự đoán
├── train.py               # Script huấn luyện mô hình
├── requirements.txt       # Thư viện cần cài đặt
└── README.md              # Mô tả dự án (file này)

````

---

 🚀 Cài đặt

 1. Clone repository
git clone https://github.com/yourusername/fruit-detection-yolov8.git
cd fruit-detection-yolov8

 2. Cài đặt thư viện

pip install -r requirements.txt

 3. Cài YOLOv8 từ Ultralytics

pip install ultralytics

---

 🧠 Mô hình

* Sử dụng YOLOv8 (có thể chọn yolov8n, yolov8s,...)
* Huấn luyện với bộ dữ liệu gồm ảnh trái cây + file nhãn (YOLO format)
* Các lớp (classes): `apple`, `banana`, `orange`

---

 🧪 Chạy thử mô hình

 Dự đoán trên hình ảnh:

yolo detect predict model=best.pt source=demo/test_fruit.jpg

 Dự đoán bằng webcam (real-time):

yolo detect predict model=best.pt source=0

---

 🏋️ Huấn luyện mô hình (nếu muốn tự train lại)

yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

* `data.yaml` cần khai báo:

```yaml
train: dataset/images/train
val: dataset/images/val

nc: 3
names: ['apple', 'banana', 'orange']
```

---

 📊 Kết quả mẫu

* Độ chính xác mAP > 90% với tập kiểm thử.
* Nhận diện rõ ràng các loại trái cây với nhiều góc độ và ánh sáng.

<p align="center">
  <img width="540" height="418" alt="image" src="https://github.com/user-attachments/assets/e2e81d6f-0f51-46b2-bd7a-70c9940612f8" />
</p>

---

 📦 requirements.txt (gợi ý nội dung)

```text
ultralytics==8.0.20
numpy
opencv-python
matplotlib
```

---

 📌 Gợi ý mở rộng

* Nhận diện thêm các loại trái cây khác: dưa hấu, xoài, nho...
* Triển khai trên Raspberry Pi / Jetson Nano
* Gắn thêm phần OCR để đọc nhãn/mã QR trên trái cây

---

 🤝 Liên hệ

Bạn có thể liên hệ với mình nếu có thắc mắc hoặc muốn góp ý cho dự án:

**Email:** [hiepbt17@gmail.com]

---
