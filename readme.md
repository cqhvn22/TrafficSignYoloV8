# TrafficSignYoloV8

Hệ thống phát hiện và phân loại biển báo giao thông Việt Nam sử dụng **YOLOv8s**, kết hợp **EasyOCR** để tự động đọc giá trị số trên biển báo giới hạn tốc độ. BTL thuộc học phần **Trí tuệ nhân tạo** – Trường Công nghệ thông tin và Truyền thông, Đại học Công nghiệp Hà Nội.

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Bộ dữ liệu](#bộ-dữ-liệu)
- [Mô hình & Kết quả huấn luyện](#mô-hình--kết-quả-huấn-luyện)
- [Pipeline nhận dạng biển báo tốc độ (YOLOv8s + EasyOCR)](#pipeline-nhận-dạng-biển-báo-tốc-độ-yolov8s--easyocr)
- [Hạn chế](#hạn-chế)
- [Hướng phát triển](#hướng-phát-triển)
- [Thành viên thực hiện](#thành-viên-thực-hiện)

## Giới thiệu

Đề tài xây dựng một hệ thống phát hiện và phân loại biển báo giao thông Việt Nam dựa trên mô hình **YOLOv8s**, kết hợp các kỹ thuật xử lý và cân bằng dữ liệu để khắc phục tình trạng mất cân bằng lớp. Hệ thống còn tích hợp thư viện **EasyOCR** để nhận dạng giá trị số trên biển báo giới hạn tốc độ (nhóm P.127), nâng hệ thống từ mức phân loại biển báo lên mức khai thác nội dung thông tin từ biển báo.

Toàn bộ quá trình xử lý dữ liệu và huấn luyện được thực hiện trên **Google Colab** (GPU Tesla T4/P100) với framework **Ultralytics YOLOv8**. Sau khi huấn luyện, mô hình (`best.pt`) được triển khai chạy suy luận cục bộ thông qua một ứng dụng desktop xây dựng bằng **Tkinter**.

## Tính năng

- Phát hiện và phân loại **21 lớp** biển báo giao thông Việt Nam (biển cấm, biển nguy hiểm, biển hiệu lệnh, biển chỉ dẫn) theo QCVN 41:2019/BGTVT.
- Quy trình xử lý dữ liệu đầy đủ: gom nhóm nhãn, lọc lớp ít mẫu, cân bằng dữ liệu (augmentation + undersampling).
- Tích hợp **EasyOCR** để đọc giá trị tốc độ cụ thể trên biển báo giới hạn tốc độ (P.127).
- Ứng dụng demo desktop (Tkinter) hỗ trợ 3 chế độ: **ảnh tĩnh**, **video tải lên**, **camera thời gian thực**.
- Theo dõi đối tượng xuyên suốt video/camera bằng **ByteTrack**, làm mượt bằng **EMA**, ngoại suy khi mất dấu (ghost tracking).

## Cấu trúc repo

```
TrafficSignYoloV8/
├── Demo Application/   # Ứng dụng desktop (Tkinter) - chạy suy luận trên ảnh/video/camera
├── Train.ipynb          # Notebook huấn luyện mô hình YOLOv8s trên Google Colab
└── .gitattributes
```

## Bộ dữ liệu

- **Nguồn**: bộ dữ liệu công khai [Vietnamese Traffic Signs](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs) trên Kaggle, do nhóm Data Palpating Team thu thập.
- Được clone về môi trường thực thi qua repository [cqhvn22/dataset](https://github.com/cqhvn22/dataset) để đồng bộ nhanh với Google Colab.
- Định dạng nhãn: **YOLO format** (`<class_id> <x_center> <y_center> <width> <height>`, tọa độ chuẩn hóa [0, 1]).

### Tiền xử lý & cân bằng dữ liệu

1. **Gom nhóm nhãn**: hợp nhất các biến thể `P.127*<giá_trị>` (ví dụ P.127*20, P.127*50...) thành một lớp duy nhất `P.127`.
2. **Lọc lớp thiểu số**: loại bỏ các lớp có dưới `MIN_SAMPLES = 100` mẫu.
3. **Cân bằng dữ liệu** với mục tiêu `TARGET = 400` mẫu/lớp:
   - **Augmentation** (OpenCV) cho lớp có dưới `LOWER_BOUND = 300` mẫu: xoay ±15°, điều chỉnh độ sáng (0.6–1.4x), nhiễu Gaussian, làm mờ Gaussian.
   - **Undersampling** ngẫu nhiên cho lớp vượt `UPPER_BOUND = 500` mẫu.
4. **Chia tập Train/Validation**: tỷ lệ 80/20, ngẫu nhiên với seed cố định, sinh file `data.yaml` cho Ultralytics.

## Mô hình & Kết quả huấn luyện

### Lựa chọn mô hình

Sử dụng **YOLOv8s** (small) — cân bằng giữa tốc độ và độ chính xác, phù hợp với GPU Tesla T4/P100 trên Google Colab và quy mô bài toán (21 lớp). Khởi tạo từ trọng số pretrained `yolov8s.pt` (transfer learning từ COCO).

### Siêu tham số huấn luyện chính

| Siêu tham số | Giá trị |
|---|---|
| Epochs | 50 (Early Stopping, patience = 30) |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate (lr0 → lrf) | 0.001 → 0.00001 (cosine annealing) |
| Weight decay | 0.0005 |
| AMP (FP16) | Bật |
| Mosaic augmentation | 1.0 (100%) |

### Kết quả

| Chỉ số | Giá trị |
|---|---|
| mAP@0.5 | **0.989** |
| mAP@0.5:0.95 | ~0.78 |
| Precision | ~0.95–0.97 |
| Recall | ~0.98 |
| F1-Score (tốt nhất) | **0.98** tại confidence = 0.539 |

Mô hình hội tụ ổn định, không có dấu hiệu overfitting rõ rệt; hiệu năng đồng đều giữa các lớp nhờ chiến lược cân bằng dữ liệu.

Trọng số được lưu dưới dạng `best.pt` (tốt nhất trên validation) và `last.pt` (epoch cuối), kèm checkpoint định kỳ mỗi 10 epoch.

## Pipeline nhận dạng biển báo tốc độ (YOLOv8s + EasyOCR)

Hệ thống mở rộng theo pipeline **hai giai đoạn**:

1. **Giai đoạn 1 – Phát hiện (YOLOv8s)**: phát hiện biển báo với `conf = 0.35`, `iou = 0.45`, `imgsz = 640`; lọc ra các phát hiện thuộc nhóm biển báo tốc độ (P.127).
2. **Giai đoạn 2 – Nhận dạng ký tự (EasyOCR)**: cắt vùng ảnh biển báo (mở rộng biên 5–10%), tiền xử lý tương phản/độ sáng, sau đó gọi:

```python
reader.readtext(cropped_image, allowlist='0123456789', detail=1)
```

EasyOCR sử dụng **CRAFT** (phát hiện vùng văn bản) và **CRNN** (nhận dạng chuỗi ký tự), giới hạn tập ký tự chỉ gồm chữ số để giảm nhận dạng nhầm.

Kết quả cuối cùng: ảnh hiển thị bounding box biển báo kèm tên lớp, độ tin cậy và giá trị tốc độ đọc được (ví dụ: `P.127 0.92 | 60 km/h`).

### Các lỗi OCR thường gặp

- Biển báo bị ngược sáng / che bóng → giảm tương phản chữ số.
- Góc chụp xiên lớn → méo phối cảnh, sai lệch hình dạng ký tự.
- Vùng cắt bao gồm viền/nền dư thừa → tăng nhiễu đầu vào OCR.

## Hạn chế

- Huấn luyện và đánh giá chỉ dựa trên bộ dữ liệu Kaggle, chưa có dữ liệu thực địa tự thu thập.
- Mới hỗ trợ 21 lớp biển báo, còn cách xa toàn bộ hệ thống biển báo theo QCVN 41:2019/BGTVT.
- Độ chính xác OCR giảm khi biển báo ngược sáng, che khuất hoặc chụp ở góc xiên lớn (chưa có chuẩn hóa phối cảnh / CLAHE).
- Chưa tối ưu cho thiết bị nhúng (Jetson Nano, Raspberry Pi), chưa thử nghiệm thực tế trên phương tiện giao thông.

## Hướng phát triển

- Thu thập thêm dữ liệu thực địa, mở rộng số lớp biển báo.
- Thử nghiệm các phiên bản YOLO mới hơn.
- Bổ sung chuẩn hóa phối cảnh (perspective normalization) và CLAHE cho pipeline OCR.
- Mở rộng nhận dạng sang các biển báo chứa thông tin văn bản khác (chiều cao, tải trọng, khoảng cách, thời gian hiệu lực).
- Tối ưu hóa mô hình cho thiết bị nhúng và tích hợp vào hệ thống ADAS.


## Thành viên thực hiện

BTL học phần **Trí tuệ nhân tạo (20252IT6094002)** (6/2026).

| Họ và tên | Mã sinh viên | Lớp |
|---|---|---|
| Trương Hồng An | 2024604813 | KHMT01 K19 |
| Hoàng Văn Anh | 2024605960 | KHMT01 K19 |
| Trần Hải Đăng | 2024606419 | KHMT01 K19 |
| Chu Quang Hưng | 2024604669 | KHMT01 K19 |
| Lại Thế Sơn | 2024605014 | KHMT01 K19 |

**Giảng viên hướng dẫn**: TS. Trần Thanh Huân
