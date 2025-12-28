# Auto Bug Detection using Graph Transformer

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Academic%20Project-blue?style=flat-square)

**Đồ án môn học: Phát hiện lỗ hổng bảo mật tự động trong mã nguồn C/C++ sử dụng Graph Transformer**

---

## Giới thiệu

Dự án này xây dựng một hệ thống phát hiện lỗ hổng bảo mật tự động trong mã nguồn C/C++ sử dụng mô hình Graph Transformer. Hệ thống học biểu diễn từ hai view đồ thị:

| View | Mô tả |
|------|-------|
| **AST** (Abstract Syntax Tree) | Cây cú pháp trừu tượng - biểu diễn cấu trúc ngữ pháp của mã nguồn |
| **CDFG** (Control Data Flow Graph) | Đồ thị luồng điều khiển và dữ liệu - biểu diễn luồng thực thi và phụ thuộc dữ liệu |

### Mục tiêu

- Phát hiện các lỗ hổng bảo mật phổ biến (CWE-77, Command Injection, Buffer Overflow...)
- Ứng dụng Deep Learning (Graph Neural Networks) vào bài toán phân tích mã nguồn
- Kết hợp thông tin từ nhiều biểu diễn đồ thị để tăng độ chính xác

---

## Kiến trúc Model

```
                         Input: Source Code (C/C++)
                                    |
                   +----------------+----------------+
                   |                                 |
                   v                                 v
            +-----------+                     +-----------+
            |    AST    |                     |   CDFG    |
            |   Graph   |                     |   Graph   |
            +-----------+                     +-----------+
                   |                                 |
                   v                                 v
            +-----------+                     +-----------+
            |   Node    |                     |   Node    |
            |  Feature  |                     |  Feature  |
            |  Encoder  |                     |  Encoder  |
            |   (MLP)   |                     |   (MLP)   |
            +-----------+                     +-----------+
                   |                                 |
                   v                                 v
            +-----------+                     +-----------+
            |   Graph   |                     |   Graph   |
            |Transformer|                     |Transformer|
            | (4 layers)|                     | (4 layers)|
            +-----------+                     +-----------+
            | Multi-head|                     | Multi-head|
            | Attention |                     | Attention |
            | Edge Bias |                     | Edge Bias |
            | FFN + LN  |                     | FFN + LN  |
            +-----------+                     +-----------+
                   |                                 |
                   v                                 v
            +-----------+                     +-----------+
            |  Weighted |                     |  Weighted |
            |    Sum    |                     |    Sum    |
            |  Readout  |                     |  Readout  |
            +-----------+                     +-----------+
                   |                                 |
                   +----------------+----------------+
                                    |
                                    v
                             +-----------+
                             |   Concat  |
                             |[AST+CDFG] |
                             +-----------+
                                    |
                                    v
                             +-----------+
                             |Classifier |
                             |   (MLP)   |
                             +-----------+
                                    |
                                    v
                             +-----------+
                             |  Output:  |
                             |Vulnerable |
                             |  / Safe   |
                             +-----------+
```

---

## Cấu trúc Project

```
Auto_Bug_Detection/
    |
    |-- Train_Model.py          # Script huấn luyện model
    |-- Detector.py             # Script phát hiện lỗ hổng
    |-- Requirements.txt        # Các thư viện cần thiết
    |-- README.md               # Tài liệu hướng dẫn
    |
    |-- TIFS_Data/              # Dataset (tải từ Drive)
    |       |-- SARD/           # Dataset SARD gốc
    |       |-- SARD_after/     # Dữ liệu đã tiền xử lý
    |       |-- graphs/         # Các đồ thị đã sinh
    |       |-- preprocess_sard.py
    |
    |-- Trained_Model/          # Các checkpoint của model
    |       |-- Auto_Bug_Detector.pt
    |       |-- Auto_Bug_Detector_best.pt
```

---

## Dataset

**Lưu ý:** Thư mục `TIFS_Data/` chứa dataset SARD rất nặng (~2GB+) nên không được commit lên Git.

### Tải Dataset

**Google Drive:** [Nhấn để tải TIFS_Data](https://drive.google.com/file/d/1pF8ca8zqUap4bv1bYMPZlBpOuzOTWJr6/view?usp=drive_link)

Sau khi tải về:
1. Giải nén file (nếu là .zip)
2. Đặt thư mục `TIFS_Data/` vào thư mục gốc của project
3. Cấu trúc đúng: `Auto_Bug_Detection/TIFS_Data/SARD/...`

### Thông tin Dataset

| Thông tin | Giá trị |
|-----------|---------|
| Nguồn | SARD (Software Assurance Reference Dataset) |
| Ngôn ngữ | C/C++ |
| Loại lỗ hổng | CWE-77 (Command Injection), CWE-119 (Buffer Overflow)... |
| Số mẫu | ~10,000+ hàm |
| Định dạng | Source code -> AST/CDFG graphs -> JSONL |

---

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/X181125/Auto_Bug_Detection.git
cd Auto_Bug_Detection
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Cài đặt các thư viện

```bash
pip install -r Requirements.txt
```

### 4. Tải dataset

Tải `TIFS_Data/` từ Google Drive (link ở trên) và đặt vào thư mục project.

---

## Huấn luyện Model

### Bắt đầu nhanh

```bash
python Train_Model.py
```

### Với tham số tùy chỉnh

```bash
python Train_Model.py --epochs 50 --lr 1e-3 --state-dim 128 --num-layers 4 --num-heads 4 --patience 10
```

### Các tham số huấn luyện

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--epochs` | 50 | Số epochs huấn luyện |
| `--lr` | 1e-3 | Tốc độ học (learning rate) |
| `--weight-decay` | 1e-4 | Weight decay (L2 regularization) |
| `--state-dim` | 128 | Số chiều ẩn (hidden dimension) |
| `--num-layers` | 4 | Số lớp Graph Transformer |
| `--num-heads` | 4 | Số attention heads |
| `--max-nodes-per-batch` | 8000 | Số node tối đa mỗi batch |
| `--patience` | 10 | Số epochs chờ trước khi dừng sớm |

### Theo dõi quá trình huấn luyện với TensorBoard

```bash
tensorboard --logdir logs
```

---

## Phát hiện lỗ hổng (Inference)

### Sử dụng file code mẫu

```bash
python Detector.py --source badExample.c
```

### Kết quả mẫu

```
Ket qua Phat hien Lo hong
================================
File: badExample.c
Du doan: CO LO HONG
Do tin cay: 87.3%
================================
```

---

## Kết quả thực nghiệm

| Chỉ số | Dataset CWE-77 |
|--------|----------------|
| Accuracy | 85-90% |
| Precision | 80-85% |
| Recall | 80-90% |
| F1-Score | 80-87% |

Kết quả có thể thay đổi tùy thuộc vào hyperparameters và quá trình tiền xử lý dữ liệu.

---

## Các thành phần chính

### 1. Code2Graph.py
Chuyển đổi mã nguồn C/C++ thành đồ thị AST và CDFG.

### 2. Train_Model.py
- `GraphTransformerLayer`: Multi-head self-attention với edge-type embedding
- `GraphTransformerEncoder`: Stack của N lớp GraphTransformerLayer
- `WeightedSumReadout`: Attention-based pooling
- `VulnDetectorGraphTransformer`: Model chính kết hợp 2 view

### 3. Detector.py
Script inference để phát hiện lỗ hổng từ file mã nguồn.

---

## Điều chỉnh Hyperparameter

### Dataset nhỏ (< 1000 mẫu)
```bash
--state-dim 64 --num-layers 2 --num-heads 2 --epochs 30
```

### Dataset vừa (1000-10000 mẫu)
```bash
--state-dim 128 --num-layers 4 --num-heads 4 --epochs 50
```

### Dataset lớn (> 10000 mẫu)
```bash
--state-dim 256 --num-layers 6 --num-heads 8 --epochs 100
```

---

## Xử lý sự cố

| Vấn đề | Giải pháp |
|--------|-----------|
| Hết bộ nhớ (Out of Memory) | Giảm `--max-nodes-per-batch`, `--state-dim`, `--num-layers` |
| Overfitting | Tăng `--weight-decay`, giảm `--epochs` |
| Underfitting | Tăng `--state-dim`, `--num-layers`, `--epochs` |
| Không tìm thấy dataset | Kiểm tra đã tải `TIFS_Data/` từ Drive chưa |

---

## Tài liệu tham khảo

1. Graph Transformer Networks - Yun et al., NeurIPS 2019
2. Devign: Effective Vulnerability Identification - Zhou et al., NeurIPS 2019
3. FUNDED: Flow-based Vulnerability Detection - Wang et al., ICSE 2020
4. SARD Dataset - NIST Software Assurance Reference Dataset

---

## Thông tin đồ án

| Thông tin | Chi tiết |
|-----------|----------|
| Môn học | An toàn Thông tin / Machine Learning |
| Trường | Đại học Công nghệ Thông tin - ĐHQG TPHCM (UIT) |
| Sinh viên | Nguyễn Đình Hưng |
| MSSV | 23520564 |

---

## Giấy phép

MIT License - Sử dụng cho mục đích học tập và nghiên cứu.
