# 🛡️ Auto Bug Detection using Graph Transformer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Status-Academic%20Project-blue?style=for-the-badge" alt="Status">
</p>

<p align="center">
  <strong>Đồ án môn học: Phát hiện lỗ hổng bảo mật tự động trong mã nguồn C/C++ sử dụng Graph Transformer</strong>
</p>

---

## 📋 Giới thiệu

Dự án này xây dựng một hệ thống **phát hiện lỗ hổng bảo mật tự động** trong mã nguồn C/C++ sử dụng mô hình **Graph Transformer**. Hệ thống học biểu diễn từ hai view đồ thị:

| View | Mô tả |
|------|-------|
| **AST** (Abstract Syntax Tree) | Cây cú pháp trừu tượng - biểu diễn cấu trúc ngữ pháp của mã nguồn |
| **CDFG** (Control Data Flow Graph) | Đồ thị luồng điều khiển và dữ liệu - biểu diễn luồng thực thi và phụ thuộc dữ liệu |

### 🎯 Mục tiêu

- Phát hiện các lỗ hổng bảo mật phổ biến (CWE-77, Command Injection, Buffer Overflow...)
- Ứng dụng Deep Learning (Graph Neural Networks) vào bài toán phân tích mã nguồn
- Kết hợp thông tin từ nhiều biểu diễn đồ thị để tăng độ chính xác

---

## 🏗️ Kiến trúc Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Source Code (C/C++)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────┴─────────────────┐
    ▼                                   ▼
┌───────────┐                     ┌───────────┐
│    AST    │                     │   CDFG    │
│   Graph   │                     │   Graph   │
└─────┬─────┘                     └─────┬─────┘
      ▼                                 ▼
┌───────────────────┐           ┌───────────────────┐
│ Node Feature      │           │ Node Feature      │
│ Encoder (MLP)     │           │ Encoder (MLP)     │
└─────────┬─────────┘           └─────────┬─────────┘
          ▼                               ▼
┌───────────────────┐           ┌───────────────────┐
│ Graph Transformer │           │ Graph Transformer │
│ Encoder (4 layers)│           │ Encoder (4 layers)│
│ • Multi-head Attn │           │ • Multi-head Attn │
│ • Edge-type Bias  │           │ • Edge-type Bias  │
│ • FFN + LayerNorm │           │ • FFN + LayerNorm │
└─────────┬─────────┘           └─────────┬─────────┘
          ▼                               ▼
┌───────────────────┐           ┌───────────────────┐
│ Weighted Sum      │           │ Weighted Sum      │
│ Readout Layer     │           │ Readout Layer     │
└─────────┬─────────┘           └─────────┬─────────┘
          │                               │
          └───────────┬───────────────────┘
                      ▼
              ┌───────────────┐
              │    Concat     │
              │ [AST + CDFG]  │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │  Classifier   │
              │    (MLP)      │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │   Output:     │
              │  Vulnerable / │
              │ Not Vulnerable│
              └───────────────┘
```

---

## 📁 Cấu trúc Project

```
Auto_Bug_Detection/
├── 📄 Train_Model.py          # Script huấn luyện model
├── 📄 Detector.py             # Script phát hiện lỗ hổng
├── 📄 Code2Graph.py           # Chuyển đổi code thành graph
├── 📄 Requirements.txt        # Dependencies
├── 📄 README.md               # Documentation
├── 📄 .gitignore              # Git ignore rules
│
├── 📂 TIFS_Data/              # 📥 Dataset (tải từ Drive)
│   ├── SARD/                  # Raw SARD dataset
│   ├── SARD_after/            # Preprocessed data
│   ├── graphs/                # Generated graphs
│   └── preprocess_sard.py     # Preprocessing script
│
├── 📂 Trained_Model/          # Model checkpoints
│   ├── Auto_Bug_Detector.pt
│   └── Auto_Bug_Detector_best.pt
│
├── 📂 logs/                   # Training logs (TensorBoard)
│
└── 📂 evaluation_results/     # Evaluation metrics
```

---

## 📥 Dataset

> ⚠️ **Lưu ý:** Folder `TIFS_Data/` chứa dataset SARD rất nặng (~2GB+) nên **không được commit lên Git**.

### Tải Dataset

📦 **Google Drive:** [Click để tải TIFS_Data](https://drive.google.com/drive/folders/YOUR_FOLDER_ID_HERE?usp=sharing)

Sau khi tải về:
1. Giải nén file (nếu là .zip)
2. Đặt folder `TIFS_Data/` vào thư mục gốc của project
3. Cấu trúc đúng: `Auto_Bug_Detection/TIFS_Data/SARD/...`

### Dataset Info

| Thông tin | Giá trị |
|-----------|---------|
| **Nguồn** | SARD (Software Assurance Reference Dataset) |
| **Ngôn ngữ** | C/C++ |
| **Loại lỗ hổng** | CWE-77 (Command Injection), CWE-119 (Buffer Overflow)... |
| **Số mẫu** | ~10,000+ functions |
| **Format** | Source code → AST/CDFG graphs → JSONL |

---

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/X181125/Auto_Bug_Detection.git
cd Auto_Bug_Detection
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r Requirements.txt
```

### 4. Tải dataset

Tải `TIFS_Data/` từ Google Drive (link ở trên) và đặt vào thư mục project.

---

## 🏋️ Huấn luyện Model

### Quick Start

```bash
python Train_Model.py
```

### Với Custom Parameters

```bash
python Train_Model.py \
    --epochs 50 \
    --lr 1e-3 \
    --state-dim 128 \
    --num-layers 4 \
    --num-heads 4 \
    --patience 10
```

### Training Arguments

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--epochs` | `50` | Số epochs huấn luyện |
| `--lr` | `1e-3` | Learning rate |
| `--weight-decay` | `1e-4` | Weight decay (L2 regularization) |
| `--state-dim` | `128` | Hidden dimension |
| `--num-layers` | `4` | Số Graph Transformer layers |
| `--num-heads` | `4` | Số attention heads |
| `--max-nodes-per-batch` | `8000` | Max nodes mỗi batch |
| `--patience` | `10` | Early stopping patience |

### Theo dõi Training với TensorBoard

```bash
tensorboard --logdir logs
```

---

## 🔍 Phát hiện Lỗ hổng (Inference)

### Sử dụng file code mẫu

```bash
python Detector.py --source badExample.c
```

### Output mẫu

```
📊 Vulnerability Detection Result
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: badExample.c
Prediction: ⚠️ VULNERABLE
Confidence: 87.3%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📊 Kết quả Thực nghiệm

| Metric | CWE-77 Dataset |
|--------|----------------|
| **Accuracy** | 85-90% |
| **Precision** | 80-85% |
| **Recall** | 80-90% |
| **F1-Score** | 80-87% |

*Kết quả có thể thay đổi tùy thuộc vào hyperparameters và quá trình tiền xử lý dữ liệu.*

---

## 🔧 Các thành phần chính

### 1. `Code2Graph.py`
Chuyển đổi mã nguồn C/C++ thành đồ thị AST và CDFG.

### 2. `Train_Model.py`
- `GraphTransformerLayer`: Multi-head self-attention với edge-type embedding
- `GraphTransformerEncoder`: Stack của N GraphTransformerLayer
- `WeightedSumReadout`: Attention-based pooling
- `VulnDetectorGraphTransformer`: Model chính kết hợp 2 view

### 3. `Detector.py`
Script inference để phát hiện lỗ hổng từ file mã nguồn.

---

## ⚙️ Hyperparameter Tuning

### Dataset nhỏ (< 1000 samples)
```bash
--state-dim 64 --num-layers 2 --num-heads 2 --epochs 30
```

### Dataset vừa (1000-10000 samples)
```bash
--state-dim 128 --num-layers 4 --num-heads 4 --epochs 50
```

### Dataset lớn (> 10000 samples)
```bash
--state-dim 256 --num-layers 6 --num-heads 8 --epochs 100
```

---

## 🐛 Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| **Out of Memory** | Giảm `--max-nodes-per-batch`, `--state-dim`, `--num-layers` |
| **Overfitting** | Tăng `--weight-decay`, giảm `--epochs` |
| **Underfitting** | Tăng `--state-dim`, `--num-layers`, `--epochs` |
| **Dataset không tìm thấy** | Kiểm tra đã tải `TIFS_Data/` từ Drive chưa |

---

## 📚 Tham khảo

1. **Graph Transformer Networks** - Yun et al., NeurIPS 2019
2. **Devign: Effective Vulnerability Identification** - Zhou et al., NeurIPS 2019
3. **FUNDED: Flow-based Vulnerability Detection** - Wang et al., ICSE 2020
4. **SARD Dataset** - NIST Software Assurance Reference Dataset

---

## 👨‍💻 Thông tin Đồ án

| Thông tin | Chi tiết |
|-----------|----------|
| **Môn học** | An toàn Thông tin / Machine Learning |
| **Trường** | Đại học Công nghệ Thông tin - ĐHQG TPHCM (UIT) |
| **Sinh viên** | Nguyễn Đình Hưng |
| **MSSV** | 23520564 |

---

## 📄 License

MIT License - Sử dụng cho mục đích học tập và nghiên cứu.

---

<p align="center">
  <strong>⭐ Nếu project hữu ích, hãy cho một star nhé!</strong>
</p>
#   A u t o _ B u g _ D e t e c t i o n  
 