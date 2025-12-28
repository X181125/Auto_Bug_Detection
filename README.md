# Auto Bug Detection using Graph Transformer

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Academic%20Project-blue?style=flat-square)

**Do an mon hoc: Phat hien lo hong bao mat tu dong trong ma nguon C/C++ su dung Graph Transformer**

---

## Gioi thieu

Du an nay xay dung mot he thong phat hien lo hong bao mat tu dong trong ma nguon C/C++ su dung mo hinh Graph Transformer. He thong hoc bieu dien tu hai view do thi:

| View | Mo ta |
|------|-------|
| **AST** (Abstract Syntax Tree) | Cay cu phap truu tuong - bieu dien cau truc ngu phap cua ma nguon |
| **CDFG** (Control Data Flow Graph) | Do thi luong dieu khien va du lieu - bieu dien luong thuc thi va phu thuoc du lieu |

### Muc tieu

- Phat hien cac lo hong bao mat pho bien (CWE-77, Command Injection, Buffer Overflow...)
- Ung dung Deep Learning (Graph Neural Networks) vao bai toan phan tich ma nguon
- Ket hop thong tin tu nhieu bieu dien do thi de tang do chinh xac

---

## Kien truc Model

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

## Cau truc Project

```
Auto_Bug_Detection/
    |
    |-- Train_Model.py          # Script huan luyen model
    |-- Detector.py             # Script phat hien lo hong
    |-- Code2Graph.py           # Chuyen doi code thanh graph
    |-- Requirements.txt        # Dependencies
    |-- README.md               # Documentation
    |-- .gitignore              # Git ignore rules
    |
    |-- TIFS_Data/              # Dataset (tai tu Drive)
    |       |-- SARD/           # Raw SARD dataset
    |       |-- SARD_after/     # Preprocessed data
    |       |-- graphs/         # Generated graphs
    |       |-- preprocess_sard.py
    |
    |-- Trained_Model/          # Model checkpoints
    |       |-- Auto_Bug_Detector.pt
    |       |-- Auto_Bug_Detector_best.pt
    |
    |-- logs/                   # Training logs (TensorBoard)
    |
    |-- evaluation_results/     # Evaluation metrics
```

---

## Dataset

**Luu y:** Folder `TIFS_Data/` chua dataset SARD rat nang (~2GB+) nen khong duoc commit len Git.

### Tai Dataset

**Google Drive:** [Click de tai TIFS_Data](https://drive.google.com/drive/folders/YOUR_FOLDER_ID_HERE?usp=sharing)

Sau khi tai ve:
1. Giai nen file (neu la .zip)
2. Dat folder `TIFS_Data/` vao thu muc goc cua project
3. Cau truc dung: `Auto_Bug_Detection/TIFS_Data/SARD/...`

### Dataset Info

| Thong tin | Gia tri |
|-----------|---------|
| Nguon | SARD (Software Assurance Reference Dataset) |
| Ngon ngu | C/C++ |
| Loai lo hong | CWE-77 (Command Injection), CWE-119 (Buffer Overflow)... |
| So mau | ~10,000+ functions |
| Format | Source code -> AST/CDFG graphs -> JSONL |

---

## Cai dat

### 1. Clone repository

```bash
git clone https://github.com/X181125/Auto_Bug_Detection.git
cd Auto_Bug_Detection
```

### 2. Tao virtual environment (khuyen nghi)

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Cai dat dependencies

```bash
pip install -r Requirements.txt
```

### 4. Tai dataset

Tai `TIFS_Data/` tu Google Drive (link o tren) va dat vao thu muc project.

---

## Huan luyen Model

### Quick Start

```bash
python Train_Model.py
```

### Voi Custom Parameters

```bash
python Train_Model.py --epochs 50 --lr 1e-3 --state-dim 128 --num-layers 4 --num-heads 4 --patience 10
```

### Training Arguments

| Argument | Default | Mo ta |
|----------|---------|-------|
| `--epochs` | 50 | So epochs huan luyen |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-4 | Weight decay (L2 regularization) |
| `--state-dim` | 128 | Hidden dimension |
| `--num-layers` | 4 | So Graph Transformer layers |
| `--num-heads` | 4 | So attention heads |
| `--max-nodes-per-batch` | 8000 | Max nodes moi batch |
| `--patience` | 10 | Early stopping patience |

### Theo doi Training voi TensorBoard

```bash
tensorboard --logdir logs
```

---

## Phat hien Lo hong (Inference)

### Su dung file code mau

```bash
python Detector.py --source badExample.c
```

### Output mau

```
Vulnerability Detection Result
================================
File: badExample.c
Prediction: VULNERABLE
Confidence: 87.3%
================================
```

---

## Ket qua Thuc nghiem

| Metric | CWE-77 Dataset |
|--------|----------------|
| Accuracy | 85-90% |
| Precision | 80-85% |
| Recall | 80-90% |
| F1-Score | 80-87% |

Ket qua co the thay doi tuy thuoc vao hyperparameters va qua trinh tien xu ly du lieu.

---

## Cac thanh phan chinh

### 1. Code2Graph.py
Chuyen doi ma nguon C/C++ thanh do thi AST va CDFG.

### 2. Train_Model.py
- `GraphTransformerLayer`: Multi-head self-attention voi edge-type embedding
- `GraphTransformerEncoder`: Stack cua N GraphTransformerLayer
- `WeightedSumReadout`: Attention-based pooling
- `VulnDetectorGraphTransformer`: Model chinh ket hop 2 view

### 3. Detector.py
Script inference de phat hien lo hong tu file ma nguon.

---

## Hyperparameter Tuning

### Dataset nho (< 1000 samples)
```bash
--state-dim 64 --num-layers 2 --num-heads 2 --epochs 30
```

### Dataset vua (1000-10000 samples)
```bash
--state-dim 128 --num-layers 4 --num-heads 4 --epochs 50
```

### Dataset lon (> 10000 samples)
```bash
--state-dim 256 --num-layers 6 --num-heads 8 --epochs 100
```

---

## Troubleshooting

| Van de | Giai phap |
|--------|-----------|
| Out of Memory | Giam `--max-nodes-per-batch`, `--state-dim`, `--num-layers` |
| Overfitting | Tang `--weight-decay`, giam `--epochs` |
| Underfitting | Tang `--state-dim`, `--num-layers`, `--epochs` |
| Dataset khong tim thay | Kiem tra da tai `TIFS_Data/` tu Drive chua |

---

## Tham khao

1. Graph Transformer Networks - Yun et al., NeurIPS 2019
2. Devign: Effective Vulnerability Identification - Zhou et al., NeurIPS 2019
3. FUNDED: Flow-based Vulnerability Detection - Wang et al., ICSE 2020
4. SARD Dataset - NIST Software Assurance Reference Dataset

---

## Thong tin Do an

| Thong tin | Chi tiet |
|-----------|----------|
| Mon hoc | An toan Thong tin / Machine Learning |
| Truong | Dai hoc Cong nghe Thong tin - DHQG TPHCM (UIT) |
| Sinh vien | Nguyen Dinh Hung |
| MSSV | 23520564 |

---

## License

MIT License - Su dung cho muc dich hoc tap va nghien cuu.
