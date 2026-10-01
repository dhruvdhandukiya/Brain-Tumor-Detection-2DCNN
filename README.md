# 🧠 Brain Tumor Detection from MRI Scans (2D-CNN)
### AIDS2 Mini-Project | Deep Learning & Medical Image Classification Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MPS GPU](https://img.shields.io/badge/Accelerator-Apple_Silicon_MPS-success.svg)](https://developer.apple.com/metal/pytorch/)

---

## 📌 1. Project Overview & Problem Statement
Early and accurate detection of brain neoplasms from Magnetic Resonance Imaging (MRI) is vital for neurosurgical planning and clinical oncology triage. Manual interpretation by radiologists is labor-intensive and susceptible to fatigue and inter-observer variations.

This project delivers an automated **2D-CNN Deep Learning system** that classifies 2D axial Brain MRI scans into four distinct diagnostic categories:
1. **`glioma`** (Glioma Brain Tumor — Intra-axial glial malignancy)
2. **`meningioma`** (Meningioma Brain Tumor — Dural-based extra-axial mass)
3. **`pituitary`** (Pituitary Macroadenoma — Sellar/suprasellar endocrine tumor)
4. **`notumor`** (Healthy Control — Normal symmetrical neuroanatomy)

---

## 🔬 2. Comparative Model Architectures

In accordance with the mini-project specifications, we developed, evaluated, and compared **two distinct deep learning paradigms**:

```
+-----------------------------------------------------------------------------------------+
|                                    INPUT MRI SCAN                                       |
|                                  (224 x 224 x 3 RGB)                                    |
+-----------------------------------------------------------------------------------------+
             |                                                          |
             v                                                          v
+------------------------------------------+   +------------------------------------------+
|       MODEL 1: CUSTOM 2D-CNN             |   |       MODEL 2: PRETRAINED RESNET18       |
|      (Designed from Scratch)             |   |           (Transfer Learning)            |
+------------------------------------------+   +------------------------------------------+
| • 4 Convolutional Feature Stages:        |   | • 18-Layer Deep Residual Backbone        |
|   - Stage 1: Conv(32) + BatchNorm + ReLU |   | • 1.28M ImageNet Pre-trained Priors      |
|   - Stage 2: Conv(64) + BatchNorm + ReLU |   | • Identity Residual Shortcuts [F(x) + x] |
|   - Stage 3: Conv(128)+ BatchNorm + ReLU |   | • Custom Dense Classifier Head:          |
|   - Stage 4: Conv(256)+ BatchNorm + ReLU |   |   - Linear(512 -> 512) + ReLU + Dropout  |
| • Spatial Regularization: Dropout2D      |   |   - Linear(512 -> 256) + ReLU + Dropout  |
| • Adaptive Avg Pooling (2x2)             |   |   - Linear(256 -> 4 classes)             |
| • Dense Head (1024 -> 512 -> 128 -> 4)   |   | • Trainable Params: ~11.7 Million        |
| • Trainable Params: ~2.6 Million         |   | • Optimization: AdamW (lr=3e-4)          |
+------------------------------------------+   +------------------------------------------+
             |                                                          |
             +──────────────────────────┬───────────────────────────────+
                                        v
+-----------------------------------------------------------------------------------------+
|                              EMPIRICAL BENCHMARK & TEST                                 |
|           Accuracy | Precision | Recall | F1-Score | Parameter Size | Latency           |
+-----------------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------------+
|                           INTERACTIVE CLINICAL WEB APP                                  |
|           Real-time MRI Diagnostics + Grad-CAM Heatmap Reticle Overlay                  |
+-----------------------------------------------------------------------------------------+
```

---

## 📁 3. Dataset Architecture & Class Distribution

The project utilizes a curated benchmark cohort of **7,200 high-resolution Brain MRI scans** partitioned into stratified training and testing subsets:

| Pathological Class | Training Set (`archive/Training`) | Testing Set (`archive/Testing`) | Total Images |
| :--- | :---: | :---: | :---: |
| **Glioma Tumor** | 1,400 | 400 | **1,800** |
| **Meningioma Tumor** | 1,400 | 400 | **1,800** |
| **Pituitary Tumor** | 1,400 | 400 | **1,800** |
| **Healthy (No Tumor)** | 1,400 | 400 | **1,800** |
| **TOTAL** | **5,600 (77.8%)** | **1,600 (22.2%)** | **7,200 (100%)** |

---

## 📊 4. Empirical Benchmark & Evaluation Results

Both models were trained and tested across the independent **1,600 test scans** in `archive/Testing`:

| Evaluation Metric | Model 1: Custom 2D-CNN (From Scratch) | Model 2: Pretrained ResNet18 (Transfer Learning) | Performance Delta / Winner |
| :--- | :---: | :---: | :--- |
| **Overall Test Accuracy** | **76.94%** | **94.56%** | 🏆 **ResNet18 (+17.62% Higher Accuracy)** |
| **Macro Precision** | **77.05%** | **95.03%** | 🏆 **ResNet18 (+17.98% Higher Precision)** |
| **Macro Recall (Sensitivity)**| **76.94%** | **94.56%** | 🏆 **ResNet18 (Minimizes False Negatives)** |
| **Macro F1-Score** | **75.92%** | **94.46%** | 🏆 **ResNet18 (+18.54% Harmonic Score)** |
| **Trainable Parameters** | **~2.6 Million** | **~11.7 Million** | 🏆 **Custom CNN is 4.5× more compact** |
| **Inference Latency** | **~8.5 ms / scan** | **~14.2 ms / scan** | 🏆 **Custom CNN is ~40% faster** |

---

## 🩺 5. Explainable AI (XAI) with Grad-CAM

To build clinical trust and transparency, the pipeline incorporates **Gradient-Weighted Class Activation Mapping (Grad-CAM)**:
- Computes partial derivatives of the predicted class score with respect to the final convolutional feature maps (`Conv2d(128, 256)` in Custom CNN / `Layer4` in ResNet18).
- Produces normalized visual heatmaps overlaid directly onto the patient's MRI slice.
- Clinically confirms that the model focuses on actual pathological tumor boundaries rather than imaging artifacts.

---

## 🚀 6. How to Run the Project

### 1️⃣ Launch the Interactive Web Application
```bash
python3 app.py
```
Open **`http://127.0.0.1:8000`** in your browser to:
- Test arbitrary patient MRI uploads or choose from pre-loaded cohort samples.
- Switch between **Model 1 (Custom 2D-CNN)** and **Model 2 (Pretrained ResNet18)**.
- View real-time probability meters and **Grad-CAM visual overlays**.

### 2️⃣ Run Full Training & Empirical Benchmark
```bash
python3 train_and_compare.py
```
Trains both models with Apple Silicon GPU (MPS) / CUDA acceleration and updates charts in `results/`.

### 3️⃣ View the Jupyter Notebook
```bash
jupyter notebook Brain_Tumor_Detection_AIDS2_Miniproject.ipynb
```
The notebook contains pre-rendered loss curves, accuracy graphs, confusion matrices, and Grad-CAM plots.

### 4️⃣ PowerPoint Presentation
Open **`Brain_Tumor_Detection_Presentation.pptx`** in PowerPoint or Keynote for project presentation.

---

## 📁 7. Repository Structure

```
├── Brain_Tumor_Detection_AIDS2_Miniproject.ipynb # Main Jupyter Notebook (with pre-rendered charts)
├── Brain_Tumor_Detection_Presentation.pptx      # 9-Slide Academic Presentation Deck
├── README.md                                    # Project Documentation & Benchmark Report
├── app.py                                       # Full-Stack FastAPI Radiology Web Suite
├── train_and_compare.py                         # Master training & comparative benchmark
├── test_pipeline.py                             # Automated test suite
├── generate_presentation.py                     # PPT generation script
├── src/
│   ├── dataset.py                               # PyTorch Dataset, splits & augmentations
│   ├── models.py                                # Custom 2D-CNN & Pretrained ResNet18
│   ├── train.py                                 # Training loops with MPS/CUDA acceleration
│   ├── evaluate.py                              # Test metrics, confusion matrix & plotters
│   └── gradcam.py                               # Grad-CAM heatmap explainability engine
├── models/                                      # Saved PyTorch model checkpoints (.pth)
└── results/                                     # Comparative charts & metrics JSON
```
