# 🧠 Brain Tumor Detection from MRI Scans (2D-CNN & Deep Learning Suite)
### AIDS2 Mini-Project | Deep Learning, Medical Imaging & Explainable AI (XAI)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MPS GPU](https://img.shields.io/badge/Accelerator-Apple_Silicon_MPS-success.svg)](https://developer.apple.com/metal/pytorch/)

---

## 📌 1. Project Overview & Clinical Problem Statement
Early and accurate detection of primary brain neoplasms from Magnetic Resonance Imaging (MRI) is vital for neurosurgical planning and clinical oncology triage. Manual interpretation by radiologists is labor-intensive, time-consuming, and susceptible to fatigue and inter-observer variations.

This project delivers an automated **2D-CNN Deep Learning system** that classifies 2D axial Brain MRI scans into four distinct diagnostic categories:
1. **`glioma`** (Glioma Brain Tumor — Intra-axial glial cell malignancy)
2. **`meningioma`** (Meningioma Brain Tumor — Dural-based extra-axial mass)
3. **`pituitary`** (Pituitary Macroadenoma — Sellar/suprasellar endocrine tumor)
4. **`notumor`** (Healthy Control — Normal symmetrical neuroanatomy)

---

## 🔬 2. Deep Learning Architecture Paradigms

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
| • Dense Head (1024 -> 512 -> 128 -> 4)   |   | • Trainable Params: ~11.5 Million        |
| • Trainable Params: ~990 Thousand        |   | • Optimization: AdamW (lr=3e-4)          |
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
|                    CLINICAL AI WORKSTATION WITH GRAD-CAM & TTA                          |
|         Real-time Diagnostics + Interactive Split-Slider Reticle Heatmap Overlay        |
+-----------------------------------------------------------------------------------------+
```

---

## 📁 3. Dataset Architecture & Class Distribution

The project utilizes a curated benchmark cohort of **7,200 high-resolution Brain MRI scans** ([Kaggle Benchmark](https://www.kaggle.com/datasets/masoudnickparamb/brain-tumor-mri-dataset)) partitioned into stratified training and testing subsets:

| Pathological Class | Training Set (`archive/Training`) | Testing Set (`archive/Testing`) | Total Images |
| :--- | :---: | :---: | :---: |
| **Glioma Tumor** | 1,400 | 400 | **1,800** |
| **Meningioma Tumor** | 1,400 | 400 | **1,800** |
| **Pituitary Tumor** | 1,400 | 400 | **1,800** |
| **Healthy (No Tumor)** | 1,400 | 400 | **1,800** |
| **TOTAL** | **5,600 (77.8%)** | **1,600 (22.2%)** | **7,200 (100%)** |

---

## 🥊 4. Cross-Architecture Empirical Benchmark: Why CNN vs. ANN vs. RNN?

To directly validate why **Convolutional Neural Networks (CNN)** are mathematically mandatory for 2D medical imaging, we implemented, trained, and evaluated an **ANN (Multi-Layer Perceptron)** and an **RNN (Bidirectional LSTM)** under the exact same dataset splits:

| Architecture Paradigm | Trainable Parameters | Test Accuracy (%) | Macro F1-Score (%) | Inference Latency | Empirical Finding & Theoretical Reason |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **RNN (Bidirectional LSTM)** | ~386 Thousand | **62.56%** 🔻 | **60.14%** | ~8.3 ms | **Worst Performer**: Sequential row-by-row scanning suffers from vanishing gradients over 224 steps and cannot capture 2D spatial tumor shapes. |
| **ANN (Deep MLP)** | **38.56 Million** ⚠️ | **70.75%** | **70.34%** | ~8.3 ms | **Parameter Explosion**: Flattening a $224 \times 224 \times 3$ image into a 1D vector destroys spatial adjacency; requires 38.5M parameters causing severe overfitting. |
| **Model 1: Custom 2D-CNN** *(From Scratch)* | **~990 Thousand** | **76.94%** 📈 | **75.92%** | ~8.5 ms | **Compact & Spatial**: Receptive fields with weight sharing capture local tissue edges with under 1M parameters. |
| **Model 2: Pretrained ResNet18** *(Transfer Learning)* | **~11.57 Million** | **94.56%** 🏆 | **94.46%** | ~14.2 ms | **Gold Standard**: Residual connections prevent vanishing gradients, and 1.28M ImageNet visual priors detect fine tumor margins. |
| **Enhanced ResNet50 (with TTA)** | **~23.18 Million** | **94.62%** 🏆 | **94.52%** | ~18.5 ms | **Highest Convergence**: Deeper 3-layer bottleneck residual blocks with Label Smoothing (0.08) achieved **98.33% Validation Accuracy**. |

![Architecture Comparison](results/architecture_comparison_chart.png)

---

## 📊 5. Deep Learning Results & Metrics Breakdown

Both models were evaluated across the independent **1,600 test scans** in `archive/Testing`:

| Evaluation Metric | Model 1: Custom 2D-CNN (From Scratch) | Model 2: Pretrained ResNet18 (Transfer Learning) | Performance Delta / Winner |
| :--- | :---: | :---: | :--- |
| **Overall Test Accuracy** | **76.94%** | **94.56%** | 🏆 **ResNet18 (+17.62% Higher Accuracy)** |
| **Macro Precision** | **77.05%** | **95.03%** | 🏆 **ResNet18 (+17.98% Higher Precision)** |
| **Macro Recall (Sensitivity)**| **76.94%** | **94.56%** | 🏆 **ResNet18 (Minimizes False Negatives)** |
| **Macro F1-Score** | **75.92%** | **94.46%** | 🏆 **ResNet18 (+18.54% Harmonic Score)** |
| **Trainable Parameters** | **~990 Thousand** | **~11.57 Million** | 🏆 **Custom CNN is 11.7× more compact** |
| **Inference Latency** | **~8.5 ms / scan** | **~14.2 ms / scan** | 🏆 **Custom CNN is ~40% faster** |

---

## 🩺 6. Explainable AI (XAI) & Test-Time Augmentation (TTA)

1. **Gradient-Weighted Class Activation Mapping (Grad-CAM)**:
   - Computes partial derivatives of the predicted class score with respect to the final convolutional feature maps (`Conv2d(128, 256)` in Custom CNN / `Layer4` in ResNet18).
   - Produces normalized visual heatmaps overlaid directly onto the patient's MRI slice.
   - Clinically confirms that the model focuses on actual pathological tumor boundaries rather than imaging artifacts.

2. **Bilateral Test-Time Augmentation (TTA)**:
   - Leverages anatomical bilateral brain symmetry.
   - During live inference in `app.py`, the engine evaluates both the standard orientation and horizontally flipped tensor:
     $$P_{\text{ensemble}} = \frac{P(X) + P(\text{Flip}(X))}{2}$$
   - Eliminates directional border noise and stabilizes diagnostic confidence.

---

## 🚀 7. How to Run the Project

### 1️⃣ Launch the Interactive Web Application
```bash
python3 app.py
```
Open **`http://127.0.0.1:8000`** in your browser to:
- Test arbitrary patient MRI uploads or choose from pre-loaded cohort samples.
- Switch between **Model 1 (Custom 2D-CNN)** and **Model 2 (Pretrained ResNet18)**.
- Use the **interactive before/after split slider** to wipe between raw MRI and the Grad-CAM heat reticle.

### 2️⃣ Run Full Architecture Comparison (ANN vs RNN vs CNN)
```bash
python3 compare_ann_rnn_cnn.py
```
Trains and benchmarks ANN, RNN, Custom CNN, and ResNet18, regenerating the comparison charts in `results/`.

### 3️⃣ Run End-to-End Training & Comparative Benchmark
```bash
python3 train_and_compare.py
```
Trains both models with Apple Silicon GPU (MPS) / CUDA acceleration and updates charts in `results/`.

### 4️⃣ View the Jupyter Notebook
```bash
jupyter notebook Brain_Tumor_Detection_AIDS2_Miniproject.ipynb
```
Contains pre-rendered loss curves, accuracy graphs, confusion matrices, and Grad-CAM plots.

### 5️⃣ PowerPoint Presentation
Open **`Brain_Tumor_Detection_Presentation.pptx`** in PowerPoint or Keynote for project presentation.

---

## 📁 8. Repository Structure

```
├── Brain_Tumor_Detection_AIDS2_Miniproject.ipynb # Main Jupyter Notebook (with pre-rendered charts)
├── Brain_Tumor_Detection_Presentation.pptx      # 9-Slide Academic Presentation Deck
├── README.md                                    # Master Project Documentation & Benchmark Report
├── app.py                                       # Full-Stack FastAPI Radiology AI Workstation (with TTA & Split Slider)
├── compare_ann_rnn_cnn.py                       # Cross-Architecture Benchmark (ANN vs RNN vs CNN)
├── train_enhanced_cnn.py                        # Enhanced Pretrained ResNet50 Training Pipeline
├── train_and_compare.py                         # Master training & comparative benchmark
├── test_pipeline.py                             # Automated test suite
├── generate_presentation.py                     # PPT generation script
├── src/
│   ├── dataset.py                               # PyTorch Dataset, splits & augmentations
│   ├── models.py                                # Custom 2D-CNN & Pretrained ResNet18/50
│   ├── train.py                                 # Training loops with MPS/CUDA acceleration
│   ├── evaluate.py                              # Test metrics, confusion matrix & plotters
│   └── gradcam.py                               # Grad-CAM heatmap explainability engine
├── models/                                      # Saved PyTorch model checkpoints (.pth)
└── results/                                     # Comparative charts & metrics JSON
    ├── architecture_comparison_chart.png        # Bar chart comparing ANN vs RNN vs CNN
    ├── architecture_comparison_complete.json    # JSON report of all 4 architectures
    ├── confusion_matrices.png                   # Test set confusion matrices
    ├── accuracy_comparison.png                  # Training/validation accuracy curves
    └── loss_comparison.png                      # Training/validation loss curves
```
