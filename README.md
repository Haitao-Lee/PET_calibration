# PETCalibrator: Geometric Consistency-Guided De-bias Learning for Precise Image-Based PET Detector Calibration

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official implementation of **PETCalibrator**, a geometric consistency-guided de-bias learning framework designed for precise and efficient image-based Positron Emission Tomography (PET) detector calibration.

> **News:** Our paper has been submitted to *Medical Image Analysis*.

---

## 📖 Introduction

Calibration of PET detectors is critical for ensuring high-quality clinical imaging.  
Compared to traditional semi-automated calibration workflows that rely heavily on manual refinement, fully automated image-based approaches provide superior adaptability and efficiency.

However, accurate signal peak localization in visually complex and inherently noisy flood maps remains a persistent challenge. Severe signal aliasing and continuous spatial label bias can significantly degrade calibration reliability.

**PETCalibrator explicitly addresses this problem by correcting continuous spatial label bias while preserving the intrinsic geometric consistency of the physical crystal array.**

---

## ⚙️ Framework Overview

![Overview of the PETCalibrator Framework](static/overview.png)

The proposed pipeline consists of three major stages:

### 1️⃣ Markov-inspired Data Regularization  
Transforms inherently disordered and noisy peak label sequences into a topologically ordered structure, improving dataset learnability while preserving structural priors.

### 2️⃣ Geometric Consistency-Guided Peak Localization  
Introduces the **Gradient-Guided De-biasing Module (GGDM)**, which actively shifts biased peak predictions along the underlying topographic gradient field to correct continuous spatial deviation.

![Workflow of the GGDM](static/GGDM.png)

### 3️⃣ Pixel-wise Image Regridding  
Applies a Voronoi diagram-based linear clustering strategy to establish accurate correspondences between pixel-level responses and crystal-level channels, generating the refined Look-Up Table (LUT).

---

## 📊 Comparison Methods

To rigorously evaluate the superiority of PETCalibrator in signal peak localization, we conduct extensive comparisons against recent state-of-the-art architectures, including.
- **ConvNeXtV2** (CVPR 2023)  
- **nnUNet** (Nature Methods 2021)  
- **SCTNet** (AAAI 2024)  
- **Focal Modulation Networks (FocalNet)** (NeurIPS 2022)  
- **ConDSeg** (AAAI 2025)  
- **UNet3Plus** (ICASSP 2020)  
- **nnFormer** (IEEE TIP 2023)  
- **Integrally Transformer Pyramid Networks (iTPN)** (CVPR 2023)  
- **Vision Transformer (ViT)** (ICLR 2020)  
- **Swin Transformer V2 (SwinV2)** (CVPR 2022)  
- **UNETR++** (IEEE TMI 2024)  
- **TransUNet** (Medical Image Analysis 2024)  
- **MobileMamba** (arXiv 2024)  
- **VMamba** (NeurIPS 2024)  
- **Vision Mamba** (ICML 2024)  
- **SegMamba** (MICCAI 2024)  
- **SparX** (AAAI 2025)  
- **UMamba** (arXiv 2024)  

---

## ⚖️ Fair Comparison Protocol

To ensure strict fairness:

- All models are trained under identical experimental settings.
- Each architecture is used strictly as a backbone feature extractor.
- A unified linear regression head is appended for peak coordinate prediction.
- Training schedules, optimizers, and data splits are fully consistent.

---

## 📈 Experimental Results

### Quantitative Comparison

![Quantitative comparison of different models](static/reliability.png)

### Qualitative Comparison

![Qualitative comparison of different models](static/Comparison.png)

Extensive experiments demonstrate that PETCalibrator consistently achieves superior localization accuracy and improved structural consistency compared with strong CNN, Transformer, and Mamba baselines.

---

## 🛠 Installation

### Prerequisites

- Linux or macOS  
- Python 3.8+  
- NVIDIA GPU with CUDA and cuDNN  

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Haitao-Lee/PET_calibration.git
cd PET_calibration

pip install -r requirements.txt
```

---

## 🚀 Training

```bash
python train.py --config configs/petcalibrator.yaml
```

---

## 📌 Citation

If this work is helpful for your research, please consider citing:

```bibtex
@article{li2025petcalibrator,
  title={PETCalibrator: Geometric Consistency-Guided De-bias Learning for Precise Image-Based PET Detector Calibration},
  author={Li, Haitao and Yu, Xinbo and Wang, Feng and Wu, Yiqun and Chen, Xiaojun},
  journal={Medical Image Analysis},
  year={2025},
  note={Under review}
}
```

---

## 📄 License

This project is released under the MIT License.
