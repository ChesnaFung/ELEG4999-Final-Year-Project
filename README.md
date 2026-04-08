# YOLOv11-Based Vehicle Turn Signal Detection for Autonomous Driving

**Final Year Project (Thesis II)**
**Department of Electronic Engineering**
**The Chinese University of Hong Kong**

**Author:** Fung Cheuk Nam
**Supervisor:** Prof. Li Hongsheng  
**Date:** 8 April 2026

## Project Overview

This repository contains the complete implementation of a high-performance **vehicle turn signal (on/off) detection system** based on **YOLOv11s**, specifically optimized for real-world Taiwan highway scenarios in autonomous driving applications.

The project builds upon the modular Faster R-CNN framework developed in Thesis I and shifts to a single-stage YOLOv11 architecture to achieve real-time performance while maintaining high accuracy for small turn signal detection.

---

**Key Highlights:**
- Custom Taiwan highway dataset collected from dashboard camera videos
- Vehicle-centric cropping + Real-ESRGAN 2× super-resolution preprocessing
- Binary classification (turn_signal_on vs. turn_signal_off)
- Strong performance on NVIDIA RTX 2080Ti: **mAP@0.5 = 0.958**, **mAP@0.5:0.95 = 0.839**, **138 FPS**
- Comprehensive comparison with YOLOv8s on the identical dataset

---

## 🛠️ Repository Structure

The project is organized into the following directory structure to ensure clarity and reproducibility:

```text
├── data/
│   └── data.yaml           # Dataset configuration (classes & paths)
├── src/
│   ├── extract_time_data_v8.py     # Measure the inference latency and FPS of YOLOv8
│   ├── extract_time_data_v11.py    # Measure the inference latency and FPS of YOLOv11
│   └── train.py                # Model training script for YOLOv11 and v8
├── weights/
│   ├── yolov11s_best.pt    # Proposed model (0.958 mAP0.5)
│   └── yolov8s_best.pt     # Baseline model (0.957 mAP0.5)
├── results/
│   ├── v11_run/          # Training curves & Confusion Matrix for YOLOv11s
│   └── v8_run/        # Metrics for the baseline comparison
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

---

## 📦 Dataset Availability
The localized Taiwan Highway Dataset (manual annotations + enhanced frames) is currently being synchronized. 
- **Status:** 🏗️ Uploading to [Google Drive]
- **Estimated Live Date:** April 15, 2026.
- **Sample Data**: A small set of sample images and labels is provided in the data/samples/ for immediate review.
- *If you require immediate access for academic review, please contact the author via the email.*

---

## 🚀 Quick Start

### 1. Environment Setup
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/ChesnaFung/ELEG4999-Final-Year-Project.git
cd FYP_TurnSignal
pip install -r requirements.txt