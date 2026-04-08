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
- `src/`: Core Python scripts for training and data preprocessing.
- `data/`: Configuration files (e.g., `data.yaml`) for the Taiwan Highway dataset.
- `weights/`: Trained model weights (`best.pt`) for reproducibility.
- `results/`: Visualization of training metrics, loss curves, and confusion matrices.
- `requirements.txt`: Environment dependencies.

---

## 📦 Dataset Availability
The localized Taiwan Highway Dataset (manual annotations + enhanced frames) is currently being synchronized. 
- **Status:** 🏗️ Uploading to [Kaggle/Google Drive]
- **Estimated Live Date:** April 15, 2026.
- *If you require immediate access for academic review, please contact the author via the email listed in the thesis.*

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt