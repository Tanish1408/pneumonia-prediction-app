# 🫁 AI-Powered Pneumonia Detection System & Research Suite

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Django](https://img.shields.io/badge/Django-5.0-green)
![Research](https://img.shields.io/badge/Research-VGG16%20vs%20Custom%20CNN-red)

## 📌 Overview
This repository houses a full-stack **Medical Diagnostic System** capable of detecting pneumonia from chest X-rays with high sensitivity.

Beyond the application, this project includes a **Research Module** comparing two deep learning architectures:
1.  **Custom CNN (Lightweight):** A resource-efficient model optimized for edge devices (10MB).
2.  **VGG16 (Transfer Learning):** A heavy-duty model using pre-trained ImageNet weights for maximum precision.

The final deployed application uses the **Custom CNN** compressed with **TensorFlow Lite**, achieving **98% Recall** while remaining light enough for mobile deployment.

Website: https://pneumonia-prediction-app.onrender.com
---

## 🔬 Research: Custom CNN vs. VGG16
*Analysis of Trade-offs between Accuracy, Sensitivity, and Deployment Efficiency.*

To determine the best model for real-world medical use, I conducted a comparative study between a custom architecture trained from scratch and VGG16 using Transfer Learning.

### **Performance Comparison Table**

| Metric | **Custom CNN (Deployed)** | **VGG16 (Research)** | **Winner** |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 90.87% | **93.00%** | **VGG16** (+2.13%) |
| **Recall (Sensitivity)** | **98.00%** | 97.00% | **Custom CNN** (Safer) |
| **Precision (Normal)** | Lower | **95.00%** | **VGG16** (Fewer False Alarms) |
| **Model Size** | **10 MB (TFLite)** | ~528 MB (.h5) | **Custom CNN** (50x Smaller) |
| **Inference Speed** | **~50ms** | ~200ms | **Custom CNN** |

### **Key Findings**
* **VGG16** is the "smarter" model overall, achieving **93% Accuracy** and better precision in identifying healthy patients.
* **Custom CNN** is the "safer" and "faster" model. It achieved a higher **Recall (98%)**, meaning it missed fewer pneumonia cases (False Negatives).
* **Conclusion:** For this specific use case—deploying a diagnostic tool to low-resource clinics—the Custom CNN was selected for production due to its **50x smaller footprint** and superior sensitivity.

---

## 🚀 The Web Application
The winning model (Custom CNN) was deployed using **Django**.

### **Features**
* **Real-time Diagnosis:** Upload a chest X-ray and get results in <2 seconds.
* **Confidence Score:** Displays the probability percentage of the diagnosis.
* **Medical-Grade Recall:** Optimized to minimize False Negatives.
* **Responsive UI:** Built with Bootstrap for access on tablets and phones.

### **Tech Stack**
* **Core:** Python 3.10+
* **ML/AI:** TensorFlow, Keras, TensorFlow Lite
* **Web Framework:** Django 5.0
* **Frontend:** HTML5, CSS3, Bootstrap 5
* **Deployment:** Render / Vercel
* **Data Processing:** NumPy, Pillow (PIL)

---
