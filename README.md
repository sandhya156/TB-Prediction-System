cd /workspaces/TB-Prediction-System
cat > README.md << 'EOF'
# 🩺 Tuberculosis Prediction System

A multimodal deep learning system that predicts Tuberculosis (TB) risk using chest X-ray images and/or patient symptom data.

## 📋 Project Overview

This project combines two independent deep learning modules into a single fusion-based prediction dashboard:

1. **X-ray Module** — A CNN (transfer learning with MobileNetV2) trained to classify chest X-rays as Normal or TB-positive.
2. **Clinical Module** — An Artificial Neural Network (ANN) trained on patient symptoms and risk factors to predict TB likelihood.

Predictions from both modules are combined using a weighted fusion logic to produce a final risk assessment.

## 🎯 Results

### X-ray Model
- **Accuracy:** 97%
- **Precision (TB):** 1.00
- **Recall (TB):** 0.85
- Trained on: [TB Chest X-ray Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) — 3500 Normal + 700 TB images

### Clinical Model
- **Accuracy:** 90%
- **Precision (TB):** 0.86
- **Recall (TB):** 0.95
- Trained on: [TB Diagnosis Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/farihamoni/tuberculosis-diagnosis-prediction)
- Features used: age, gender, cough duration, fever, weight loss, night sweats, chest pain, shortness of breath, hemoptysis, smoking status, BMI, TB contact history

## 🧠 Fusion Logic

Since no single dataset had X-ray and clinical data for the same patients, we use a weighted ensemble approach: