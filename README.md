# 🧠 Agency Type Prediction using Machine Learning - Travel Insurance

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Neural%20Network-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📋 Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Data Pre-processing](#data-pre-processing)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Results & Visualizations](#results--visualizations)
6. [Conclusion](#conclusion)
7. [Contributors](#contributors)
8. [Tools and Libraries](#tools-and-libraries)
9. [License](#license)

---

## 🏁 Introduction

In the modern travel industry, insurance agencies and airlines generate large volumes of data about customer policies, claims, destinations, and transactions.  
This project uses **machine learning** to classify whether a policy belongs to an **Airlines** or a **Travel Agency**.

We implemented and compared several models:
- 🧮 **K-Nearest Neighbors (KNN)**
- 📈 **Logistic Regression** *(from scratch)*
- 🧠 **Neural Network (Feed-Forward)**
- 🎯 **K-Means Clustering** *(unsupervised)*
- 🔁 **K-Fold Cross-Validation** *(robust evaluation)*

The goal was to explore how these algorithms handle **mixed categorical and numerical data** and how model choice affects accuracy and generalization.

---

## 📊 Dataset Description

**Source:** Google Sheets  
**Target Variable:** `Agency Type` *(Binary – Airlines or Travel Agency)*

| Feature Type | Columns |
|:--------------|:-----------------------------------------------------------|
| **Numerical** | Duration, Net Sales, Commission (in value), Age |
| **Categorical** | Distribution Channel, Destination, Claim, Gender, Agency |

### 🧩 Preprocessing Steps:
- Dropped unnecessary columns: `Agency`, `Product Name`, `Gender` (had 45k+ nulls)
- No missing values found  
- **OneHotEncoder** → categorical features  
- **LabelEncoder** → Claim, Agency Type  
- **StandardScaler** → numerical features  
- Train-Test Split → 70% / 30%

### 🔍 Observations:
- Dataset was **imbalanced** (more *Travel Agency* samples).
- Strong correlations among `Agency`, `Agency Type`, and `Distribution Channel`.

---

## ⚙️ Data Pre-processing

1. **Dropped Columns:** `Gender`, `Agency`, `Product Name`  
2. **Encoding:** Applied One-Hot and Label encoding  
3. **Scaling:** Used `StandardScaler` for normalization  
4. **Train-Test Split:** 70% training, 30% testing  

This ensured that models trained efficiently without feature dominance due to scale.

---

## 🤖 Model Training and Evaluation

### 1️⃣ Logistic Regression
**Performance:**
- Accuracy: **82%**
- Precision: 70% (Airlines), 86% (Travel Agency)
- Recall: 60% (Airlines), 90% (Travel Agency)
- ROC-AUC: **0.85**

💡 *Served as a baseline model — struggled with imbalanced classes.*

---

### 2️⃣ K-Nearest Neighbors (KNN)
**Best K:** 3  
**Cross-Validation Accuracy:** 98.57%

**Performance:**
- Accuracy: **98.7%**
- Precision: 98% (Airlines), 99% (Travel Agency)
- Recall: 97% (Airlines), 99% (Travel Agency)
- ROC-AUC: **0.99**

🔥 *Best overall performer — strong balance between both classes.*

---

### 3️⃣ Neural Network
**Performance:**
- Accuracy: **97.47%**
- Precision: 92% (Airlines), 100% (Travel Agency)
- Recall: 99% (Airlines), 97% (Travel Agency)
- ROC-AUC: **0.90**

⚙️ *Captured nonlinear patterns but required more computation time.*

---

### 4️⃣ K-Means Clustering (Unsupervised)
Used **PCA (2D)** projection with **K=2** to visualize natural groupings between agency types.

---

## 📈 Results & Visualizations

### 🧮 Model Comparison Summary

| Model | Accuracy | ROC-AUC | Precision (Avg) | Recall (Avg) | Notes |
|:------|:---------:|:-------:|:----------------:|:-------------:|:------|
| Logistic Regression | 82% | 0.85 | 78% | 75% | Baseline linear model |
| KNN | **98.7%** | **0.99** | **98.5%** | **98%** | Best overall |
| Neural Network | 97.4% | 0.90 | 96% | 98% | Slightly behind KNN |

---

### 🧠 Precision vs Recall Visualization
*(Insert your precision/recall chart here)*  
```bash
📊 Placeholder: precision_recall_comparison.png
```

### 🧩 Confusion Matrix Examples
*(Insert confusion matrix image for each model)*  
```bash
🧾 Placeholder: confusion_matrix_knn.png
🧾 Placeholder: confusion_matrix_nn.png
```

---

## 🧾 Conclusion

✅ **KNN** achieved the **best performance** with:
- Accuracy: **98.7%**
- ROC-AUC: **0.99**
- Balanced Precision & Recall

⚙️ **Neural Network** also performed strongly, capturing nonlinear dependencies effectively.  
📉 **Logistic Regression** was helpful as a baseline but not ideal for this mixed, imbalanced dataset.

### 💡 Key Insights
- Distance-based models (like KNN) are powerful for scaled mixed datasets.  
- Neural networks excel at pattern recognition but need tuning.  
- Proper scaling and encoding dramatically improve model performance.

### 🚀 Future Work
- Handle class imbalance with oversampling/SMOTE.  
- Tune hyperparameters for the neural network.  
- Try ensemble models: **Random Forest**, **XGBoost**, or **LightGBM**.

---

## 👨‍💻 Contributors
**Developed by:** Khandkar Sajid  
**Institution:** BRAC University (BRACU)  
**Major:** Computer Science & Finance  
**Specialization:** Cybersecurity, Penetration Testing, and Machine Learning  

---

## 🧠 Tools and Libraries
- Python 🐍  
- NumPy / Pandas  
- Matplotlib / Seaborn  
- Scikit-learn  
- TensorFlow / Keras  
- Jupyter Notebook  

---

## 📜 License
This project is licensed under the **MIT License**.  
Feel free to use, modify, and share with proper attribution.

---

⭐ *If you found this project helpful, consider giving it a star on GitHub!* ⭐
