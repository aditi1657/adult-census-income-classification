#  Adult Census Income Classification
### MP Online Internship – Machine Learning Assignment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---
## Project Overview

This project is part of the **MP Online Internship Program** and involves building a complete **binary classification pipeline** on the [Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income).

The goal is to predict whether an individual's annual income **exceeds $50,000** based on demographic and employment-related features. In this project, I worked on predicting income levels using census data. This was my first time working with a real-world imbalanced dataset and I learned a lot about how data cleaning affects model performance.


---

## 📂 Repository Structure

```
 adult-census-income-classification/
│
├── 📄 app.py       # Complete Python solution (all 5 tasks)
├── 📄 requirements.txt             # All required Python libraries
├── 📄 README.md                    # Project documentation (this file)
│
├── 📁 outputs/                     # Generated charts and results
│   ├── task1_target_distribution.png
│   ├── task3_correlation_heatmap.png
│   ├── task5_model_comparison.png
│   ├── task5_roc_curves.png
│   ├── task5_confusion_matrix.png
│   └── task5_performance_results.csv
│
└── 📁 dataset/
    └── (Download adult.csv from Kaggle — link below)
```

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| **Name** | Adult Census Income Dataset |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income) / [UCI Repository](https://archive.ics.uci.edu/ml/datasets/adult) |
| **Rows** | ~48,842 |
| **Features** | 14 (age, education, occupation, etc.) |
| **Target** | `income` → `<=50K` or `>50K` |
| **Task Type** | Binary Classification |

---

## ✅ Tasks Completed

### Task 1 – Dataset Understanding 
- Loaded and explored the dataset
- Examined shape, data types, and column names
- Generated statistical summary using `describe()`
- Visualized **target variable distribution** with a bar chart

### Task 2 – Data Cleaning 
- Replaced `'?'` entries with `NaN`
- Imputed missing values using **mode** (most frequent value)
- Removed **duplicate rows**
- Stripped extra whitespace from all categorical columns

### Task 3 – Feature Engineering 
- Applied **Label Encoding** to all categorical columns
- Encoded the target variable: `<=50K → 0`, `>50K → 1`
- Generated a **Correlation Heatmap**
- Applied **StandardScaler** for feature normalization
- Performed **80:20 stratified train-test split**

### Task 4 – Model Building

Trained **5 Classification Algorithms**:

| # | Algorithm | Library |
|---|-----------|---------|
| 1 | Logistic Regression | `sklearn.linear_model` |
| 2 | Decision Tree | `sklearn.tree` |
| 3 | Random Forest | `sklearn.ensemble` |
| 4 | K-Nearest Neighbors (KNN) | `sklearn.neighbors` |
| 5 | Support Vector Machine (SVM) | `sklearn.svm` |

### Task 5 – Performance Evaluation 

Evaluated all models on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC Score**

Generated:
- 📈 Bar chart comparing all models
- 📉 ROC curves for all 5 algorithms
- 🔵 Confusion Matrix for the best model
- 📋 Full Classification Report

---

## 📈 Results Summary

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-----------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.85 | ~0.73 | ~0.60 | ~0.66 | ~0.91 |
| Decision Tree | ~0.85 | ~0.71 | ~0.63 | ~0.67 | ~0.80 |
| **Random Forest** | **~0.87** | **~0.77** | **~0.63** | **~0.69** | **~0.93** |
| KNN | ~0.84 | ~0.70 | ~0.60 | ~0.65 | ~0.89 |
| SVM | ~0.85 | ~0.75 | ~0.57 | ~0.65 | ~0.91 |

>  **Random Forest** achieves the best overall performance.

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/adult-census-income-classification.git
cd adult-census-income-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download `adult.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income) and place it in the project root folder.

### 4. Run the Script
```bash
python app.py
```

All output charts and the results CSV will be automatically saved.

---

## Tech Stack

- **Language:** Python 3.8+
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn

---

## Author

**Aditi Gupta (23MIM10204) , VIT Bhopal University**
MP Online Internship – Machine Learning Track
Batch: [2023]


---

## Mentor

**Dr. Nishant Shrivastava**
MP Online Internship Program

---

## License

This project is for educational purposes under the MP Online Internship Program.
