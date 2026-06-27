# MP Online Internship Assignment - Adult Census Income Dataset
# Complete Solution: Tasks 1-5


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix, roc_curve)

# TASK 1: DATASET UNDERSTANDING 

print("=" * 60)
print("TASK 1: DATASET UNDERSTANDING")
print("=" * 60)

df = pd.read_csv("adult.csv")  

print("\n📌 Shape of Dataset:", df.shape)
print("\n📌 First 5 Rows:\n", df.head())
print("\n📌 Column Names:\n", df.columns.tolist())
print("\n📌 Data Types:\n", df.dtypes)
print("\n📌 Statistical Summary:\n", df.describe(include='all'))
print("\n📌 Target Variable Distribution:\n", df['income'].value_counts())

# Visualize target distribution
plt.figure(figsize=(6, 4))
df['income'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
plt.title("Income Distribution (Target Variable)")
plt.xlabel("Income Category")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("task1_target_distribution.png")
plt.show()
print("✅ Task 1 Complete\n")


# TASK 2: DATA CLEANING 

print("=" * 60)
print("TASK 2: DATA CLEANING")
print("=" * 60)

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

print("\n📌 Missing Values Before Cleaning:\n", df.isnull().sum())

# Fill missing values with mode (most frequent value) for categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\n📌 Missing Values After Cleaning:\n", df.isnull().sum())

# Remove duplicate rows
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"\n📌 Duplicates Removed: {before - after} rows")

# Strip extra whitespace from string columns
df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

print("\n📌 Cleaned Dataset Shape:", df.shape)
print("✅ Task 2 Complete\n")


# TASK 3: FEATURE ENGINEERING 

print("=" * 60)
print("TASK 3: FEATURE ENGINEERING")
print("=" * 60)

# 3a. Encode target variable
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# 3b. Label Encode all categorical columns
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("\n📌 Categorical Columns to Encode:", cat_cols)

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\n📌 Encoded Dataset (first 5 rows):\n", df.head())

# 3c. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("task3_correlation_heatmap.png")
plt.show()

# 3d. Feature Scaling
X = df.drop('income', axis=1)
y = df['income']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3e. Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📌 Training Set Size: {X_train.shape}")
print(f"📌 Test Set Size:     {X_test.shape}")
print("✅ Task 3 Complete\n")


# TASK 4: MODEL BUILDING 

print("=" * 60)
print("TASK 4: MODEL BUILDING - CLASSIFICATION ALGORITHMS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "SVM":                 SVC(probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n🔄 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results[name] = {
        "Accuracy":  round(acc, 4),
        "Precision": round(prec, 4),
        "Recall":    round(rec, 4),
        "F1 Score":  round(f1, 4),
        "ROC-AUC":   round(auc, 4)
    }
    print(f"   Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

print("✅ Task 4 Complete\n")


# TASK 5: PERFORMANCE EVALUATION 

print("=" * 60)
print("TASK 5: PERFORMANCE EVALUATION")
print("=" * 60)

# 5a. Summary Table
results_df = pd.DataFrame(results).T
print("\n📌 Performance Comparison Table:\n")
print(results_df.to_string())

# 5b. Save table to CSV
results_df.to_csv("task5_performance_results.csv")
print("\n📌 Results saved to 'task5_performance_results.csv'")

# 5c. Bar Chart Comparison
results_df[["Accuracy", "F1 Score", "ROC-AUC"]].plot(
    kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black')
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.ylim(0.5, 1.0)
plt.xticks(rotation=30, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("task5_model_comparison.png")
plt.show()

# 5d. ROC Curves for all models
plt.figure(figsize=(10, 7))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = results[name]["ROC-AUC"]
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("task5_roc_curves.png")
plt.show()

# 5e. Best Model
best_model = results_df["F1 Score"].idxmax()
print(f"\n🏆 Best Performing Model (by F1 Score): {best_model}")
print(f"   {results[best_model]}")

# 5f. Confusion Matrix for Best Model
best = models[best_model]
y_pred_best = best.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['<=50K', '>50K'],
            yticklabels=['<=50K', '>50K'])
plt.title(f"Confusion Matrix - {best_model}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("task5_confusion_matrix.png")
plt.show()

print(f"\n📌 Classification Report ({best_model}):\n")
print(classification_report(y_test, y_pred_best, target_names=['<=50K', '>50K']))

print("\n" + "=" * 60)
print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("""
Files Generated:
  📊 task1_target_distribution.png
  🗺️  task3_correlation_heatmap.png
  📈 task5_model_comparison.png
  📉 task5_roc_curves.png
  🔵 task5_confusion_matrix.png
  📋 task5_performance_results.csv
""")
