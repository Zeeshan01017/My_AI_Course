# Machine learning classification Loan dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Models that are applying
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Load Dataset
df = pd.read_csv("Practical–DevelopmentProgrammingBatch_2/loan_data.csv")
print("Dataset Shape:", df.shape)
print(df.head())
print("Info(): ", df.info())
print("datatype(): ", df.dtypes)
print("describe(): ", df.describe())

# Data Preprocessing
print("\nMissing Values:\n", df.isnull().sum())

# Encode categorical column 'purpose'
le = LabelEncoder()
df['purpose'] = le.fit_transform(df['purpose'])

# Features and Target
X = df.drop(columns=['not.fully.paid'])
y = df['not.fully.paid']  # 1 = loan not fully paid

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Exploratory Data Analysis
plt.figure(figsize=(10,6))
sns.countplot(x='not.fully.paid', data=df)
plt.title("Class Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Train Multiple Models using Logistic regression, Randome Forest, Gradient decent and Support 
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "Support Vector Machine": SVC(probability=True, kernel='rbf')
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:,1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba) if proba is not None else None
    
    print(f"\n=== {name} ===")
    print("Accuracy:", round(acc,4))
    if auc:
        print("ROC-AUC:", round(auc,4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "ROC-AUC": auc
    })

# Compare Results
results_df = pd.DataFrame(results)
print("\nModel Comparison:\n", results_df.sort_values(by="Accuracy", ascending=False))

# Plot Comparison
plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="Accuracy", data=results_df.sort_values(by="Accuracy", ascending=False))
plt.xticks(rotation=15)
plt.title("Model Accuracy Comparison")
plt.show()
