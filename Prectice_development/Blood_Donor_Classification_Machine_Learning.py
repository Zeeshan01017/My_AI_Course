# BLOOD DONOR CLASSIFICATION Machine learning

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

# Load Data
df = pd.read_csv("Prectical_Development_programing_Assesment/blood_donor_dataset.csv")
print(df.head(), "\n")
print(df.info())

# Drop Unnecessary Columns
drop_cols = ['donor_id','name','email','password','contact_number','created_at']
df.drop(columns=drop_cols, inplace=True)

# Encode Categorical Variables
le_avail = LabelEncoder()
df['availability'] = le_avail.fit_transform(df['availability'])  # Yes=1, No=0

le_blood = LabelEncoder()
df['blood_group'] = le_blood.fit_transform(df['blood_group'])

le_city = LabelEncoder()
df['city'] = le_city.fit_transform(df['city'])

# EDA (Visualizations)
plt.figure(figsize=(6,4))
sns.countplot(x='availability', data=df, palette="Set2")
plt.title("Target Variable Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

sns.boxplot(x='availability', y='months_since_first_donation', data=df)
plt.title("Months Since First Donation vs Availability")
plt.show()

# Feature/Target Split
X = df.drop('availability', axis=1)
y = df['availability']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Random Forest with Hyperparameter Tuning
params = {
    'n_estimators': [100,200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2,5]
}
rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# Evaluation
y_pred = best_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
y_prob = best_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

