import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('projects practic/us_house_Sales_data.csv')

# Data cleaning function
def clean_data(df):
    # Clean price column
    df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)
    
    # Clean other numeric columns
    df['Bedrooms'] = df['Bedrooms'].str.extract('(\d+)').astype(float)
    df['Bathrooms'] = df['Bathrooms'].str.extract('(\d+)').astype(float)
    df['Area (Sqft)'] = df['Area (Sqft)'].str.replace(' sqft', '').str.replace(',', '').astype(float)
    df['Lot Size'] = df['Lot Size'].str.replace(' sqft', '').str.replace(',', '').astype(float)
    
    return df

# Clean the data
df = clean_data(df)

# Create price categories for classification
def create_price_categories(price):
    if price < 300000:
        return 'Low'
    elif price < 700000:
        return 'Medium'
    elif price < 1200000:
        return 'High'
    else:
        return 'Premium'

df['Price_Category'] = df['Price'].apply(create_price_categories)

# Prepare features and target
features = ['Bedrooms', 'Bathrooms', 'Area (Sqft)', 'Lot Size', 'Year Built']
X = df[features]
y = df['Price_Category']

# Handle missing values
X = X.fillna(X.mean())

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # For multiclass ROC AUC (one-vs-rest)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Print results
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# Compare model performance
print("\n=== MODEL COMPARISON ===")
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1-Score': [results[m]['f1_score'] for m in results],
    'ROC AUC': [results[m]['roc_auc'] for m in results]
})

print(comparison_df)

# Visualize metrics comparison
plt.figure(figsize=(12, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
x_pos = np.arange(len(metrics))

for i, model_name in enumerate(results.keys()):
    values = [results[model_name][metric.lower().replace(' ', '_')] for metric in metrics]
    plt.plot(x_pos, values, marker='o', label=model_name, linewidth=2)

plt.xticks(x_pos, metrics)
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Confusion matrix for the best model (based on F1-score)
best_model_name = max(results, key=lambda x: results[x]['f1_score'])
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['y_pred']

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Feature importance for Random Forest
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.show()

# ROC Curve for multiclass (one-vs-rest)
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange']

for i, model_name in enumerate(results.keys()):
    y_pred_proba = results[model_name]['y_pred_proba']
    
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for class_idx in range(len(le.classes_)):
        fpr[class_idx], tpr[class_idx], _ = roc_curve(y_test_bin[:, class_idx], y_pred_proba[:, class_idx])
        roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])
    
    # Plot ROC curves
    for class_idx, color in zip(range(len(le.classes_)), colors):
        plt.plot(fpr[class_idx], tpr[class_idx], color=color, lw=2,
                label=f'{model_name} - {le.classes_[class_idx]} (AUC = {roc_auc[class_idx]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multiclass Classification')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Cross-validation scores
print("\n=== CROSS-VALIDATION RESULTS ===")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
    print(f"{name} - F1 Score CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Final summary
print("\n=== FINAL SUMMARY ===")
print(f"Best model based on F1-Score: {best_model_name}")
print(f"Best F1-Score: {results[best_model_name]['f1_score']:.4f}")
print(f"Best Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"Best Precision: {results[best_model_name]['precision']:.4f}")
print(f"Best Recall: {results[best_model_name]['recall']:.4f}")