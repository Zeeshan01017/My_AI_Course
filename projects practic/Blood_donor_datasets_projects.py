import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np


df = pd.read_csv("projects practic/blood_donor_dataset.csv", index_col="donor_id")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())



df = df.drop(columns=['name','email','password','contact_number','created_at','availability'])

encoder = LabelEncoder()
df['city'] = encoder.fit_transform(df['city'])
df['blood_group'] = encoder.fit_transform(df['blood_group'])

X = df.drop(columns=['pints_donated'])
y = df['pints_donated']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Pints Donated")
plt.ylabel("Predicted Pints Donated")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nModel Intercept:", model.intercept_)
print("Model Coefficients:", model.coef_)

print("\nR² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root MSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


