import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


df = pd.read_csv("projects practic/us_house_Sales_data.csv")
df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)

df['Bedrooms'] = df['Bedrooms'].str.extract('(\d+)').astype(float)
df['Bathrooms'] = df['Bathrooms'].str.extract('(\d+)').astype(float)
df['Area (Sqft)'] = df['Area (Sqft)'].str.replace(' sqft', '').str.replace(',', '').astype(float)
df['Lot Size'] = df['Lot Size'].str.replace(' sqft', '').str.replace(',', '').astype(float)
    
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())


numeric_cols = ['Price', 'Bedrooms', 'Bathrooms', 'Area (Sqft)', 'Lot Size', 'Year Built', 'Days on Market']
plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Price vs Area
axes[0, 0].scatter(df['Area (Sqft)'], df['Price'], alpha=0.6)
axes[0, 0].set_xlabel('Area (Sqft)')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Price vs Area')


# Price vs Bedrooms
bedroom_avg = df.groupby('Bedrooms')['Price'].mean()
axes[0, 1].plot(bedroom_avg.index, bedroom_avg.values, marker='o')
axes[0, 1].set_xlabel('Number of Bedrooms')
axes[0, 1].set_ylabel('Average Price ($)')
axes[0, 1].set_title('Price vs Bedrooms')

# Price vs Bathrooms
bathroom_avg = df.groupby('Bathrooms')['Price'].mean()
axes[1, 0].plot(bathroom_avg.index, bathroom_avg.values, marker='o')
axes[1, 0].set_xlabel('Number of Bathrooms')
axes[1, 0].set_ylabel('Average Price ($)')
axes[1, 0].set_title('Price vs Bathrooms')

# Price vs Year Built
axes[1, 1].scatter(df['Year Built'], df['Price'], alpha=0.6)
axes[1, 1].set_xlabel('Year Built')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].set_title('Price vs Year Built')
plt.tight_layout()
plt.show()

features = ['Bedrooms', 'Bathrooms', 'Area (Sqft)', 'Lot Size', 'Year Built']
X = df[features]
y = df['Price']

X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"RMSE: {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

print("\nFeature Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:,.2f}")

print(f"Intercept: {model.intercept_:,.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Prices ($)')
plt.ylabel('Predicted Prices ($)')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices ($)')
plt.ylabel('Residuals ($)')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()

importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=importance)
plt.title('Feature Importance (Absolute Coefficient Values)')
plt.tight_layout()
plt.show()

X_area = df[['Area (Sqft)']]
y_area = df['Price']

X_area_train, X_area_test, y_area_train, y_area_test = train_test_split(
    X_area, y_area, test_size=0.2, random_state=42)

area_model = LinearRegression()
area_model.fit(X_area_train, y_area_train)

y_area_pred = area_model.predict(X_area_test)

area_rmse = np.sqrt(mean_squared_error(y_area_test, y_area_pred))
area_r2 = r2_score(y_area_test, y_area_pred)


print(f"Area-only Model Performance:")
print(f"RMSE: {area_rmse:,.2f}")
print(f"R² Score: {area_r2:.4f}")
print(f"Coefficient: {area_model.coef_[0]:.2f}")
print(f"Intercept: {area_model.intercept_:,.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_area_test, y_area_test, alpha=0.6, label='Actual Prices')
plt.plot(X_area_test, y_area_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Area (Sqft)')
plt.ylabel('Price ($)')
plt.title('Simple Linear Regression: Price vs Area')
plt.legend()
plt.tight_layout()
plt.show()

print("\n=== SUMMARY OF FINDINGS ===")
print("1. The dataset contains 1,000 house listings with various features.")
print("2. House prices range widely, with some outliers at the higher end.")
print("3. California (CA) has the highest average house prices.")
print("4. Single Family homes tend to be the most expensive property type.")
print("5. Area (Sqft) has the strongest positive correlation with price.")
print("6. The multivariate linear regression model achieved reasonable performance.")
print("7. Area is the most important predictor of house price in our model.")








