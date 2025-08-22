import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Machine_learning_practice/student_scores.csv")
print(df.head())
print("df_shape: ",df.shape)

df.plot.scatter(x = "Hours",y = "Scores",title = "Scatter plot of hours and scores percentages")
plt.show()

print("df_corr:   ", df.corr())
print("df_describe:   ", df.describe())
print("df['score']", df["Scores"])
print("df['Hours']", df["Hours"])

y = df["Scores"].values.reshape(-1,1)
x = df["Hours"].values.reshape(-1,1)
print("y: ",y)
print("x: ",x)

print(df["Hours"].values)
print(df["Hours"].values.shape)
print(x.shape)
print(x)

SEED = 42
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y ,test_size=0.2, random_state=SEED)

print(x_train)
print(y_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.intercept_)
print(regressor.coef_)

def calc(slope, intercept, hours):
    return slope*hours+intercept

score = calc(regressor.coef_, regressor.intercept_, 9.5)
print(score)

score = regressor.predict([[9.5]])
print(score)

y_pred = regressor.predict(x_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')