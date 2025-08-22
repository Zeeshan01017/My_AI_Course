import pandas as pd
from sklearn.datasets import load_wine

wine_data = load_wine()

wine_data_return_x_y = load_wine(return_X_y=True)

print("wine_data_return_x_y: ",wine_data_return_x_y)
print("wine_data_return_x_y[0]: ",wine_data_return_x_y[0])
print("wine_data_return_x_y[1]: ",wine_data_return_x_y[1])

wine_data_as_frame = load_wine(as_frame=True)
print("data_as_frame: ",wine_data_as_frame)

wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
print("wine_df_dataframe: ",wine_df)

wine_df["target"] = wine_data.target

print("wine_df.head() ", wine_df.head())
print("wine_df.info() ", wine_df.info())
print("wine_df.describe() ",wine_df.describe())
print("wine_df.tail() ",wine_df.tail())

from sklearn.preprocessing import StandardScaler
x = wine_df[wine_data.feature_names].copy()
y =wine_df["target"].copy()

print("X:" , x)
print("y:", y)

scaler = StandardScaler()
scaler.fit(x)

x_scaled = scaler.transform(x.values)
print(x_scaled[0])

from sklearn.model_selection import train_test_split

x_train_scaled, x_test_scaled, y_train, y_test = train_test_split(x_scaled,y,train_size=0.7,random_state=25)

print(f"train size: {round(len(x_train_scaled) / len(x) * 100)}% \n\
      Test size: {round(len(x_test_scaled) / len(x) * 100)}%")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

Logistic_regression = LogisticRegression()
svm = SVC()
tree = DecisionTreeClassifier()

Logistic_regression.fit(x_train_scaled,y_train)
svm.fit(x_train_scaled,y_train)
tree.fit(x_train_scaled,y_train)

log_reg_preds = Logistic_regression.predict(x_test_scaled)
svm_preds = svm.predict(x_test_scaled)
tree_preds = tree.predict(x_test_scaled)

from sklearn.metrics import classification_report

model_preds = {"Logistic Regression": log_reg_preds,
               "Support Vector Machine": svm_preds,
               "Decision Tree": tree_preds}

for model, preds in model_preds.items():
    print(f"{model},Results:\n{classification_report(y_test,preds)}","\n\n") 