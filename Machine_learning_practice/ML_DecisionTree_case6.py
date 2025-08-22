import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("Machine_learning_practice/diabetes.csv",header=1,names=col_names)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols]
y = pima.label

x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.30, random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print("Accuracy",metrics.accuracy_score(y_test,y_pred))

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetsV1.png')
Image(graph.create_png)

clf = DecisionTreeClassifier(criterion="entropy",max_depth=3)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print("Accuracy",metrics.accuracy_score(y_test,y_pred))
dot_data = StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,
                special_characters=True,feature_names=feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetesV2.png')
Image(graph.create_png())


