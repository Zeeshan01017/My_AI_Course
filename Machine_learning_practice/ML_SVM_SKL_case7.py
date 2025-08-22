import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

data_link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
col_names = ["variance", "skewness", "curtosis", "entropy", "class"]

bankdata = pd.read_csv(data_link, names=col_names, sep=",", header=None)
bankdata.head()

print(bankdata['class'].unique())
print(bankdata.shape)

print ( " Exploring the Dataset:  bankdata['class'].value_counts()) \n " , bankdata['class'].value_counts())
print ( " Exploring the Dataset:  bankdata['class'].value_counts()) \n " , bankdata['class'].value_counts(normalize=True) )
bankdata['class'].plot.hist();
plt.show()

print("bankdata.describe().T   :    \n" , bankdata.describe().T )

for col in bankdata.columns[:-1]:
    plt.title(col)
    gc=bankdata[col].plot.hist()
    gc.figure.show()

sns.pairplot(bankdata, hue='class')
plt.show()

y = bankdata["class"]
x = bankdata.drop('class',axis=1)

from sklearn.model_selection import train_test_split
SEED = 42
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20,random_state=SEED)

xtrain_samples = x_train.shape[0]
xtest_samples = x_test.shape[0]

print(f'There are {xtrain_samples} samples for training and {xtest_samples} samples for testing.')

from sklearn.svm import SVC
svc = SVC(kernel='linear')

svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test,y_pred)
gg=sns.heatmap(cm, annot=True, fmt='d').set_title('Confusion matrix of linear SVM') 
gg.figure.show() 
print(classification_report(y_test,y_pred))
