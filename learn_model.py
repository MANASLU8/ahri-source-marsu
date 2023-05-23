from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

models = [LogisticRegression(), RandomForestClassifier()]

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')
X_train.drop(columns=X_train.columns[0], inplace=True)
X_test.drop(columns=X_test.columns[0], inplace=True)
y_train.drop(columns=y_train.columns[0], inplace=True)
y_test.drop(columns=y_test.columns[0], inplace=True)

def print_accuracy(model):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_rep =classification_report(y_test, y_pred)
    print(f'Model: \n{model}\n'
          f'Confusion matrix: \n{conf_matrix}\n'
          f'Classification report: \n{class_rep}\n'
         )


for mod in models:
	model = mod.fit(X_train, y_train)
	print_accuracy(model)
