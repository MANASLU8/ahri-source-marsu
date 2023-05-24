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
import time
import joblib

models = [ExtraTreesClassifier(), RandomForestClassifier()]

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')
X_train.drop(columns=X_train.columns[0], inplace=True)
X_test.drop(columns=X_test.columns[0], inplace=True)
y_train.drop(columns=y_train.columns[0], inplace=True)
y_test.drop(columns=y_test.columns[0], inplace=True)

X_train2 = pd.read_csv('X_train2.csv')
X_test2 = pd.read_csv('X_test2.csv')
y_train2 = pd.read_csv('y_train2.csv')
y_test2 = pd.read_csv('y_test2.csv')
X_train2.drop(columns=X_train2.columns[0], inplace=True)
X_test2.drop(columns=X_test2.columns[0], inplace=True)
y_train2.drop(columns=y_train2.columns[0], inplace=True)
y_test2.drop(columns=y_test2.columns[0], inplace=True)

models_info = {}
models_info2 = {}
i = 1

def print_accuracy(model,i):
    start = time.time()
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_rep =classification_report(y_test, y_pred)
    print(f'Model: \n{model}\n'
          f'Confusion matrix: \n{conf_matrix}\n'
          f'Classification report: \n{class_rep}\n'
         )
    result_time = time.time() - start
    
    if i == 1:
      models_info['LR time:'] = result_time
      models_info['LR CR: '] = classification_report(y_test, y_pred)
    if i == 2:
      models_info['RFC time:'] = result_time
      models_info['RFC CR: '] = classification_report(y_test, y_pred)

def print_accuracy2(model,i):
    start = time.time()
    y_pred = model.predict(X_test2)
    conf_matrix = confusion_matrix(y_test2, y_pred)
    class_rep =classification_report(y_test2, y_pred)
    print(f'Model: \n{model}\n'
          f'Confusion matrix: \n{conf_matrix}\n'
          f'Classification report: \n{class_rep}\n'
         )
    result_time = time.time() - start
    
    if i == 1:
      models_info2['LR time:'] = result_time
      models_info2['LR CR: '] = classification_report(y_test2, y_pred)
    if i == 2:
      models_info2['RFC time:'] = result_time
      models_info2['RFC CR: '] = classification_report(y_test2, y_pred)


for mod in models:
  model = mod.fit(X_train, y_train)
  if i == 1:
    joblib.dump(model,'LR.joblib')
  if i == 2:
    joblib.dump(model,'RFC.joblib')
  print_accuracy(model,i)
  i = i + 1

i = 1

for mod in models:
  model = mod.fit(X_train2, y_train2)
  if i == 1:
    joblib.dump(model,'LR2.joblib')
  if i == 2:
    joblib.dump(model,'RFC2.joblib')
  print_accuracy2(model,i)
  i = i + 1

def save_info(text, name):
  with open(name, "w") as file:
      for key, value in text.items():
          file.write(f"{key}: {value}\n")
  print("Сохранение: result.txt")

save_info(models_info,'result.txt')
save_info(models_info2,'result2.txt')
