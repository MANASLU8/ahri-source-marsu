import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('winequality-red.csv')
last_150 = df.tail(150)
last_150.to_csv('last_150.csv', index=False)
without_last_150 = df.iloc[:-150]
without_last_150.to_csv('without_last_150.csv', index=False)

X = df.drop(['quality'], axis=1)
y = df['quality']

X, y = SMOTE(random_state=42).fit_resample(X, y)

scaler = StandardScaler()
stand_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
normalized_df=(X-X.mean())/X.std()

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X = without_last_150.drop(['quality'], axis=1)
y = without_last_150['quality']

X, y = SMOTE(random_state=42).fit_resample(X, y)

scaler = StandardScaler()
stand_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
normalized_df=(X-X.mean())/X.std()

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')
X_train2.to_csv('X_train2.csv')
X_test2.to_csv('X_test2.csv')
y_train2.to_csv('y_train2.csv')
y_test2.to_csv('y_test2.csv')
