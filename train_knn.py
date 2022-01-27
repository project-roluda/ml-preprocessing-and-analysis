from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

X_data = pd.read_csv("data/X_data.csv", header=None)

y_data = pd.read_csv("data/y_data.csv", header=None)
y_data_one_hot = pd.read_csv("data/y_data_one_hot.csv", header=None)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)

clf = KNeighborsClassifier()
clf.fit(X=X_train, y=y_train)
print(clf.score(X_test, y_test))

joblib.dump(clf, "static/knn.sav")