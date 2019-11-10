import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

dataset = load_iris()
# print(dataset.keys())
# print(dataset.data)

X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size = 0.2, random_state = 0)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print(accuracy)

cm = confusion_matrix(y_test, y_predict)
print(cm)