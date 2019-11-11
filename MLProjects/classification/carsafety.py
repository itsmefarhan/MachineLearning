import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR

dataset = pd.read_csv('car.csv')
# print(dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(y)
label_e = LabelEncoder()
X[:, 0] = label_e.fit_transform(X[:, 0])
X[:, 1] = label_e.fit_transform(X[:, 1])
X[:, 2] = label_e.fit_transform(X[:, 2])
X[:, 3] = label_e.fit_transform(X[:, 3])
X[:, 4] = label_e.fit_transform(X[:, 4])
X[:, 5] = label_e.fit_transform(X[:, 5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

knn = cross_val_score(KNeighborsClassifier(), X_train, y_train, cv=5)
# print(knn.mean())
# 90.8%
cart = cross_val_score(DecisionTreeClassifier(), X_train, y_train, cv=5)
print(cart.mean())
# 98%
lda = cross_val_score(LinearDiscriminantAnalysis(), X_train, y_train, cv=5)
# print(lda.mean())
# 70.3
gaus = cross_val_score(GaussianNB(), X_train, y_train, cv=5)
# print(gaus.mean())
# 63.2

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(model.score(X_test, y_test))