import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

dataset = load_breast_cancer()
print(dataset['data'].shape)
data = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
data['target'] = dataset['target']
data['class'] = data['target'].apply(lambda x: dataset['target_names'][x])
# print(data.head())
data = data.drop('target', axis='columns')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

cart = cross_val_score(DecisionTreeClassifier(random_state=0), X_train, y_train, cv=6)
# print(cart.mean())
# 90.9

knn = cross_val_score(KNeighborsClassifier(n_neighbors=7), X_train, y_train, cv=6)
# print(knn.mean())
# 93.2

log = cross_val_score(LogisticRegression(multi_class='auto'), X_train, y_train, cv=6)
# print(log.mean())
# 95.15

gaus = cross_val_score(GaussianNB(), X_train, y_train, cv=6)
# print(gaus.mean())
# 94

model = LogisticRegression(multi_class='auto')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

