import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

data = pd.read_csv('../Datasets/titanic_train.csv')
# print(data.head())

# print('Data length', str(len(data.Survived)))

# Analyzing data
# sns.countplot(x='Survived', data=data) # less survived
# sns.countplot(x='Survived', hue='Sex', data=data) # more female survived
# sns.countplot(x='Survived', hue='Pclass', data=data) # more class 1 survived
# sns.countplot(x='Survived', hue='Cabin', data=data) # more class 1 survived

# data.Age.plot.hist()
# data.Fare.plot.hist()

# plt.show()

# print(data.isnull().sum())
X = data.iloc[:, [2, 4, 6, 7, 8, 9]].values
# print(X)
y = data.iloc[:, 1].values
# print(y)
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 4] = le.fit_transform(X[:, 4])
# print(X)

model = LogisticRegression()
model.fit(X, y)

test_data = pd.read_csv('../Datasets/titanic_test.csv')
# print(test_data.head())
print(test_data.isnull().sum())
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].median())
X_test = test_data.iloc[:, [1, 3, 5,6,7,8]].values
# print(X_test)

X_test[:, 1] = le.fit_transform(X_test[:, 1])

X_test[:, 4] = le.fit_transform(X_test[:, 4])
y_predict = model.predict(X_test)
# print(y_predict)
prediction_data = pd.read_csv('../Datasets/gender_submission.csv')
# print(prediction_data.head())

y_test = prediction_data.iloc[:, 1].values
score = model.score(X_test, y_test)
print(score)

cm = confusion_matrix(y_test, y_predict)
print(cm)
