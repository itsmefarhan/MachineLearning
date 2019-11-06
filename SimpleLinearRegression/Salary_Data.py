import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('../Datasets/Salary_Data.csv')
# print(data)

X = data.iloc[:, :-1].values
y= data.iloc[:, -1].values
# print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = LinearRegression()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
# print(y_predict)
# print(y_test)

print(model.score(X_test, y_test))

plt.scatter(data['YearsExperience'], data['Salary'], color='blue')
plt.plot(X_test, y_predict, color='red')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()