import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('50startups.csv')
# print(data.head())
X = data.iloc[:, :-1].values
y = data.iloc[:, -4].values
# print(X)

# Encode categorical data
le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])

ohe = OneHotEncoder(categorical_features=[3])
X = ohe.fit_transform(X).toarray()
# print(X)

# Avoid dummy variable trap (encoder created 2 cols, but we only need 1 col)
X = X[:, 1:]
# print(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting MLR to training set
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test dataset result
y_predicted = model.predict(X_test)
# print(y_predicted)
# print(y_test)
print(model.score(X_test, y_test))