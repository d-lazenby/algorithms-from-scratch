import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from linear_regression import LinearRegression


def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], y, color='b', marker='o', s=30)
# plt.show()

regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)

preds = regressor.predict(X_test)

mse_value = mse(y_test, preds)
print(mse_value)
