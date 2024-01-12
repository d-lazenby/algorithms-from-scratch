import numpy as np


class LinearRegression:
    """
    Perform linear regression using y_hat = theta0 + theta1 * X
    """
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.theta1 = None
        self.theta0 = None

    def fit(self, inputs, target):
        # init params
        n_features = inputs.shape[1]
        # Initialize from zero
        self.theta1 = np.zeros(n_features)
        self.theta0 = 0

        for _ in range(self.n_iter):
            y_hat = np.dot(inputs, self.theta1) + self.theta0

            d_theta1 = np.dot(inputs.T, (y_hat - target))
            d_theta0 = np.sum(y_hat - target)

            self.theta1 -= self.lr * d_theta1
            self.theta0 -= self.lr * d_theta0

    def predict(self, inputs):
        predictions = np.dot(inputs, self.theta1) + self.theta0
        return predictions


def mse(predictions, y_true):
    return np.mean((y_true - predictions) ** 2)
