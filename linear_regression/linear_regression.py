import numpy as np


class LinearRegression:
    """
    Perform linear regression using h = theta_0 + sum_j theta_j * x_j (j = 1, ..., n_features)
    """
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.theta_j = None
        self.theta0 = None

    def fit(self, inputs, target):
        # init params
        n_features = inputs.shape[1]
        # Initialize from zero
        self.theta_j = np.zeros(n_features) * 10 ** -5
        self.theta0 = 10 ** -5

        for _ in range(self.n_iter):
            h = self.theta0 + np.dot(inputs, self.theta_j)

            d_theta_j = np.dot(inputs.T, (h - target))
            d_theta0 = np.sum(h - target)

            self.theta_j -= self.lr * d_theta_j
            self.theta0 -= self.lr * d_theta0

    def predict(self, inputs):
        predictions = self.theta0 + np.dot(inputs, self.theta_j)
        return predictions


def mse(predictions, target):
    return np.mean((target - predictions) ** 2)
