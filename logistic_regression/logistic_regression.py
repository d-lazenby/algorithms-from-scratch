import numpy as np


class LogisticRegression:
    """
    Perform logistic regression using h = 1 / (1 + exp(-theta^T*X)) with
    theta^Tx = theta_0 + sum_j theta_j * x_j (j = 1, ..., n_features)
    """

    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.threshold = 0.5
        self.theta_j = None
        self.theta0 = None

    def fit(self, inputs, target):
        # init params
        n_features = inputs.shape[1]
        # Initialize from zero
        self.theta_j = np.zeros(n_features)
        self.theta0 = 0

        for _ in range(self.n_iter):
            thetaX = np.dot(inputs, self.theta_j) + self.theta0
            h = sigmoid(thetaX)

            d_theta_j = np.dot(inputs.T, (target - h))
            d_theta0 = np.sum(h - target)

            self.theta_j += self.lr * d_theta_j
            self.theta0 += self.lr * d_theta0

    def predict(self, inputs):
        thetaX = np.dot(inputs, self.theta_j) + self.theta0
        predictions = sigmoid(thetaX)
        predictions = np.where(predictions >= self.threshold, 1, 0)
        return predictions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy(predictions, y_true):
    acc = np.sum(y_true == predictions) / len(predictions)
    return round(acc, 4)

