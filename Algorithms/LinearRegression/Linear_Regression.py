import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iter=10000, theta=None):
        self.theta = theta
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y):
        X = X.copy()
        m, n = X.shape
        # Scale features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        X = (X - self.mean_) / self.std_

        # Add bias term
        X_bias = np.c_[np.ones(m), X]

        if self.theta is None:
            self.theta = np.random.randn(n + 1) * 0.01  # small initialization

        for i in range(self.max_iter):
            h = X_bias.dot(self.theta)
            error = h - y.values.ravel()
            gradient = (1 / m) * np.dot(X_bias.T, error)
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        X = (X - self.mean_) / self.std_  # scale using training mean/std
        m, n = X.shape
        X_bias = np.c_[np.ones(m), X]
        h = X_bias.dot(self.theta)
        return h

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_total)
