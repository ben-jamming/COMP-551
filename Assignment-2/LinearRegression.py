import numpy as np

class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        self.w = None

    def fit(self, x, y):
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])  # Add bias term
        self.w = np.linalg.lstsq(x, y, rcond=None)[0]
        return self

    def predict(self, x):
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])
        return x @ self.w