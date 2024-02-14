import numpy as np

class LogisticRegression:
    
    def __init__(self, add_bias=True, learning_rate=0.1, epsilon=1e-4, max_iters=1e5, verbose=False, lambda_reg=0.01, regularize=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        # Tolerance for the norm of gradients 
        self.max_iters = max_iters                    # Maximum number of iteration of gradient descent
        self.verbose = verbose
        self.lambda_reg = lambda_reg
        self.regularize = regularize                  # Option to turn regularization on/off
        self.logistic = lambda z: 1. / (1 + np.exp(-z))

    def gradient(self, x, y):
        N, D = x.shape
        yh = self.logistic(np.dot(x, self.w))  # predictions
        grad = np.dot(x.T, (yh - y)) / N
        if self.regularize:
            grad += (self.lambda_reg / N) * self.w  # L2 regularization, included conditionally
        return grad
        
    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x, np.ones(N)])
        N, D = x.shape
        self.w = np.zeros(D)
        g = np.inf
        t = 0
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            g = self.gradient(x, y)
            self.w = self.w - self.learning_rate * g
            t += 1
        
        if self.verbose:
            print(f'Terminated after {t} iterations, with norm of the gradient = {np.linalg.norm(g)}')
            print(f'Weights: {self.w}')

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])
        yh = self.logistic(np.dot(x, self.w))
        return yh