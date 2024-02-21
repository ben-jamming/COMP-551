import numpy as np

class CustomLogisticRegression:
    
    def __init__(self, add_bias=True, learning_rate=0.01, epsilon=1e-15, max_iters=1e5, verbose=False, lambda_reg=0.01, regularize=False, record_training=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        # Tolerance for the norm of gradients 
        self.max_iters = max_iters                    # Maximum number of iteration of gradient descent
        self.verbose = verbose
        self.lambda_reg = lambda_reg
        self.regularize = regularize
        self.record_training = record_training
        self.loss_history = []
        self.gradient_norm_history = []
        
    def compute_loss(self, x, y):
        yh = self.logistic(np.dot(x, self.w))
        loss = -np.mean(y * np.log(yh+self.epsilon) + (1 - y) * np.log(1 - yh +self.epsilon))
        if self.regularize:
            loss += (self.lambda_reg / 2) * np.sum(self.w ** 2)
        return loss
    
    def gradient(self, x, y):
        N, D = x.shape
        yh = self.logistic(np.dot(x, self.w))  # predictions
        grad = np.dot(x.T, (yh - y)) / N
        if self.regularize:
            grad += (self.lambda_reg / N) * self.w  # L2 regularization, included conditionally
        return grad
    
    def logistic(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
    def fit(self, x, y):
        if self.verbose:
            print(f"Shape of x: {x.shape}")
            print(f"Shape of y: {y.shape}")
        if x.ndim == 1:
            x = x.reshape(-1, 1)
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
            if self.record_training:
                self.loss_history.append(self.compute_loss(x, y))
                self.gradient_norm_history.append(np.linalg.norm(g))
            t += 1
        
        if self.verbose:
            print(f'Terminated after {t} iterations, with norm of the gradient = {np.linalg.norm(g)}')
            print(f'Weights: {self.w}')

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])
        yh = self.logistic(np.dot(x, self.w))
        return yh