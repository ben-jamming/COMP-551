import numpy as np

class MultiClassRegression:
    
    def __init__(self, nFeatures, nClasses):
        self.W = np.random.rand(nFeatures, nClasses)
        
    def predict(self, X):
        y_pred = np.exp(np.matmul(X, self.W))
        return y_pred / np.sum(y_pred, axis=1).reshape(X.shape[0], 1)
    
    def gradient(self, X, y):
        return np.matmul(X.T, self.predict(X) - y) / X.shape[0]
    
    def cross_entropy(self, y, X):
        return -np.sum(y * np.log(self.predict(X)))
        
    def fit(self, X, y, lr=0.005, niters=100):
        for i in range(niters):
            self.W -= lr * self.gradient(X, y)
        
