import numpy as np

class MultiClassRegression:
    
    def __init__(self, nFeatures, nClasses, regularization_strength=0.1):
        self.W = np.random.randn(nFeatures, nClasses) * 0.01  # Small random values
        self.regularization_strength = regularization_strength
        
    def softmax(self, X):
        z = np.dot(X, self.W)
        z -= np.max(z, axis=1, keepdims=True)  # Numerical stability
        exp_scores = np.exp(z)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probabilities
    
    def predict(self, X):
        return np.argmax(self.softmax(X), axis=1)
    
    def loss_and_gradient(self, X, y):
        probabilities = self.softmax(X)
        N = X.shape[0]
        correct_logprobs = -np.log(probabilities[range(N), y])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * self.regularization_strength * np.sum(self.W * self.W)
        loss = data_loss + reg_loss
        
        dscores = probabilities
        dscores[range(N), y] -= 1
        dscores /= N
        dW = np.dot(X.T, dscores) + self.regularization_strength * self.W
        
        return loss, dW
    
    def fit(self, X, y, lr=0.005, niters=100, verbose=False):
        for i in range(niters):
            loss, dW = self.loss_and_gradient(X, y)
            self.W -= lr * dW
            
            if verbose and i % 10 == 0:
                print(f"Iteration {i}: Loss {loss}")
                
        return self
