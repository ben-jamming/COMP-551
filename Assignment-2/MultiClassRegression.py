import numpy as np

class MultiClassRegression:
    
    def __init__(self, nFeatures, nClasses):
        self.W = np.random.randn(nFeatures, nClasses) * 0.01  # Small random values
        
    def softmax(self, X):
        z = np.dot(X, self.W)
        z -= np.max(z, axis=1, keepdims=True)  # Improves numerical stability
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
        loss = data_loss
        
        dscores = probabilities
        dscores[range(N), y] -= 1
        dscores /= N
        dW = np.dot(X.T, dscores)
        
        return loss, dW
    
    def fit(self, X, y, lr=0.005, niters=100, verbose=False):
        for i in range(niters):
            loss, dW = self.loss_and_gradient(X, y)
            self.W -= lr * dW
            
            if verbose and i % 10 == 0:
                print(f"Iteration {i}: Loss {loss}")
                
        return self
    
    def gradient_check(self, X, y, epsilon=1e-5):
        numerical_gradients = np.zeros_like(self.W)
        _, analytic_gradient = self.loss_and_gradient(X, y)
        
        it = np.nditer(self.W, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            old_value = self.W[ix]
            self.W[ix] = old_value + epsilon
            loss_plus_epsilon = self.loss_and_gradient(X, y)[0]
            self.W[ix] = old_value - epsilon
            loss_minus_epsilon = self.loss_and_gradient(X, y)[0]
            self.W[ix] = old_value
            
            numerical_gradients[ix] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
            it.iternext()
   
        return np.abs(numerical_gradients - analytic_gradient) / (np.abs(numerical_gradients) + np.abs(analytic_gradient))