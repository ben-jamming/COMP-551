import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

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
    
    def predict_top_k(self, X, k=3):
        """
        Predict the top k classes for each sample in X.
        """
        probabilities = self.softmax(X)
        top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]

        return top_k_predictions
    
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
    
    def fit(self, X, y, lr=0.005, niters=100, validation_split=0):
        validation_split = int(X.shape[0] * validation_split)
        X_train, y_train = X[:-validation_split], y[:-validation_split]
        X_val, y_val = X[-validation_split:], y[-validation_split:]
        loss_history = []
        validation_loss = []

        # Create a tqdm object with a dynamic description
        pbar = tqdm(range(niters), desc='Initializing')
        best_weights = self.W

        for i in pbar:
            loss, dW = self.loss_and_gradient(X_train, y_train)
            self.W -= lr * dW

            if validation_split > 0:
                val_loss, val_dw = self.loss_and_gradient(X_val, y_val)
                validation_loss.append(val_loss)
            else:
                val_loss = 'N/A'  # No validation loss if validation_split is 0
                validation_loss.append(loss)
            
            if validation_loss[-1] != min(validation_loss):
                best_weights = self.W
            
            # if the min validation loss is more than 100 iterations ago, stop training
            if i > 10 and min(validation_loss) not in validation_loss[-100:]:
                print(f"Stopping early at iteration {i}, min validation loss: {min(validation_loss):.4f}")
                print(f"Best weights found at iteration {validation_loss.index(min(validation_loss))}")
                print("validation loss did not improve for 100 iterations")
                break
            
            loss_history.append(loss)
            # Update tqdm description
            if i % 10 == 0:
                pbar.set_description(f"Iter {i}, Loss: {float(loss):.4f}, Val Loss: {float(val_loss):.4f}")
        
        self.W = best_weights

        return self, loss_history, validation_loss
    
    
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