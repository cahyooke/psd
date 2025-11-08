import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden=1000, activation_function=sigmoid_activation):
        self.n_hidden = n_hidden
        self.activation_function = activation_function
        self.input_weights = None
        self.bias = None
        self.output_weights = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # One-hot encoding untuk y jika perlu
        if len(y.shape) == 1:
            y_onehot = np.zeros((y.size, y.max() + 1))
            y_onehot[np.arange(y.size), y] = 1
        else:
            y_onehot = y

        n_features = X.shape[1]
        self.input_weights = np.random.randn(n_features, self.n_hidden)
        self.bias = np.random.randn(self.n_hidden)

        H = self.activation_function(np.dot(X, self.input_weights) + self.bias)
        self.output_weights = np.dot(np.linalg.pinv(H), y_onehot)

        return self

    def predict(self, X):
        X = np.asarray(X)
        H = self.activation_function(np.dot(X, self.input_weights) + self.bias)
        y_pred = np.dot(H, self.output_weights)
        return np.argmax(y_pred, axis=1)
