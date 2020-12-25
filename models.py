import numpy as np


class myNN:
    def __init__(self, n_feature, n_hidden, n_class):
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.model = dict(
            W1=np.random.randn(n_feature, n_hidden),
            W2=np.random.randn(n_hidden, n_class)
        )


    def forward(self, x):
        # Input to hidden
        h = x.dot(self.model['W1'])
        # ReLU non-linearity
        h[h < 0] = 0
        prob = self.softmax(h.dot(self.model['W2']))
        return h, prob

    def backward(self, xs, hs, errs):
        """xs, hs, errs contain all informations (input, hidden state, error) of all data in the minibatch"""
        dW2 = hs.T.dot(errs)
        # Get gradient of hidden layer
        dh = errs.dot(self.model['W2'].T)
        dh[hs <= 0] = 0
        dW1 = xs.T.dot(dh)
        return dict(W1=dW1, W2=dW2)

    def softmax(self, x):
        return np.exp(x) / np.exp(x).sum()

    def update_layer_weight(self, layer, weights):
        self.model[layer] += weights
        pass
