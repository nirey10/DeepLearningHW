# https://wiseodd.github.io/techblog/2016/06/21/nn-sgd/

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import scipy.io
import numpy as np


X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
import utils

# SwissRoll = scipy.io.loadmat('dataset/SwissRollData.mat')
# y_train = np.array(SwissRoll['Ct'])
# X_train = np.array(SwissRoll['Yt'])
# y_test = np.array(SwissRoll['Cv'])
# X_test = np.array(SwissRoll['Yv'])
#
# X_train = X_train.reshape(X_train.shape[1], X_train.shape[0])
# y_train_hot = y_train.reshape(-y_train.shape[1], y_train.shape[0])
# X_test = X_test.reshape(X_test.shape[1], X_test.shape[0])
# y_test_hot = y_test.reshape(y_test.shape[1], y_test.shape[0])
#
# y_train = np.array([y_train_hot[i][1] for i in range(len(y_train_hot))])
# y_test = np.array([y_test_hot[i][1] for i in range(len(y_test_hot))])
#
# y_train = np.argmax(y_train_hot, axis=1)
# y_test = np.argmax(y_test_hot, axis=1)

minibatch_size = 50
epochs = 10
n_feature = 2
n_class = 2
n_iter = 10
learning_rate = 0.0002  # 1e-4
n_iter = int(len(X_train) / minibatch_size)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

def make_network(n_hidden=3):
    # Initialize weights with Standard Normal random variables
    model = dict(
        W1=np.random.randn(n_feature, n_hidden),
        W2=np.random.randn(n_hidden, n_class)
    )
    return model

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def forward(x, model):
    # Input to hidden
    h = x.dot(model['W1'])
    # ReLU non-linearity
    h[h < 0] = 0

    # Hidden to output
    prob = softmax(h.dot(model['W2']))

    return h, prob

def backward(model, xs, hs, errs):
    """xs, hs, errs contain all informations (input, hidden state, error) of all data in the minibatch"""
    # errs is the gradients of output layer for the minibatch
    dW2 = hs.T.dot(errs)

    # Get gradient of hidden layer
    dh = errs.dot(model['W2'].T)
    dh[hs <= 0] = 0
    dW1 = xs.T.dot(dh)
    return dict(W1=dW1, W2=dW2)

def evaluate(model):
    y_pred = np.zeros_like(y_test)
    for i, x in enumerate(X_test):
        _, prob = forward(x, model)
        y = np.argmax(prob)
        y_pred[i] = y
    accuracy = (y_pred == y_test).sum() / y_test.size
    return accuracy



def SGD_Optimizer(model, X_train, y_train, minibatch_size):
    accuracy_scores = []
    for epoch in range(epochs):
        #print('Iteration {}'.format(iter))

        # Randomize data point
        X_train, y_train = shuffle(X_train, y_train)

        for i in range(0, X_train.shape[0], minibatch_size):
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]
            model = SGD_Step(model, X_train_mini, y_train_mini)

        accuracy = evaluate(model)
        accuracy_scores.append(accuracy)
        print('Epoch {}, accuracy {}'.format(epoch, accuracy))
    return model, accuracy_scores

def SGD_Step(model, X_train, y_train):
    grads = get_minibatch_grad(model, X_train, y_train)
    model = model.copy()
    for layer in grads:
        model[layer] += learning_rate * grads[layer]
    return model

def get_minibatch_grad(model, X_train, y_train):
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)

        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.

        err = y_true - y_pred

        # Accumulate the informations of minibatch
        # x: input
        # h: hidden state
        # err: gradient of output layer
        xs.append(x)
        hs.append(h)
        errs.append(err)

    # Backprop using the informations we get from the current minibatch
    return backward(model, np.array(xs), np.array(hs), np.array(errs))

model = make_network()
trained_model, accuracy_scores = SGD_Optimizer(model, X_train, y_train, minibatch_size)
utils.plot_scores(accuracy_scores)
