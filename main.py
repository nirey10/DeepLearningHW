# https://wiseodd.github.io/techblog/2016/06/21/nn-sgd/

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import scipy.io
import numpy as np
from models import myNN
from optimizers import SGD_Optimizer
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
epochs = 20
n_feature = 2
n_class = 2
n_iter = 10
n_hidden = 3
learning_rate = 0.0002  # 1e-4
n_iter = int(len(X_train) / minibatch_size)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = myNN(n_feature, n_hidden, n_class)
optimizer = SGD_Optimizer(model, X_train, y_train, X_test, y_test, minibatch_size, epochs, learning_rate, n_class)
accuracy_scores = optimizer.fit()
utils.plot_scores(accuracy_scores)
