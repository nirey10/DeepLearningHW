from sklearn.utils import shuffle
import numpy as np


class SGD_Optimizer:
    def __init__(self, model, X_train, y_train, X_test, y_test, minibatch_size, epochs, learning_rate, n_class, do_shuffle=True):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.n_class = n_class
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        if do_shuffle:
            self.X_train, self.y_train = shuffle(X_train, y_train)
        else:
            self.X_train = X_train
            self.y_train = y_train

    def fit(self):
        accuracy_scores = []
        for epoch in range(self.epochs):
          for i in range(0, self.X_train.shape[0], self.minibatch_size):
              X_train_mini = self.X_train[i:i + self.minibatch_size]
              y_train_mini = self.y_train[i:i + self.minibatch_size]
              self.SGD_Step(X_train_mini, y_train_mini)

          accuracy = self.evaluate()
          accuracy_scores.append(accuracy)
          print('Epoch {}, accuracy {}'.format(epoch, accuracy))
        return accuracy_scores

    def SGD_Step(self, X_train_mini, y_train_mini):
        grads = self.get_minibatch_grad(X_train_mini, y_train_mini)
        for layer in grads:
          self.model.update_layer_weight(layer, self.learning_rate * grads[layer])

    def get_minibatch_grad(self, X_train, y_train):
        xs, hs, errs = [], [], []

        for x, cls_idx in zip(X_train, y_train):
            h, y_pred = self.model.forward(x)

            y_true = np.zeros(self.n_class)
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
        return self.model.backward(np.array(xs), np.array(hs), np.array(errs))

    def evaluate(self):
        y_pred = np.zeros_like(self.y_test)
        for i, x in enumerate(self.X_test):
            _, prob = self.model.forward(x)
            y = np.argmax(prob)
            y_pred[i] = y
        accuracy = (y_pred == self.y_test).sum() / self.y_test.size
        return accuracy