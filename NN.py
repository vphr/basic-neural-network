import numpy as np
from mlxtend.data import loadlocal_mnist
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    return z*(1-z)
def sigmoid_p(x):
        return sigmoid(x) * (1-sigmoid(x))


class NeuralNet:
    def __init__(self, iNodes, hNodes, oNodes, lr, epochs):
        self.input_nodes = iNodes
        self.hidden_nodes = hNodes
        self.output_nodes = oNodes
        self.lr = lr
        self.epochs = epochs
        self.wih = 2 * np.random.random((self.hidden_nodes, self.input_nodes)) -1
        self.who = 2 * np.random.random((self.output_nodes, self.hidden_nodes)) -1


        self.hb = 2 * np.random.random((self.hidden_nodes, 1)) -1
        self.ob = 2 * np.random.random((self.output_nodes, 1)) -1
        

    def train(self, x, y):
        inputs = np.array(x, ndmin=2).T
        targets = np.array(y, ndmin=2).T

        hidden = np.dot(self.wih, inputs) + self.hb
        hidden_sig = sigmoid(hidden)
        output = np.dot(self.who, hidden_sig) + self.ob
        output_sig = sigmoid(output)
        output_error = targets - output_sig
        output_gradient = output_error * sigmoid_prime(output_sig)
        self.ob += output_gradient
        output_gradient = self.lr * np.dot(output_gradient, hidden_sig.T)
        self.who += output_gradient
        hidden_error = np.dot(self.who.T, output_error)
        hidden_gradient = hidden_error * sigmoid_prime(hidden_sig)
        self.hb += hidden_gradient
        self.wih += self.lr * np.dot(hidden_gradient, inputs.T)
    def predict(self, x):
        inputs = np.array(x, ndmin=2).T
        hidden = np.dot(self.wih, inputs) + self.hb
        hidden_sig = sigmoid(hidden)

        output = np.dot(self.who, hidden_sig) + self.ob
        output_sig = sigmoid(output)
        return output_sig
