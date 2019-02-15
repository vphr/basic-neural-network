from NN import NeuralNet
import numpy as np
from mlxtend.data import loadlocal_mnist
np.random.seed(0)
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
X, y = loadlocal_mnist(
    images_path='./training_data/train-images.idx3-ubyte',
    labels_path='./training_data/train-labels.idx1-ubyte')
X = X / 255
y = [vectorized_result(y) for y in y]
X_test, y_test = loadlocal_mnist(
    images_path='./training_data/t10k-images.idx3-ubyte',
    labels_path='./training_data/t10k-labels.idx1-ubyte')
X_test = X_test / 255

nn = NeuralNet(784, 300, 10, 0.1, epochs =1)
for j in range(nn.epochs):
    for i in range(len(X)):
        nn.train(X[i], y[i].T)


test_accuracy = 0
test_sample_size = 100
for j in range(test_sample_size):
        output = np.argmax(nn.predict(X_test[j]))
        if(output == y_test[j]): test_accuracy = test_accuracy + 1
        print('predicted: %s actual: %s' % (output, y_test[j]))

print('test accuracy: %s' % ((test_accuracy / test_sample_size )* 100))