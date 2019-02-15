from NN import NeuralNet
import numpy as np
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
targets = np.array([ [0],   [1],   [1],   [0]])

nn = NeuralNet(2,20,1, 0.5, epochs=5000)
for i in range(nn.epochs):
    for j in range(len(inputs)):
        nn.train(inputs[j], targets[j])

for j in range(len(inputs)):
    output = nn.predict(inputs[j])
    print(output)
    print(targets[j])