import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = 0.01 * np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims= True)
        self.dinputs = np.dot(dvalues, self.weights.T)