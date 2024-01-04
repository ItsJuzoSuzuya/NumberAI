import numpy as np
import math
from keras.datasets import mnist

(train_X, trainNumbers), (test_X, testNumbers) = mnist.load_data()
train_X = train_X / 1000
np.random.seed(0)

class Layer:
    """Neuronal Layer including the incoming weights, biases of each neuron in the layer
    and the outputs or values of the neurons themselfs"""
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.outputs = []
    def forward(self, inputs):
        """Calculates the outputs multiplying each incoming weight with their connected 
        neurons output aka value"""
        self.outputs = np.dot(inputs, self.weights) + self.biases


class Loss:
    """Loss or cost of accuracy of the current neuronal function"""
    def calculate(self, output, y):
        "Using the actual output and the expected output to calculate the inacuraccy"
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1+1e-7)




class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        propabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = propabilities


hiddenLayer1 = Layer(784, 16)
hiddenLayer2 = Layer(16, 16)
outputLayer = Layer(16, 10)

activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_Softmax()

layer1.forward(train_X[12].flatten())
activation1.forward(layer1.outputs)

layer2.forward(activation1.outputs)
activation2.forward(layer2.outputs)

layer3.forward(activation2.outputs)
activation3.forward(layer3.outputs)

print(np.sum(activation3.outputs))