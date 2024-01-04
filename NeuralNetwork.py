import numpy as np
from neuron import Neuron
from Layers.InputLayer import InputLayer
from Layers.OutputLayer import OutputLayer
from Layers.HiddenLayer import HiddenLayer

class NeuralNetwork():
    def __init__(self, firstInputImage, firstNumber) -> None:
        self.inputLayer = InputLayer(firstInputImage)
        self.hiddenLayer1 = HiddenLayer(self.inputLayer)
        self.hiddenLayer2 = HiddenLayer(self.hiddenLayer1)
        self.outputLayer = OutputLayer(self.hiddenLayer2)
        self.train(firstNumber)

    def train(self, input):
        weightInaccuracy, biasInaccuracy = calculateGradientDescent(self.hiddenLayer2.values, self.hiddenLayer2.weights, self.outputLayer.biases, self.outputLayer.values)
        self.hiddenLayer1.weights -= weightInaccuracy
        self.outputLayer.biases -= biasInaccuracy 

        cost = self.calculateCost(self.outputLayer.getNeurons(), input)
        print(cost)

    def calculateCost(outputs: ['Neuron'], expectedResult):
        cost = 0
        for number in range(10):
            if number == expectedResult:
                cost += np.power(outputs[number].getValue() - 1, 2)
            else: cost += np.power(outputs[number].getValue(), 2)
        
        return cost
    
    def backward(self):
        self.output 

def calculateGradientDescent(values, weights, biases, outputLayer):

    valueToCost = 2*(1-outputLayer)
    sigmoidDiff = sigmoid_derivative(np.dot(values, weights.T) + biases)
    weightsTopPreSigmoid = np.dot(values, weights.T) + biases

    weightGradientDescent = valueToCost*sigmoidDiff*weightsTopPreSigmoid
    
    biasGradientDescent = -2*(1-outputLayer)

    return weightGradientDescent, biasGradientDescent

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


