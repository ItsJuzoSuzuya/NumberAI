import numpy as np
from Neuron import Neuron
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
       calculateGradientDescent(self.outputLayer.getNeurons(), self.hiddenLayer2)

       cost = calculateCost(self.outputLayer.getNeurons(), input)
       


def calculateCost(outputs: ['Neuron'], expectedResult):
    cost = 0
    for number in range(10):
        if number == expectedResult:
           cost += np.power(outputs[number].getValue() - 1, 2)
        else: cost += np.power(outputs[number].getValue(), 2)
    
    return cost

def calculateGradientDescent(values, weights, biases):
    pass
