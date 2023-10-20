import math
from typing import List

class Neuron():
    def __init__(self, value=None, bias = 0) -> None:
        self._value = value
        self._bias = bias
        self._weights = {}

    def getValue(self):
        return self._value
    
    def setValue(self, value):
        self._value = value
    
    def getBias(self):
        return self._bias
    
    def addWeight(self, weight, destination):
        self._weights[destination] = weight

    def getWeight(self, destination):
        return self._weights.get(destination)
    
    def calculateValue(self, linkedNeurons: List['Neuron']):
        hiddenValue = 0
        for neuron in linkedNeurons:
            value = neuron.getValue()
            weight = neuron.getWeight(self)
            hiddenValue = hiddenValue + (value * weight)

        hiddenValue = sigmoid(hiddenValue+ self.getBias())

        self.setValue(hiddenValue)

def sigmoid(x):
    y = 1/(1 + math.exp(-x))
    return y