import random
import numpy as np
from neuron import Neuron
from typing import List

class Layer():
    def __init__(self) -> None:
        self._neurons: List[Neuron] = []
        self.values = np.empty((0,))
        self.weights = np.empty((0,))
        self.biases = np.empty((0,))

    def addNeuron(self, neuron: 'Neuron'):
        self._neurons.append(neuron)
        self.values = np.append(self.values, neuron.getValue())

    def addToBiases(self, bias):
        self.biases = np.append(self.biases, bias)

    def getNeurons(self):
        return self._neurons
    
    def clearNeurons(self):
        self._neurons.clear()
    
    def connectLayer(self, nextLayer: 'Layer'):
        numNeurons = len(self.getNeurons())
        for neuron in self.getNeurons():
            weightsToAppend = [] 
            for destination in nextLayer.getNeurons():
                randomWeight = random.uniform(-1,1)
                neuron.addWeight(randomWeight, destination)
                weightsToAppend.append(randomWeight)
                
            self.weights = np.append(self.weights, weightsToAppend)
        
        self.weights = self.weights.reshape(-1, numNeurons)

    def calculateValue(self, previousLayer: 'Layer'):
        values = previousLayer.values.reshape(len(previousLayer.getNeurons()), 1)
        biases = self.biases
        weights = previousLayer.weights


        result = np.dot(weights, values)
        result = result.reshape(-1)
        self.values = sigmoid(result + biases)
        
        neurons = self.getNeurons()
        for i in range(len(neurons)):
            neurons[i].setValue(result[i])


        
def sigmoid(x):
    y = 1/(1 + np.exp(-x))
    return y
                




