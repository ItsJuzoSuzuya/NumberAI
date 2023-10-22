import random
import numpy as np
from Neuron import Neuron
from typing import List

class Layer():
    def __init__(self, isInputLayer = False) -> None:
        self._neurons: List[Neuron] = []
        if notInputLayer:
            self.calculateValue()

    def addNeuron(self, neuron: 'Neuron'):
        self._neurons.append(neuron)

    def getNeurons(self):
        return self._neurons
    
    def clearNeurons(self):
        self._neurons.clear()
    
    def connectLayer(self, nextLayer: 'Layer'):
        for neuron in self.getNeurons():
            for destination in nextLayer.getNeurons():
                randomWeight = random.uniform(-1,1)
                neuron.addWeight(randomWeight, destination)

    def calculateValue(self, previousLayer: 'Layer'):
        valueMatrix = np.array([neuron.getValue() for neuron in previousLayer.getNeurons()])
        self.valueMatrix = valueMatrix.reshape(len(previousLayer.getNeurons()), 1)

        self.biasMatrix = np.array([neuron.getBias() for neuron in self.getNeurons()])

        self.weightMatrix = np.array([neuron.getAllWeights() for neuron in previousLayer.getNeurons()])

        result = np.dot(self.weightMatrix.T, valueMatrix)
        result = result.reshape(-1)
        result = sigmoid(result + self.biasMatrix)

        neurons = self.getNeurons()
        for i in range(len(neurons)):
            neurons[i].setValue(result[i])


        
def sigmoid(x):
    y = 1/(1 + np.exp(-x))
    return y
                




