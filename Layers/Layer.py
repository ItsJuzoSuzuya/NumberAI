import random
from Neuron import Neuron
from typing import List

class Layer():
    def __init__(self, size) -> None:
        self._neurons: List[Neuron] = []

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
                




