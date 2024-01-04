import ValueCalculator
import random
import numpy as np
from neuron import Neuron
from .Layer import Layer

class OutputLayer(Layer):
    def __init__(self, linkedLayer: 'Layer') -> None:
        super().__init__()
        for _ in range(10):
            randomnBias = random.uniform(-2,2)
            neuron = Neuron()
            self.addNeuron(neuron)
            self.addToBiases(randomnBias)

        linkedLayer.connectLayer(self)
        self.calculateValue(linkedLayer)
        
        # neuron.calculateValue(linkedLayer.getNeurons())

        # ValueCalculator.calculateValueOfNeuron(self, linkedLayer)
        
    