import random
import numpy as np
from .Layer import Layer
from neuron import Neuron 
# import ValueCalculator

class HiddenLayer(Layer):
    def __init__(self, linkedLayer: 'Layer') -> None:
        super().__init__()
        for _ in range(16):
            randomnBias = random.uniform(-2,2)
            neuron = Neuron(None, randomnBias)
            self.addNeuron(neuron)
            self.addToBiases(randomnBias)

        linkedLayer.connectLayer(self)
        self.calculateValue(linkedLayer)

        # for neuron in self.getNeurons():
        #     neuron.calculateValue(linkedLayer.getNeurons())
        
        # ValueCalculator.calculateValueOfNeuron(self, linkedLayer)


            
            
        
    