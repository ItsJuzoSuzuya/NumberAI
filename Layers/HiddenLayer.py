import random
import time
from .Layer import Layer
from Neuron import Neuron 
import ValueCalculator

class HiddenLayer(Layer):
    def __init__(self, linkedLayer: 'Layer') -> None:
        super().__init__(16)
        for _ in range(16):
            randomnBias = random.uniform(-2,2)
            neuron = Neuron(None, randomnBias)
            self.addNeuron(neuron)
            
        linkedLayer.connectLayer(self)

        # for neuron in self.getNeurons():
        #     neuron.calculateValue(linkedLayer.getNeurons())
        
        ValueCalculator.calculateValueOfNeuron(self, linkedLayer)


            
            
        
    