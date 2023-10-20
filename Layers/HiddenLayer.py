import random
from .Layer import Layer
from Neuron import Neuron 

class HiddenLayer(Layer):
    def __init__(self, linkedLayer: 'Layer') -> None:
        super().__init__(16)
        for _ in range(16):
            randomnBias = random.uniform(-2,2)
            neuron = Neuron(None, randomnBias)
            self.addNeuron(neuron)
            linkedLayer.connectLayer(self)
            neuron.calculateValue(linkedLayer.getNeurons())
            
        
    