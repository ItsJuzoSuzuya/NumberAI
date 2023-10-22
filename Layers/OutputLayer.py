import ValueCalculator
from Neuron import Neuron
from .Layer import Layer

class OutputLayer(Layer):
    def __init__(self, linkedLayer: 'Layer') -> None:
        super().__init__()
        self.clearNeurons()
        for _ in range(10):
            neuron = Neuron()
            self.addNeuron(neuron)
            
        linkedLayer.connectLayer(self)

        # neuron.calculateValue(linkedLayer.getNeurons())

        ValueCalculator.calculateValueOfNeuron(self, linkedLayer)
        
    