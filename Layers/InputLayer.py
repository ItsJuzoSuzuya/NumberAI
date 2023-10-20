from .Layer import Layer
from Neuron import Neuron

class InputLayer(Layer):
    def __init__(self, image) -> None:
        super().__init__(784)
        self.clearNeurons()
        for row in image:
            for pixel in row:
                neuron = Neuron(pixel/1000)
                self.addNeuron(neuron)
        
    