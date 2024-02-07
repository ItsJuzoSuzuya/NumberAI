import numpy as np
from .Layer import Layer
from neuron import Neuron

class InputLayer(Layer):
    def __init__(self, image) -> None:
        super().__init__()
        self.clearNeurons()
        for row in image:
            for pixel in row:
                neuron = Neuron(pixel/1000)
                self.addNeuron(neuron)
        
    