from layer import Layer
from neuron import Neuron

class InputLayer(Layer):
    def __init__(self, image) -> None:
        self.clearNeurons()
        for row in image:
            for pixel in row:
                neuron = Neuron(pixel)
                self.addNeuron(neuron)
        super().__init__()
    