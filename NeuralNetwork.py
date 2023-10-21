from Layers.InputLayer import InputLayer
from Layers.OutputLayer import OutputLayer
from Layers.HiddenLayer import HiddenLayer

class NeuralNetwork():
    def __init__(self, firstInput) -> None:
        self.inputLayer = InputLayer(firstInput)
        self.hiddenLayer1 = HiddenLayer(self.inputLayer)
        self.hiddenLayer2 = HiddenLayer(self.hiddenLayer1)
        self.outputLayer = OutputLayer(self.hiddenLayer2)
        