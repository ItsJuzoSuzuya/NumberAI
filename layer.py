class Layer():
    neurons = []

    def __init__(self) -> None:
        pass

    def addNeuron(self, neuron):
        self.neurons.append(neuron)

    def getNeurons(self):
        return self.neurons
    
    def clearNeurons(self):
        self.neurons.clear()