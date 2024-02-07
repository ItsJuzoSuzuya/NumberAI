import numpy as np
from Layers.Layer import Layer

def calculateValueOfNeuron(outputLayer: 'Layer', previousLayer: 'Layer'):
    
    valueMatrix = np.array([neuron.getValue() for neuron in previousLayer.getNeurons()])
    valueMatrix = valueMatrix.reshape(len(previousLayer.getNeurons()), 1)
    biasMatrix = np.array([neuron.getBias() for neuron in outputLayer.getNeurons()])

    weightMatrix = np.array([neuron.getAllWeights() for neuron in previousLayer.getNeurons()])

    result = np.dot(weightMatrix.T, valueMatrix)
    result = result.reshape(-1)
    result = sigmoid(result + biasMatrix)

    neurons = outputLayer.getNeurons()
    for i in range(len(neurons)):
        neurons[i].setValue(result[i])


        
def sigmoid(x):
    y = 1/(1 + np.exp(-x))
    return y



