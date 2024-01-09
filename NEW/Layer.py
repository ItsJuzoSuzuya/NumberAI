import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from keras.datasets import mnist


#(train_X, trainNumbers), (test_X, testNumbers) = mnist.load_data()
#train_X = train_X / 1000

#X = np.reshape(train_X, [60000, 784])

nnfs.init()

np.random.seed(0)

class Layer:
    """
    Neuronal Layer including the incoming weights, biases of each neuron in the layer
    and the outputs or values of the neurons themselfs
    """
    def __init__(self, n_inputs, n_neurons):
        #forward data
        self.outputs = []
        self.inputs = []
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        #backward data
        self.dweights = []
        self.dbiases = []
        self.dinputs = []

    def forward(self, inputs):
        """
        Calculates the outputs multiplying each incoming weight with their connected 
        neurons output aka value
        """
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases 

    def backward(self, dvalues):
        """
        Backward pass
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.dweights.T)

class ActivationReLU:
    """
    Activation fuction for hidden layers
    """
    def __init__(self):
        #forward data
        self.outputs = []
        self.inputs = []

        #backward data
        self.dinputs = []
    def forward(self, inputs):
        """
        Activates only on positive values
        """
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        """
        Derivative where negative Numbers are Zero and the rest 1
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax:
    """
    Activation function for the output layer
    """
    def __init__(self):
        #forward data
        self.outputs= []
        self.inputs = []

        #backward data
        self.dinputs = []
        
    def forward(self, inputs):
        """
        Calculates the probability of the
        """
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        propabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = propabilities

    def backward(self, dvalues):
        """
        Backkward pass 
        """
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    """
    Base class representing the loss or cost of accuracy for a neural network.
    """
    def calculate(self, output, y):
        """
        Calculates the overall inaccuracy based on the given output and expected output.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss


class LossCategoricalCrossentropy(Loss):
    """
    Categorical Crossentropy loss function for classification problems with multiple classes.
    """
    def __init__(self):
        self.dinputs = []

    def forward(self, y_pred, y_true):
        """
        Calculates the negative log likelihood loss based on 
        predicted and true class probabilities.
        """
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        """
        Derivative of the loss function
        """
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        epsilon = 1e-15
        self.dinputs = -y_true / (dvalues + epsilon)
        self.dinputs = self.dinputs / samples


class ActivationSoftmaxLossCategoricalCrossentropy():
    """
    Combines the Softmax and Loss calculation
    """
    def __init__(self):
        #forward data
        self.outputs = []
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()

        #backward data
        self.dinputs = []

    def forward(self, inputs, y_true):
        """
        calls the individual forward of softmax and loss
        """
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs

        return self.loss.calculate(self.outputs, y_true)

    def backward(self, dvalues, y_true):
        """
        Backward pass
        """
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class OptimizerSGD:
    """
    Class for the adjustments of weights and biases
    """
    def __init__(self, learning_rate=0.4, decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay 
        self.iterations = 0

    def pre_update_params(self):
        """
        Before updating the params adjust learning rate 
        """
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
                (1 / (1 + self.decay * self.iterations))


    def update_params(self, layer):
        """
        Adjustment of given layers weights and biases
        """
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases

    def post_update_params(self):
        """
        Updates current iteration
        """
        self.iterations += 1


X, y = spiral_data(samples=1000, classes= 3)

hiddenLayer = Layer(2, 64)
activation1 = ActivationReLU()

outputLayer = Layer(64, 3)
lossActivation = ActivationSoftmaxLossCategoricalCrossentropy()

optimizer = OptimizerSGD(decay=1e-2)

for epoch in range(10001):
    hiddenLayer.forward(X)
    activation1.forward(hiddenLayer.outputs)

    outputLayer.forward(activation1.outputs)

    loss = lossActivation.forward(outputLayer.outputs, y)

    predictions = np.argmax(lossActivation.outputs, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f}')

    lossActivation.backward(lossActivation.outputs, y)
    outputLayer.backward(lossActivation.dinputs)
    activation1.backward(outputLayer.dinputs)
    hiddenLayer.backward(activation1.dinputs)
    
    optimizer.pre_update_params()
    optimizer.update_params(hiddenLayer)
    optimizer.update_params(outputLayer)
    optimizer.post_update_params()