import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from keras.datasets import mnist


class Layer:
    """
    Neuronal Layer including the incoming weights, biases of each neuron in the layer
    and the outputs or values of the neurons themselfs
    """
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        Calculates the outputs multiplying each incoming weight with their connected 
        neurons output aka value
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        """
        Backward pass
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ActivationReLU:
    """
    Activation fuction for hidden layers
    """
    def forward(self, inputs):
        """
        Activates only on positive values
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
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
        
    def forward(self, inputs):
        """
        Calculates the probability of the
        """
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        propabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = propabilities

    def backward(self, dvalues):
        """
        Backkward pass 
        """
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
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

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class ActivationSoftmaxLossCategoricalCrossentropy():
    """
    Combines the Softmax and Loss calculation
    """
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()

    def forward(self, inputs, y_true):
        """
        calls the individual forward of softmax and loss
        """
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

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

class OptimizerAdam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)
        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)

    def post_update_params(self):
        self.iterations += 1



(train_X, trainNumbers), (test_X, testNumbers) = mnist.load_data()
train_X = train_X / 1000

X = np.reshape(train_X, [60000, 784])

hiddenLayer1 = Layer(784, 16)
activation1 = ActivationReLU()

hiddenLayer2 = Layer(16,16)
activation2 = ActivationReLU()

outputLayer = Layer(16, 10)
lossActivation = ActivationSoftmaxLossCategoricalCrossentropy()

optimizer = OptimizerAdam(learning_rate=0.05, decay=5e-7)

for epoch in range(10001):
    hiddenLayer1.forward(X)
    activation1.forward(hiddenLayer1.output)

    hiddenLayer2.forward(activation1.output)
    activation2.forward(hiddenLayer2.output)

    outputLayer.forward(activation2.output)

    loss = lossActivation.forward(outputLayer.output, trainNumbers)

    predictions = np.argmax(lossActivation.output, axis=1)
    if len(trainNumbers.shape) == 2:
        trainNumbers = np.argmax(trainNumbers, axis=1)
    accuracy = np.mean(predictions == trainNumbers)

    
    print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f}')

    lossActivation.backward(lossActivation.output, trainNumbers)
    outputLayer.backward(lossActivation.dinputs)
    activation2.backward(outputLayer.dinputs)
    hiddenLayer2.backward(activation2.dinputs)
    activation1.backward(hiddenLayer2.dinputs)
    hiddenLayer1.backward(activation1.dinputs)
    
    optimizer.pre_update_params()
    optimizer.update_params(hiddenLayer1)
    optimizer.update_params(hiddenLayer2)
    optimizer.update_params(outputLayer)
    optimizer.post_update_params()