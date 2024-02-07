import numpy as np
from Layer import Layer
from ActivationReLU import ActivationReLU
from ActivationSoftmaxLossCategoricalCrossentropy import ActivationSoftmaxLossCategoricalCrossentropy
from OptimizerAdam import OptimizerAdam
from keras.datasets import mnist

(train_X, trainNumbers), (test_X, testNumbers) = mnist.load_data()
train_X = train_X / 1000
test_X = test_X / 1000

X = np.reshape(train_X, [60000, 784])
X_test = np.reshape(test_X, [10000, 784])



hiddenLayer1 = Layer(784, 16)
activation1 = ActivationReLU()

hiddenLayer2 = Layer(16,16)
activation2 = ActivationReLU()

outputLayer = Layer(16, 10)
lossActivation = ActivationSoftmaxLossCategoricalCrossentropy()

optimizer = OptimizerAdam(learning_rate=0.05, decay=5e-7)

for epoch in range(50):
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

hiddenLayer1.forward(X_test)
activation1.forward(hiddenLayer1.output)

hiddenLayer2.forward(activation1.output)
activation2.forward(hiddenLayer2.output)

outputLayer.forward(activation2.output)

loss = lossActivation.forward(outputLayer.output, testNumbers)

predictions = np.argmax(outputLayer.output, axis=1)
accuracy = np.mean(predictions == testNumbers)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
