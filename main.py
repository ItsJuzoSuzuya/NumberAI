from window import Window
from keras.datasets import mnist
from matplotlib import pyplot
from neuron import Neuron
from inputLayer import InputLayer

(train_X, train_y), (test_X, test_y) = mnist.load_data()


# for i in range(9):  
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#     pyplot.show()



for image in train_X:
    inputLayer = InputLayer(image)
    print(inputLayer.getNeurons())
        