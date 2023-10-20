import time
from keras.datasets import mnist
from Layers.InputLayer import InputLayer
from Layers.OutputLayer import OutputLayer
from Layers.HiddenLayer import HiddenLayer

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# for i in range(9):  
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#     pyplot.show()

# for image in train_X:


start = time.perf_counter()

inputLayer = InputLayer(train_X[0])
hiddenLayer1 = HiddenLayer(inputLayer)
hiddenLayer2 = HiddenLayer(hiddenLayer1)
outputLayer = OutputLayer(hiddenLayer2)

time = time.perf_counter() - start







    
