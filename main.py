from keras.datasets import mnist
from NeuralNetwork import NeuralNetwork

(trainImages, trainNumbers), (testImages, testNumbers) = mnist.load_data()

# for i in range(9):  
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#     pyplot.show()

# for image in train_X:

neuralNetwork = NeuralNetwork(trainImages[0], trainNumbers[0])   
#neuralNetwork.train(trainNumbers[1])








     
