# digit_recogniser
Recognizing digits using CNNs
This project consists of a 4 layers Sequential Convolutional Neural Network for digits recognition trained on MNIST dataset. I have build it with keras API (Tensorflow backend) which is very intuitive. I have achieved 99.10% accuracy in less than 5 mins, to run the whole code.

## Libraries used:
1. Pandas
2. Numpy
3. Matplotlib
4. Sklearn

## Defining the Model
I used the Keras Sequential API, where you have to just add one layer at a time, starting from the input.

The first is the convolutional (Conv2D) layer. It is like a set of learnable filters. I chose to set 32 filters of size 5x5 for the first conv2D layer and 64 filters of same size for the next conv2D layer, followed by a 32 filters of size 5x5. Each filter transforms a part of the image using the kernel filter. Filters can be seen as a transformation of the image.

The CNN can isolate features that are useful everywhere from these transformed images (feature maps).

The second important layer in CNNs is the pooling (MaxPool2D) layer. This layer simply acts as a down sampling filter. I chose a MaxPool layer, which simply looks at the pixels in the matrix size defined (I fixed it to 2x2) and picks the maximum value. These are used to reduce computational cost, and to some extent also reduce overfitting.

Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.

Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their weights to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting.

'relu' is the rectifier (which equals max(0,x)). The rectifier activation function is used to add non linearity to the network.

The Flatten layer is use to convert the final feature maps into a single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.

In the end I passed these features into two fully-connected (Dense) layers which is just artificial neural networks (ANN) classifier. In the last layer(Dense(10,activation="softmax")) the net outputs distribution of probability of each class.

Thank you!
