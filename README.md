# Convolutional-Neural-Network
Implementation of convolution neural network on the MNIST data set for digit recognition using Python
The CNN was implemented in Python, based on the provided C# code. The implementation was done on the MNIST dataset. The MNIST training dataset of 1000 images provided on the web site, then the implementation as tested on 10,000 images for testing. For CNN architecture of two CNN layers the first one has six feature maps, the second one has 12 feature maps, the kernel size is 5*5. Then the output of the last CNN layer is fed in to two layers of normal neural network, they have 50 and 10 neurons respectively. RELU activation function was used for the CNN and the first layer of the normal neural network and soft max activation function for the last neural layer. Average pooling was used for the CNN. The accuracy of the implemented structure is 93.56 %.
The MNIST data set is needed to run the code after editing the data path in the code. 
