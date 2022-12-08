# For us to create a fully functional neural network we need to follow some
# steps. It is essensial to have a class which represents each layer of 
# the neural network. We also need to implement forward propagation to 
# propagate the initial input to the next layers. We need to implement
# backward propagation to better optimize the weights and biases. We also
# need an activation function because our NN need to be able to handle
# non-linear problems.

import numpy as np

# The first step in our code is to create a class which we will call "Layer"
# Other classes that we will create will inherit from the class "Layer"
# Class Layer handles the input, output
# Class Layer handles forward and backward propogation

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward_propagation(self, input_data):
        pass
    def backward_propagation(self, output_gradient, learning_rate):
        pass
      
# The second step is to create a class which we will call "Dense_Layer"
# This class will inherit the class "layer"

class Dense_Layer(Layer):
    def __init__(self, input_size, output_size):
        
        # input_size is the number of input neurons
        # ouput_size is the number of ouput 
        # We set random weights according to the number of neurons
        self.weights = np.random.rand(output_size, input_size)
        
        # We set a bias parameter for each output neuron
        self.bias = np.random.rand(output_size, 1)
        
    def forward_propagation(self, input_data):
        self.input = input_data
        
        # We calculate the output
        # We use the formula output = sum of input * weight + bias
        # The function "dot" gives us the dot-product of two matrices
        # dot-product is the multiplication of two compatible matrices
        return np.dot(self.weights, self.input) + self.bias
        
    # output_gradient is the derivative of the Error with respect to
    # the biases. dE/dB = dE/dY
    def backward_propagation(self, output_gradient, learning_rate):
        
        # Derivative of the Error with respect to the weights
        # dE/dW = (dE/dY) * X(Transposed)
        weights_gradient = np.dot(output_gradient, self.input.T)
        
        # We update the weights and biases with gradient decent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        
        # We return the derivative of the Error with respect to the input.
        # dE/dX = W(transposed) * (dE/dY)
        return np.dot(self.weights.T, output_gradient)
      
# The third step is to create a class which we will call "Activation_Layer"
# We will use this class to put our input into an activation function

class Activation_Layer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward_propagation(self, input_data):
        self.input = input_data
        
        # We need an activation function for solving non-linear problems
        # We put the input values inside an activation function
        return self.activation(self.input)
        
    def backward_propagation(self, output_gradient, learning_rate):
        
        # We calculate the Hadamard Product of dE/dY and 
        # the activation function of the input.
        # The Hadamard Product is a matrix multiplication where every
        # element of the first matrix is multiplied by the second matrix's
        # corresponding element.
        return np.multiply(output_gradient, self.activation_prime(self.input))
      
# The forth step is to write the activation function that will be used.
# We create a class for the activation function which will inherit
# the Activation_Layer class.
# There are lots of activation functions such as ReLU, sigmoid etc.
# For this example, we use the Hyperbolic activation function.

# Hyperbolic tangent Activation function
class Tanh(Activation_Layer):
    def __init__(self):
        
        # Hyperbolic tangent function
        tanh = lambda x: np.tanh(x)
        
        # Hyperbolic tangent function derivative
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        
        super().__init__(tanh, tanh_prime)
        
# The fifth step is Loss function
# y_pred = the predicted outputs
# y_true = the desired outputs

def mean_squared_error(y_true, y_pred):
    
    # Error = (sum of (y_true - y_pred)^2) / n 
    return np.mean(np.power(y_true - y_pred, 2))

def mean_squared_error_prime(y_true, y_pred):
    
    # dError/dY = (2 * (sum of (y_pred - y_true))) / n
    return 2 * (y_pred - y_true) / np.size(y_true)
  
# Implementing XOR problem to test the network
# I chose the XOR problem because it's non-linear

# Input data and their shape
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))

# Desired output and their shape
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Creating the network
network = [Dense_Layer(2, 3), Tanh(), Dense_Layer(3, 1), Tanh()]

# You can change the epochs to your liking
epochs = 10000

# Learning rate
learning_rate = 0.1

# Training the network
for e in range(epochs):
    
    # We set the default error to 0
    error = 0
    for x, y in zip(X, Y):
        output = x
        
        # Forward propagation
        for layer in network:
            output = layer.forward_propagation(output)
            
        error += mean_squared_error(y, output)
        
        grad = mean_squared_error_prime(y, output)
        
        # Backward propagation
        for layer in reversed(network):
            grad = layer.backward_propagation(grad, learning_rate)
            
    error /= len(x)
    
    print('%d/%d, error=%f' % (e + 1, epochs, error))
