"""
Activation functions and their derivatives used by the neural network.

Each activation function maps a layer's pre-activation values to its
output values. The derivative activation functions are used during backpropagation.
The keywork 'prime' is used to denote a function's derivative.

ACTIVATIONS and ACTIVATIONS_PRIME provide string-based lookup dictionaries
for selecting activation functions within the NeuralNetwork class.
"""

import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_prime(X):
    return sigmoid(X) * (1 - sigmoid(X))

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def leakyRelu(z):
    return np.maximum(0.01*z, z)

def leakyRelu_prime(z):
    return np.where(z>=0, 1, 0.01)

def gelu(z):
    return 0.5 * z * (
            1 + np.tanh(
                np.sqrt(2 / np.pi) * (z + 0.044715 * z**3) # using erf approximation
            )
        )
    
def gelu_prime(z):
    a = np.sqrt(2.0 / np.pi)
    T = np.tanh(a * (z + 0.044715 * z**3))
    return (
        0.5 * (1 + T) 
        + 0.5 * z * (1 - T**2) * a * (1 + 3 * 0.044715 * z**2)
    )

# lookup dictionaries used by NeuralNetwork
ACTIVATIONS = {
            "relu": relu,
            "leakyrelu": leakyRelu,
            "gelu": gelu,
            "sigmoid": sigmoid
        }

ACTIVATIONS_PRIME = {
            "relu": relu_prime,
            "leakyrelu": leakyRelu_prime,
            "gelu": gelu_prime,
            "sigmoid": sigmoid_prime
        }