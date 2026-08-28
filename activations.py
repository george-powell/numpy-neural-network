"""
Activation functions and their derivatives used by the neural network.

Each activation function maps a layer's pre-activation values to its
output values. The derivative activation functions are used during backpropagation.
The keywork 'prime' is used to denote a function's derivative.

ACTIVATIONS and ACTIVATIONS_PRIME provide string-based lookup dictionaries
for selecting activation functions within the NeuralNetwork class.
"""

import numpy as np
from scipy.special import erf

# CONSTANTS
SQRT_2 = np.sqrt(2.0)
INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)

def sigmoid(X):
    return 1 / (1.0 + np.exp(-X))

def sigmoid_prime(X):
    return sigmoid(X) * (1.0 - sigmoid(X))

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def leakyRelu(z):
    return np.maximum(0.01*z, z)

def leakyRelu_prime(z):
    return np.where(z>=0, 1.0, 0.01)

def gelu(z):
    return 0.5 * z * (
            1.0 + erf(z / SQRT_2)
        )
    
def gelu_prime(z):
    cdf = 0.5*(1.0 + erf(z / SQRT_2))
    pdf = INV_SQRT_2PI * np.exp(-0.5*z**2.0)
    return cdf * z * pdf

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax_prime(z): 
    s = softmax(z)
    return np.diag(s) - np.outer(s, s)

# lookup dictionaries used by NeuralNetwork
ACTIVATIONS = {
            "relu": relu,
            "leakyrelu": leakyRelu,
            "gelu": gelu,
            "sigmoid": sigmoid,
            "softmax": softmax
        }

ACTIVATIONS_PRIME = {
            "relu": relu_prime,
            "leakyrelu": leakyRelu_prime,
            "gelu": gelu_prime,
            "sigmoid": sigmoid_prime,
            "softmax": softmax_prime
        }