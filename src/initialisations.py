"""
Initialisation functions for training neural networks.

Contains initialisation functions for weights and biases used when creating a network to be trained. 
Implemented functions include: He, Uniform_Xavier, Normal_Xavier
"""


import numpy as np

def He(layer_sizes, RNG):
    """
    He initialisation of weights and biases. 
    
    Parameters
    ----------
    layer_sizes : list[int]
        The number of neurons in each layer.
    RNG : int
        Seeded random number generator.
    Returns
    -------
    weights : np.ndarray
        The matrix of numerical values which scale input values to determine feature significance.
    biases : np.ndarray
        The vector of numerical values which scale input values to determine feature significance.
    """
    weights = []
    biases = []
    
    for layer in range(len(layer_sizes) - 1):
        Wi = RNG.standard_normal(
            (layer_sizes[layer + 1], layer_sizes[layer])
        ) * np.sqrt(2 / layer_sizes[layer])
        weights.append(Wi)
            
        bi = (np.zeros(layer_sizes[layer + 1]))
        biases.append(bi)
        
    return weights, biases

def Uniform_Xavier(layer_sizes, RNG):
    """
    Uniform Xavier initialisation of weights and biases. 
    
    Parameters
    ----------
    layer_sizes : list[int]
        The number of neurons in each layer.
    RNG : int
        Seeded random number generator.
    Returns
    -------
    weights : np.ndarray
        The matrix of numerical values which scale input values to determine feature significance.
    biases : np.ndarray
        The vector of numerical values which scale input values to determine feature significance.
    """
    
    weights = []
    biases = []
    
    x = np.sqrt(6 / (layer_sizes[0] + layer_sizes[-1]))
    
    for layer in range(len(layer_sizes) - 1):
        Wi = RNG.uniform(
            low=-x, high=x, 
            size = (layer_sizes[layer + 1], layer_sizes[layer])
        )
        weights.append(Wi)
            
        bi = (np.zeros(layer_sizes[layer + 1]))
        biases.append(bi)
        
    return weights, biases

def Normal_Xavier(layer_sizes, RNG):
    """
    Normal Xavier initialisation of weights and biases. 
    
    Parameters
    ----------
    layer_sizes : list[int]
        The number of neurons in each layer.
    RNG : int
        Seeded random number generator.
    Returns
    -------
    weights : np.ndarray
        The matrix of numerical values which scale input values to determine feature significance.
    biases : np.ndarray
        The vector of numerical values which scale input values to determine feature significance.
    """
    
    weights = []
    biases = []
    
    sigma = np.sqrt(2 / (layer_sizes[0] + layer_sizes[-1]))
    
    for layer in range(len(layer_sizes) - 1):
        Wi = RNG.normal(
            scale=sigma,
            size = (layer_sizes[layer + 1], layer_sizes[layer])
        )
        weights.append(Wi)
            
        bi = (np.zeros(layer_sizes[layer + 1]))
        biases.append(bi)
        
    return weights, biases