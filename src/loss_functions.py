""" 
Loss functions for training neural networks. 

This module provides loss functions used to measure the difference between 
the predictions produced by a neural network and the corresponding target values.
Implemented functions include: binary cross-entropy, categorical cross-entropy. 

"""

import numpy as np

def binary_cross_entropy(y, A): 
    """
    Computes binary cross-entropy loss.

    Predictions are clipped before taking logarithms to avoid numerical
    extremes when predictions are exactly 0 or 1.

    Parameters
    ----------
    y : np.ndarray
        True binary labels.
    A : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        Mean binary cross-entropy loss.
    """
    
    eps = 1e-15
    A = np.clip(A, eps, 1-eps)
    
    return - np.mean(
        y * np.log(A) + 
        (1 - y) * np.log(1 - A)
    )

def categorical_cross_entropy(y, A):
    """ 
    Computes categorical cross-entropy loss for multi-class classification. 
    Predictions are clipped before taking logarithms to avoid numerical 
    extremes when predicted probabilities are exactly 0 or 1. 
    
    Parameters
    ---------- 
    y : np.ndarray 
        One-hot encoded class labels. 
    A : np.ndarray 
        Predicted class probabilities, with each row corresponding
        to a sample and each column corresponding to a class. 
        
    Returns
    ------- 
    float 
        Mean categorical cross-entropy loss across all samples. 
    """
    eps = 1e-12
    A = np.clip(A, eps, 1.0 - eps)
    return -np.mean(np.sum(y * np.log(A), axis=1))