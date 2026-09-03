"""
Sample data used for a proof-of-concept demonstration of the neural network.
The dataset contains 2 features and 32 samples and is classified into two classes by binary classification.
Features have a non-linear relationship. 
"""

import numpy as np

X = np.array([
    [-2.0, -1.4],
    [-1.7, -0.8],
    [-1.5, -1.8],
    [-1.2, -0.3],
    [-1.0,  1.5],
    [-0.8,  0.7],
    [-0.6,  1.8],
    [-0.4, -1.6],
    [-0.2,  0.2],
    [ 0.0,  1.4],
    [ 0.2, -0.4],
    [ 0.4, -1.8],
    [ 0.6,  1.6],
    [ 0.8,  0.4],
    [ 1.0, -1.3],
    [ 1.2, -0.2],
    [ 1.4,  1.7],
    [ 1.6,  0.8],
    [ 1.8, -0.9],
    [ 2.0,  0.1],
    [-1.8,  0.4],
    [-1.4,  1.1],
    [-1.1, -1.5],
    [-0.9, -0.7],
    [-0.5,  1.2],
    [-0.1, -1.1],
    [ 0.3,  0.9],
    [ 0.7, -0.8],
    [ 1.1,  1.2],
    [ 1.5, -1.5],
    [ 1.9,  1.4],
    [-1.6, -1.2]
])

y = np.array([
    0, 0, 0, 0, 0, 1, 0, 0,
    1, 0, 1, 0, 0, 1, 0, 1,
    0, 1, 0, 1, 1, 0, 0, 1,
    0, 1, 0, 1, 0, 0, 0, 1
])