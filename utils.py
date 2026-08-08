"""
Utility functions for evaluating and visualising the neural network.

Includes binary cross-entropy loss and plotting functions for monitoring
training and visualising two-dimensional classification results.
"""

import numpy as np
import matplotlib.pyplot as plt

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
    
def plot_log_loss(data):
    """
    Plots training loss data on a logarythmic y-axis.

    Parameters
    ----------
    data : list[float]
        Loss values from training indexed by epoch.
    """
    plt.figure(figsize=(8,5))
    plt.semilogy(data, label=f"Loss at epoch {len(data)}: {data[-1]:.3g}")
    
    plt.title(f"Reduction in Loss over {len(data)} epochs")
    plt.ylabel("logged loss of the neural network")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()
    
def plot_decision_boundary(X, y, model):
    """
    Plots a model's decision boundary for 2-dimensional input data.

    Parameters
    ----------

    X : np.ndarray
        Input data for exactly two features.
    y : np.ndarray
        True binary labels.
    Note
    ----
    X must be given with inputs separated by column.
    Y must be given as a row vector.
    """
    x_min, x_max = X[:,0].min() * 1.1, X[:,0].max() * 1.1
    y_min, y_max = X[:,1].min() * 1.1, X[:,1].max() * 1.1

    xx, yy = np.meshgrid(
        np.linspace(x_min,x_max,200),
        np.linspace(y_min,y_max,200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model(grid)
    probs = probs.reshape(xx.shape)

    plt.contour(
        xx,
        yy,
        probs,
        levels=[0.5]
    )

    plt.scatter(
        X[:,0],
        X[:,1],
        c=y.flatten()
    )

    plt.title("Decision Boundary")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")

    plt.show()