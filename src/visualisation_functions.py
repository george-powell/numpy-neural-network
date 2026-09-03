"""
Visualisation functions for evaluating neural network models.

Includes plotting functions for loss over training and decision boundary evolution.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_log_loss(data):
    """
    Plots training loss data on a logarithmic y-axis.

    Parameters
    ----------
    data : list[float]
        Loss values from training indexed by epoch.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(
        range(1, len(data) + 1),
        data,
        label=f"Loss at epoch {len(data)}: {data[-1]:.3g}"
    )

    ax.set_title(f"Reduction in Loss over {len(data)} epochs")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_xlabel("Epoch")
    ax.legend()

    fig.tight_layout()

    return fig
    
def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    """
    Creates a decision-boundary plot for 2-dimensional input data.

    Parameters
    ----------
    X : np.ndarray
        Input data with exactly two features.
    y : np.ndarray
        True binary labels.
    model : callable
        Model function used to generate predictions.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """

    x_min, x_max = X[:, 0].min() * 1.1, X[:, 0].max() * 1.1
    y_min, y_max = X[:, 1].min() * 1.1, X[:, 1].max() * 1.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = model(grid)
    probs = probs.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.contour(
        xx,
        yy,
        probs,
        levels=[0.5]
    )

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y.flatten()
    )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    fig.tight_layout()

    return fig