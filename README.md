# NumPy Neural Network

A feedforward neural network built using only **Python and NumPy**, with no machine-learning frameworks. The project was developed to understand the mathematics and implementation behind neural networks rather than relying on high-level libraries.

## Features

* Arbitrary number of layers and neurons
* Configurable activation functions

  * ReLU
  * Leaky ReLU
  * GELU
  * Sigmoid
* Forward propagation
* Backpropagation
* Mini-batch training
* Input normalisation
* Multiple optimisation algorithms:

  * Stochastic Gradient Descent (SGD)
  * Momentum
  * Adam
* Binary cross-entropy loss
* Decision-boundary and loss visualisation

## Project Structure

```text
Neural_network/
├── neural_network.py       # Neural network architecture and training
├── optimisers.py           # SGD, Momentum and Adam
├── activations.py          # Activation functions and derivatives
├── utils.py                # Loss functions and visualisation
├── sample_data.py          # Example datasets
├── main.ipynb              # Main demonstration
├── notebooks/              # Earlier implementations and learning steps
│   ├── _1_linear_regression.ipynb
│   ├── _2_g_des_lin_reg.ipynb
│   ├── _3_logistic_regression.ipynb
│   └── _4_backprob_neuralnetwork.ipynb
├── requirements.txt
└── README.md
```
## Learning Progression

The `notebooks/` directory contains the progression used to build the final implementation:

1. Linear regression
2. Gradient descent
3. Logistic regression
4. Neural network with backpropagation

The final notebook was extended into a reusable neural-network class with mini-batching and multiple optimisers.

## Purpose

This project is part of a larger **from-scratch machine learning project** aimed at understanding the mathematical foundations of modern machine learning systems.

The implementation focuses on the underlying concepts of:

* Linear algebra
* Gradient-based optimisation
* Probability and loss functions
* Computational graphs
* Backpropagation
* Numerical optimisation

The next stage of the project is to build **self-attention and transformer components from scratch using NumPy**.

## Requirements

* Python 3
* NumPy
* Matplotlib

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Limitations

This implementation is primarily educational rather than production-oriented. It currently focuses on binary classification and CPU-based NumPy computation, without GPU acceleration or the optimisation and numerical safeguards found in established machine-learning frameworks.
