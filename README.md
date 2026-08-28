# NumPy Neural Network

A small neural network framework built using only **Python and NumPy**, with no machine-learning libraries used to implement the models. The project was developed to understand the mathematics and implementation behind neural networks. 

The [`main.ipynb`](https://github.com/george-powell/numpy-neural-network/blob/main/main.ipynb) notebook demonstrates the framework on two classification problems. A simple 2-feature dataset is first used to visualise and analyse the learned decision boundary. The Iris dataset is then used to conduct a controlled comparison of **SGD, Momentum and Adam** optimisers, training and being tested while keeping network architecture and training procedure fixed. 

The repository also contains a sequence of notebooks documenting the learning progression from linear regression and gradient descent to a 2-layer Multilayer Perceptron.

## Features

* Arbitrary number of layers and neurons
* Configurable activation functions

  * ReLU
  * Leaky ReLU
  * GELU
  * Sigmoid
  * Softmax
* Forward propagation
* Backpropagation
* Mini-batch training
* Input normalisation
* Multiple optimisation algorithms:

  * Stochastic Gradient Descent (SGD)
  * Momentum
  * Adam
* Binary and categorical cross-entropy loss
* Reproducible random initialisation and mini-batch shuffling through a configurable random seed

## Demonstrations

The [`main.ipynb`](https://github.com/george-powell/numpy-neural-network/blob/main/main.ipynb) notebook provides practical demonstrations of the network on two classification problems, and how network parameters can be altered and compared in isolation.

### 2D-classification &mdash; Decision Boundary
A small 2-feature binary classification dataset used as a proof of concept. The network is trained on the data with the log-loss and decision boundary visualised to show the successful implementation. 

The experiment demonstrates:

* Forward and backward propagation working to learn a classification boundary
* The effect of increasing the number of neurons on the decision boundary
* The difference between the piecewise-linear boundaries produced by **ReLU** and the smooth non-linear boundaries produced by **GELU**

### Iris-Dataset &mdash; Optimiser Comparison
The **Iris Dataset** from `scikt-learn` is used to compare **SGD, Momentum and Adam**.

The experiment uses three otherwise identical networks, varying only the optimisation algorithm. The dataset is standardised, shuffled and split into training and test sets. **4-fold stratified cross-validation** is then used on the training set to compare the accuracy and spread of each network. Finally, the most accurate optimiser in the experiment at 0.90, Adam, is trained on the entire training set and then tested on the unseen test set, yielding an accuracy of 1.00.

The experiment measures:
* Mean cross-validation accuracy
* Accuracy standard deviation across folds
* Final test accuracy

This demonstration of how optimiser choice can affect neural network training and performance illustrates how parameters can be compared in a controlled manner in order to determine the appropriate architecture of a network specific to a dataset. 

The experiment also demonstrates the network's ability to perform **multi-class classification** with an **arbitrary number of output neurons**, using a three-neuron softmax output layer and categorical cross-entropy. 

## Project Structure

```text
Neural_network/
├── neural_network.py       # Neural network architecture and training
├── optimisers.py           # SGD, Momentum and Adam
├── activations.py          # Activation functions and derivatives
├── utils.py                # Loss functions and visualisation
├── sample_data.py          # Example dataset
├── main.ipynb              # Main demonstration and experiments
├── notebooks/              # Earlier implementations and learning steps
│   ├── _1_linear_regression.ipynb
│   ├── _2_g_des_lin_reg.ipynb
│   ├── _3_logistic_regression.ipynb
│   └── _4_backprob_neuralnetwork.ipynb
├── requirements.txt
└── README.md
```
## Learning Progression

The `notebooks/` directory contains the key progression steps (pre-requisite learning) used to learn how to build the final implementation:

1. Linear regression
2. Gradient descent
3. Logistic regression
4. Neural network with backpropagation

## Purpose

This project is part of a larger **from-scratch NumPy machine learning project** aimed at understanding the mathematical foundations of modern machine learning systems.

The implementation focuses on:

* Linear algebra
* Gradient-based optimisation
* Probability and loss functions
* Computational graphs
* Backpropagation
* Numerical optimisation

The next stage of the project is to build **self-attention and transformer components**.

## Requirements

* Python 3
* NumPy
* Matplotlib

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Limitations

This implementation is purely educational, rather than production-oriented. Hence, it lacks GPU acceleration or the optimisation and numerical safeguards found in established machine-learning frameworks, instead focusing solely on being digestible to learners.
