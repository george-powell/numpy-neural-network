# NumPy Neural Network

A small neural network framework built using only **Python and NumPy**, with no machine-learning libraries used to implement the models. The project was developed to understand the mathematical foundations and implementation of neural networks. 

The framework is tested on two classification problems. A simple 2-feature dataset is first used to visualise and analyse the learned decision boundary in [`poc_classification.py`](https://github.com/george-powell/numpy-neural-network/blob/main/poc_classification.py). The MNIST handwritten digits dataset is then used to conduct a controlled comparison of **SGD, Momentum and Adam.** The three models have identical architectures and training conditions, with only the optimiser varied. Performance is evaluate using **5-fold stratified cross-validation**, followed by fresh models being evaluated on a test/train split, with code found in [`optimiser_comparison.py`](https://github.com/george-powell/numpy-neural-network/blob/main/optimiser_comparison.py). The resulting visualisations from both problems are stored in [`visualisations`](https://github.com/george-powell/numpy-neural-network/blob/main/visualisations/). 

The repository also contains a sequence of notebooks documenting the learning progression from linear regression and gradient descent to a 2-layer Multilayer Perceptron.

## Features

### Neural Network

* Configurable layer architecture and number of neurons
* Forward propagation
* Backpropagation
* Mini-batch training
* Input normalisation

### Activation Functions

* ReLU
* Leaky ReLU
* GELU
* Sigmoid
* Softmax

### Optimisation

* Stochastic Gradient Descent (SGD)
* Momentum
* Adam

### Weight Initialisation

* He initialisation
* Uniform Xavier initialisation
* Normal Xavier initialisation

### Loss Functions

* Binary Cross-Entropy (BCE)
* Categorical Cross-Entropy (CCE)

### Reproducibility

* Configurable random seed
* Reproducible weight initialisation
* Reproducible mini-batch shuffling


## Demonstrations

### 2D Binary Classification — Decision Boundary

A small two-feature binary classification problem was used as a proof of concept to validate the implementation of the neural network. The model was trained using backpropagation and gradient-based optimisation, with the training loss and evolution of the decision boundary visualised throughout training. The code lies in [`poc_visualisation.py`](https://github.com/george-powell/numpy-neural-network/blob/main/poc_visualisation.py) with the plots in [`visualisations`](https://github.com/george-powell/numpy-neural-network/blob/main/visualisations/)

The experiment demonstrates:

* Forward and backward propagation successfully learning a classification boundary
* The different decision-boundary geometries produced by **ReLU** and **GELU** activations
* Successful classification of the training data

#### Results

The log-loss decreases consistently throughout training, indicating that the optimisation process is successfully adjusting the network's weights and biases to reduce the classification error.

The decision-boundary animation shows how the model progressively learns the structure of the dataset. By the final checkpoint, the network correctly classifies all datapoints in this particular dataset.

#### ReLU vs GELU

Using **ReLU** in the hidden layers produces piecewise-linear decision boundaries. For a ReLU neuron, the activation changes at \(z=0\). With two input features,

$$
z = w_1x_1 + w_2x_2 + b = 0
$$

defines a straight-line boundary. Taking $x=x_1$ and $y=x_2$, this becomes

$$
y = -\frac{w_1}{w_2}x - \frac{b}{w_2}.
$$

Each ReLU neuron therefore introduces a linear boundary over the region in which it is active. Combining multiple ReLU neurons allows the network to construct a **piecewise-linear decision boundary**, as observed in the experiment.

In contrast, **GELU** applies a smooth, non-linear transformation to $z$, allowing the network to produce smoother and more complex boundaries. On this small proof-of-concept dataset, GELU produced noticeably more complex decision boundaries and showed a greater tendency towards overfitting than ReLU.

Overall, the experiment demonstrates that the implementation is capable of learning non-linear classification boundaries and that the choice of activation function has a significant effect on the geometry and complexity of the learned boundary.


### MNIST Dataset &mdash; Optimiser Comparison
The **MNIST Dataset** is used to compare **SGD, Momentum and Adam** in [`optimiser_comparison.py`](https://github.com/george-powell/numpy-neural-network/blob/main/optimiser_comparison.py) with plots in [`visualisations`](https://github.com/george-powell/numpy-neural-network/blob/main/visualisations/).

The experiment uses three otherwise identical networks, varying only the optimisation algorithm. **5-fold stratified cross-validation** is used on 10% of the standardised dataset to compare the accuracy and spread of each network. The dataset is then shuffled and **split into training and test sets**. Finally, another set of otherwise identical models are trained then tested on the same 10% subset.

The experiment measures:
* Mean cross-validation accuracy
* Accuracy standard deviation across folds
* Final test accuracy

Results:
| Optimiser | Mean CV Accuracy | Std. Deviation | Test Accuracy |
|-----------|:----------------:|:---------------:|:--------------:|
| SGD       | 0.918             | ±0.004           | 0.896           |
| Momentum  | 0.940             | ±0.006           | 0.937           |
| Adam      | 0.950             | ±0.007           | 0.946           |

The experiment demonstrates how optimiser choice can affect neural network training and performance, while illustrating how controlled experiments can be used to compare different optimisation strategies under otherwise identical conditions.

The experiment also demonstrates the network's ability to perform **multi-class classification** with an **arbitrary number of output neurons**, using a three-neuron softmax output layer and categorical cross-entropy. 

## Project Structure

```text
├── poc_classification.py
├── optimiser_comparison.py
├── visualisations/              
├── src/              
│   ├── neural_network.py       # Neural network architecture and training
│   ├── optimisers.py           # SGD, Momentum and Adam
│   ├── activations.py          # Activation functions and derivatives
│   ├── initialisations         # He, Uniform Xavier, Normal Xavier
│   ├── loss_functions          # Binary and categorical cross entropy
│   ├── visualisation_functions # Decision boundary and log loss plots
│   └── sample_data.py          # Example dataset
├── requirements.txt
└── README.md
```

## Purpose

This project is part of a larger **machine learning project built from scratch using NumPy** aimed at understanding the mathematical foundations of modern machine learning systems.

The implementation focuses on:

* Linear algebra
* Gradient-based optimisation
* Probability and loss functions
* Computational graphs
* Backpropagation
* Numerical optimisation

I am currently working on: **learning self-attention and transformer components**

## Requirements

* Python 3
* NumPy
* Matplotlib
* pandas
* scipy
* scikit-learn
* tensorflow (to load the MNIST dataset ONLY)

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Limitations

This implementation is purely educational, rather than production-oriented. Hence, it lacks GPU acceleration or the optimisation and numerical safeguards found in established machine-learning frameworks, instead prioritising clarity and accessibility over production-level performance.
