# NumPy Neural Network

A **multi-layer perceptron neural network framework** built using only **Python and NumPy**, with no machine-learning libraries used to implement the models. The project was developed to understand the mathematical foundations and implementation of neural networks. 

The framework is tested on two classification problems. A simple 2-feature dataset is first used to visualise and analyse the learned decision boundary in [`poc_classification.py`](https://github.com/george-powell/numpy-neural-network/blob/main/poc_classification.py). The MNIST handwritten digits dataset is then used to conduct a controlled comparison of **SGD, Momentum and Adam** in [`main.py`](https://github.com/george-powell/numpy-neural-network/blob/main/main.py). The three models have identical architectures and training conditions, with only the optimiser varied and learning rate tuned. After tuning the learning rate, each model is trained and tested on the entire MNIST dataset 5 times with different seeds. Visualisations and results are stored in [`visualisations`](https://github.com/george-powell/numpy-neural-network/blob/main/visualisations/) and [`results`](https://github.com/george-powell/numpy-neural-network/blob/main/results/). 

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

* Reproducible weight initialisation with random seed being passed in
* Reproducible mini-batch shuffling


## Demonstrations

### 2D Binary Classification — Decision Boundary

A small two-feature binary classification problem was used as a proof of concept to validate the implementation of the neural network. The model was trained using backpropagation and gradient-based optimisation, with the training loss and evolution of the decision boundary visualised throughout training. The code lies in [`poc_classification.py`](https://github.com/george-powell/numpy-neural-network/blob/main/poc_classification.py) with the plots in [`visualisations`](https://github.com/george-powell/numpy-neural-network/blob/main/visualisations/)

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
The **MNIST Dataset** is used to compare **SGD, Momentum and Adam** in [`main.py`](https://github.com/george-powell/numpy-neural-network/blob/main/main.py) with plots in [`visualisations`](https://github.com/george-powell/numpy-neural-network/blob/main/visualisations/), and results in [`results`](https://github.com/george-powell/numpy-neural-network/blob/main/results/).

The experiment uses three otherwise identical networks (`784-128-64-10`, ReLU hidden layers, softmax output, categorical cross-entropy, He initialisation, 25 epochs, batch size 100), varying only the optimisation algorithm and its learning rate. Inputs are standardised using training-set statistics only.

The experiment has two stages:
1. **Learning-rate tuning:** the 60,000 official training images are split into a stratified training set (54,000) and validation set (6,000). Each optimiser is trained once for every learning rate in its own 5-value grid, and the learning rate with the highest validation accuracy is selected. The test set is not used at this stage.
2. **Final evaluation:** each optimiser is retrained from scratch on all 60,000 training images with its selected learning rate, over 5 random seeds (weight initialisation and mini-batch order). Every run is evaluated once on the 10,000-image test set. The same seeds are used for every optimiser, so the runs can be compared as paired differences.

**The experiment measures:**
* Mean test accuracy across seeds
* Accuracy standard deviation across seeds
* Paired accuracy differences between optimisers

Results:
| Optimiser | Selected Learning Rate | Mean Test Accuracy | Std. Deviation |
|-----------|:----------------------:|:------------------:|:--------------:|
| SGD       | 0.3                    | 0.9795             | ±0.0025        |
| Momentum  | 0.03                   | 0.9790             | ±0.0041        |
| Adam      | 0.0003                 | 0.9770             | ±0.0020        |

Once each optimiser's learning rate is tuned, the three reach similar accuracy. The largest paired difference (SGD &minus; Adam) is +0.0025 ± 0.0032 across seeds, which is within seed-to-seed variation, so these results cannot separate the optimisers. The finite test set adds a further binomial standard error of about ±0.0014. Learning-rate choice mattered more than optimiser choice: for SGD, validation accuracy rose from 0.9695 at its default learning rate of 0.01 to 0.9782 at 0.3 (single runs). The full sweep is plotted in [`learning_rate_sensitivity.png`](https://github.com/george-powell/numpy-neural-network/blob/main/visualisations/learning_rate_sensitivity.png).

The experiment illustrates why optimisers must be compared with each learning rate tuned on held-out data and the test set used only once, rather than at shared default settings.

The experiment also demonstrates the network's ability to perform **multi-class classification** with an **arbitrary number of output neurons**, using a ten-neuron softmax output layer and categorical cross-entropy.

## Project Structure

```text
├── poc_classification.py
├── main.py
├── visualisations/
├── results/            
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

This implementation is purely educational, rather than production-oriented. Hence, it lacks GPU acceleration or the optimisation and numerical safeguards found in established machine-learning frameworks, instead prioritising clarity and accessibility over production-level performance. The framework only has support for MLP's. Other neural networks, such as CNN's or random forests are not supported. 
