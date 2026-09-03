"""
Implementation of a fully connected neural network using NumPy.

Features:
- Arbitrary layers and activation functions (from activations.py)
- Backpropagation
- Mini-batch training 
- External optimisers (from optimisers.py)
"""

import numpy as np
import src.initialisations as initialisations
import src.loss_functions as loss_functions
import src.activations as activations

class NeuralNetwork:
    """
    Fully connected feed-forward neural network implemented from scratch.

    Parameters
    ----------
    sample_size : int
        Number of samples in the training dataset.
    layer_sizes : list[int]
        Number of neurons in each layer, including the input and output
        layers. For example, [2, 64, 32, 1].
    layer_activations : list[str]
        Activation function used after each layer transition. Must contain
        one fewer element than layer_sizes.
    seed : int
        Random number generator seed used in np.random.

    Notes
    -----
    Weights use the convention W.shape = (n_output, n_input), so that
    forward propagation is performed as X @ W.T + b.
    He initialisation and random seed of 10 set by default.
    """

    def __init__(self, layer_sizes, layer_activations, loss, initialisation="He", seed=10):

        if (len(layer_sizes) != len(layer_activations) + 1):
            raise ValueError(
                "Incorrect match of layers and activation functions. "
                "Check size of layer_sizes and layer_activations."
            )
            
        self.layer_sizes = layer_sizes
        self.layer_activations = [a.lower() for a in layer_activations]
        self.loss = loss.lower()
        self.initialisation = initialisation
        self.RNG = np.random.default_rng(seed)
        
        self.bce = False
        self.cce = False

        if self.loss == "bce":
            self.bce = True
            if self.layer_sizes[-1] != 1:
                raise ValueError("BCE requires exactly 1 output neuron.")
            if self.layer_activations[-1] != "sigmoid":
                raise ValueError("BCE only supports sigmoid output activation.")

        elif self.loss == "cce":
            self.cce = True
            if self.layer_sizes[-1] < 2:
                raise ValueError("Performing single-class classification with CCE")
            if self.layer_activations[-1] != "softmax":
                raise ValueError("CCE only supports softmax output activation.")

        # Initialising weights and biases
        self.weights, self.biases = getattr(initialisations, initialisation)(self.layer_sizes, self.RNG)

        # Optimiser state. (Momentum & Adam)
        # Maintained by optimiser classes during training.
        self.mWs = [np.zeros_like(W) for W in self.weights]
        self.mbs = [np.zeros_like(b) for b in self.biases]

        self.vWs = [np.zeros_like(W) for W in self.weights]
        self.vbs = [np.zeros_like(b) for b in self.biases]


    def forward(self, X):
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Output activation of the final layer.

        Notes
        -----
        Zs and As stored as required in backpropagation.
        """
        
        if (X.ndim > 1) and (self.layer_sizes[0] != X.shape[1]):
            raise ValueError(
                "layer_sizes[0] does not equal the provided number of inputs."
            )
            
        self.Zs = [X @ self.weights[0].T + self.biases[0]]
        self.As = [
            X, 
            activations.ACTIVATIONS[
                self.layer_activations[0]
                ](self.Zs[0])
        ]
        
        for layer in range(len(self.layer_sizes) - 2):
            
            Z = self.As[layer + 1] @ self.weights[layer + 1].T + self.biases[layer + 1]
            self.Zs.append(Z)
            A = activations.ACTIVATIONS[
                self.layer_activations[layer + 1]
                ](Z)
            self.As.append(A)

        return self.As[-1]

    def backward(self, y, optimiser):
        """
        Perform backpropagation and update network parameters.

        Parameters
        ----------
        y : np.ndarray
            True binary labels for the current mini-batch.
        optimiser : Optimiser
            Optimisation algorithm used to update network parameters.

        Notes
        -----
        Gradients are calculated using the stored activations and
        pre-activation values from the forward pass.
        """
        
        dZ = self.As[-1] - y
        
        for layer in reversed(range(len(self.weights))):

            dW = (dZ.T @ self.As[layer]) / y.shape[0]
            db = (np.sum(dZ, axis=0)) / y.shape[0]
            
            if layer != 0:
                
                dA = dZ @ self.weights[layer]
                dZ = dA * activations.ACTIVATIONS_PRIME[
                    self.layer_activations[layer - 1]
                    ](self.Zs[layer - 1])

            optimiser.step(self, layer, dW, db)


    def fit(self, X, y, epochs, batch_size, optimiser, checkpoint_interval=None):
        """
        Train the network using mini-batch gradient descent with specified optimiser.

        Parameters
        ----------
        X : np.ndarray
            Training inputs with shape (n_samples, n_features).
        y : np.ndarray
            True binary labels.
        epochs : int
            Number of passes through the training dataset.
        batch_size : int
            Number of samples used for each parameter update.
        optimiser : Optimiser
            Optimisation algorithm used to update network parameters.

        Returns
        -------
        list[float]
            loss after each epoch.
        """

        if self.loss in ["bce", "binary cross entropy", "binary_cross_entropy"]:
            y = y.reshape(-1,1)
        elif self.loss in ["cce", "categorical cross entropy", "categorical_cross_entropy"]:
            
            # convert integer targets to the equivalent 2D one-hot matrix
            if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
                y = y.squeeze() 
                y = np.eye(self.layer_sizes[-1])[y]
        else:
            raise ValueError("Only binary or categorical cross entrophy allowed. enter 'BCE' or 'CCE'.")
            
        samples = X.shape[0]
        loss_data = []
        checkpoints = {}

        for epoch in range(epochs):

            # shuffle dataset at beginning of each epoch.
            indicies = self.RNG.permutation(samples)

            X_shuffled = X[indicies]
            y_shuffled = y[indicies]

            # divide dataset into mini-batches.
            for start in range(0, samples, batch_size):

                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # one optimiser step corresponds to one mini-batch.
                optimiser.begin_step()
                self.forward(X_batch)
                self.backward(y_batch, optimiser)

            # evaluate loss of complete training dataset for visualisation
            An = self.forward(X)
            if self.loss in ["bce", "binary cross entropy", "binary_cross_entropy"]:
                loss = loss_functions.binary_cross_entropy(y, An)
            else:
                loss = loss_functions.categorical_cross_entropy(y, An)
            
            loss_data.append(loss)

            if checkpoint_interval is not None and (epoch + 1) % checkpoint_interval == 0:

                checkpoints[epoch + 1] = {
                    "weights": [W.copy() for W in self.weights],
                    "biases": [b.copy() for b in self.biases]
                }
        return loss_data, checkpoints

    
    def load_checkpoint(self, checkpoint):
        """
        Load a previously saved model checkpoint.
        """

        self.weights = [W.copy() for W in checkpoint["weights"]]
        self.biases = [b.copy() for b in checkpoint["biases"]]

    
    def test(self, X, y):
        """
        Evaluate classification accuracy on unseen data.

        Parameters
        ----------
        X : np.ndarray
            Test inputs with shape (n_samples, n_features).
        y : np.ndarray
            True binary labels.

        Returns
        -------
        Classifiation accuracy as a probability.
        
        """
        if self.bce:
            A = self.forward(X)
            predictions = (A >= 0.5).astype(int).ravel()
            y = y.ravel()

        elif self.cce:
            A = self.forward(X)
            predictions = np.argmax(A, axis=1)

        accuracy = np.mean(predictions == y)

        return accuracy