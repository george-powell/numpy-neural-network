"""
Implementation of a fully connected neural network using NumPy.

Features:
- Arbitrary layers and activation functions (from activations.py)
- Backpropagation
- Mini-batch training 
- External optimisers (from optimisers.py)
"""

import numpy as np
import utils # for BCE function.
import activations

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

    Notes
    -----
    Weights use the convention W.shape = (n_output, n_input), so that
    forward propagation is performed as X @ W.T + b.
    """

    def __init__(self, layer_sizes, layer_activations):

        if (len(layer_sizes) != len(layer_activations) + 1):
            raise ValueError(
                "Incorrect match of layers and activation functions. "
                "Check size of layer_sizes and layer_activations."
            )
            
        self.layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.layer_activations = [a.lower() for a in layer_activations]

        self.weights = []
        self.biases = []

        for layer in range(self.layers - 1):

            # He intiialisation, suitable with ReLU-like activation functions.
            W = np.random.randn(
                self.layer_sizes[layer + 1], 
                self.layer_sizes[layer]
            ) * np.sqrt(2 / self.layer_sizes[layer])
            self.weights.append(W)
            
            b = (np.zeros(self.layer_sizes[layer + 1]))
            self.biases.append(b)

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
        if (self.layer_sizes[-1] != 1):
            raise ValueError(
                "Only single neuron output currently supported"
            )
            

        self.Zs = [X @ self.weights[0].T + self.biases[0]]
        self.As = [
            X, 
            activations.ACTIVATIONS[
                self.layer_activations[0]
                ](self.Zs[0])
        ]
        
        for layer in range(self.layers - 2):
            
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

            optimiser.step(self, layer, dW, db)
            
            if layer != 0:
                
                dA = dZ @ self.weights[layer]
                dZ = dA * activations.ACTIVATIONS_PRIME[
                    self.layer_activations[layer - 1]
                    ](self.Zs[layer - 1])


    def fit(self, X, y, epochs, batch_size, optimiser):
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
            Binary cross-entropy loss after each epoch.
        """
        
        # standardise input features.
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        # ensure shape of (n_samples, 1) to prevent broadcasting 
        # producing an unintended (n_samples, n_samples) array.
        y = y.reshape(-1, 1)

        samples = X.shape[0]
        loss_data = []

        for epoch in range(epochs):

            # shuffle dataset at beginning of each epoch.
            indicies = np.random.permutation(samples)

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
            loss = utils.binary_cross_entropy(y, An)
            loss_data.append(loss)

        return loss_data