"""
Optimiser classes for training the network.
Each optimiser recieves a model and its parameter gradients.
Then it updates the model's parameters according to its optimistation algorithm.
The optimiser subclasses inherit the same interface from Optimiser. 
"""

import numpy as np

class Optimiser:
    """
    Parent class for optimisation algorithms.
    """

    def begin_step(self):
        pass
        
    def step(self, model, layer, dW, db):
        raise NotImplementedError
        
    
class SGD(Optimiser):
    """
    Updates network parameters with Stochastic Gradient Descent.

    Parameters
    ----------
    lr : float, default=0.01
        learning rate used for parameter updates.
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, model, layer, dW, db):
        """
        Updates model weights and biases by Stochastic Gradient Descent.

        Parameters
        ----------
        model : NeuralNetwork
            Network model being trained. 
        layer : int
            The network layer being propagated through.
        dW : np.ndarray
            The gradients of the weights parameter with respect to BCE loss.
        db : np.ndarray
            The gradients of the biases parameter with respect to BCE loss.
        """
            
        model.weights[layer] -= self.lr * dW
        model.biases[layer] -= self.lr * db
        

class Momentum(Optimiser):
    """
    Updates network parameters with Momentum.

    Parameters
    ----------
    lr : float, default=0.01
        learning rate used for parameter updates.
    beta : float, default=0.9
        controls the decay rate of past gradients.
    """
    
    def __init__(self, lr=0.01, beta=0.9):
        
        self.lr = lr
        self.beta = beta
        
    def step(self, model, layer, dW, db):
        """
        Updates model weights and biases by Momentum.

        Parameters
        ----------
        model : NeuralNetwork
            Network model being trained. 
        layer : int
            The network layer being propagated through.
        dW : np.ndarray
            The gradients of the weights parameter with respect to BCE loss.
        db : np.ndarray
            The gradients of the biases parameter with respect to BCE loss.
        """

        model.vWs[layer] = self.beta * model.vWs[layer] + dW
        model.vbs[layer] = self.beta * model.vbs[layer] + db

        model.weights[layer] -= self.lr * model.vWs[layer]
        model.biases[layer] -= self.lr * model.vbs[layer]
        
        

class Adam(Optimiser):
    """
    Updates network parameters with Adaptive Moment Estimation.

    Parameters
    ----------
    lr : float, default=0.01
        learning rate used for parameter updates
    beta_1 : float, default=0.9
        exponential decay rate for the first moment's EWMA
    beta_2 : float,default=0.999
    exponential decay rate for the second moment's EWMA

    eps : float

    notes
    -----
    Default values obtained from the original Adam paper. 
    """
    
    
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
        
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.t = 0
        
    def begin_step(self):
        """
        Increments the optimisation step counter used for Adam's zero bias correction.
        """
        self.t += 1
    
    def step(self, model, layer, dW, db):
        """
        Updates model weights and biases by Adaptive Moment Estimation.

        Parameters
        ----------
        model : NeuralNetwork
            Network model being trained. 
        layer : int
            The network layer being propagated through.
        dW : np.ndarray
            The gradients of the weights parameter with respect to BCE loss.
        db : np.ndarray
            The gradients of the biases parameter with respect to BCE loss.
        """
        mW = self.beta_1 * model.mWs[layer] + (1 - self.beta_1)*dW
        mb = self.beta_1 * model.mbs[layer] + (1 - self.beta_1)*db

        vW = self.beta_2 * model.vWs[layer] + (1 - self.beta_2) * dW**2
        vb = self.beta_2 * model.vbs[layer] + (1 - self.beta_2) * db**2

        model.mWs[layer] = mW
        model.mbs[layer] = mb

        model.vWs[layer] = vW
        model.vbs[layer] = vb
            
        # (zero bias correction)
        mW_hat = mW / (1 - self.beta_1**self.t)
        mb_hat = mb / (1 - self.beta_1**self.t)
        
        vW_hat = vW / (1 - self.beta_2**self.t)
        vb_hat = vb / (1 - self.beta_2**self.t)

        model.weights[layer] -= (
            self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        )
        
        model.biases[layer] -= (
            self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
        )