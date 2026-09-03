"""
How does the choice of optimisation algorithm affect neural-network performance?

This script trains otherwise identical networks on 10% of the MNIST handwritten
digits dataset's training set with SGD, Momentum and Adam optimisers.

The three models are tested by 5-fold stratified cross-validation on the
entire dataset, with the mean accuracy and standard deviation for each
optimiser outputted.

Next, the dataset is shuffled and split into training and test sets. Another
3 otherwise identical models are trained on the training data and tested on
the unseen test data. Training log-loss plots are created and stored in the
visualisations folder, and the test accuracy for each optimiser is outputted.

Note: scikit-learn used for creating the stratified folds and splitting data into training and test sets.
"""

# importing dependencies
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import os

import src.visualisation_functions as visualisation_functions
from src.neural_network import NeuralNetwork
from src.optimisers import SGD, Momentum, Adam

# set current working directory as base for file paths
pwd = os.getcwd()
visualisations_path = os.path.join(pwd, "visualisations")
os.makedirs(visualisations_path, exist_ok=True)


# import MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[:6000] / 255.0
X_test = X_test[:1000] / 255.0

y_train = y_train[:6000]
y_test = y_test[:1000]

# reshape X matrix -> vector
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
print("--------------------------------")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("--------------------------------")

# parameters
epochs = 50
batch_size = 100
seed = 10

# layer architecture and activations
layer_architecture = [784, 128, 64, 10]
activations = ["ReLU", "ReLU", "Softmax"]

optimisers = [SGD, Momentum, Adam]


#################################
# 5-FOLD CROSS-VALIDATION
#################################

# create stratified folds across the entire dataset
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=seed
)

# to store cross-validation optimiser data
optimiser_performance_df = pd.DataFrame(
    columns=["Mean CV Accuracy", "Standard Deviation"]
)
optimiser_performance_df.index.name = "Optimiser"


for optimiser_class in optimisers:

    fold_accuracy = []

    for fold, (train_indices, test_indices) in enumerate(skf.split(X_train, y_train)):

        # split entire dataset into current training and validation folds
        Xf_train = X_train[train_indices]
        yf_train = y_train[train_indices]

        Xf_test = X_train[test_indices]
        yf_test = y_train[test_indices]

        # initialise model
        model = NeuralNetwork(
            layer_architecture,
            activations,
            loss="CCE",
            initialisation="He",
            seed=seed + fold
        )

        # train model
        model.fit(
            Xf_train,
            yf_train,
            epochs,
            batch_size,
            optimiser_class()
        )

        # evaluate model on validation fold
        model_accuracy = model.test(
            Xf_test,
            yf_test
        )

        fold_accuracy.append(model_accuracy)

    # calculate mean and standard deviation across the 5 folds
    mean_accuracy = np.mean(fold_accuracy)
    std_accuracy = np.std(fold_accuracy)

    optimiser_performance_df.loc[optimiser_class.__name__] = [
        mean_accuracy,
        std_accuracy
    ]


#################################
# CROSS-VALIDATION RESULTS
#################################

print("\n5-Fold Cross-Validation Results")
print("--------------------------------")
print(optimiser_performance_df)


#################################
# TRAIN / TEST EXPERIMENT
#################################

print("\nTrain/Test Results")
print("------------------")

for optimiser_class in optimisers:

    # initialise model
    model = NeuralNetwork(
        layer_architecture,
        activations,
        loss="CCE",
        initialisation="He",
        seed=seed
    )

    # train model
    loss_data, checkpoints = model.fit(
        X_train,
        y_train,
        epochs,
        batch_size,
        optimiser_class()
    )

    # evaluate model on unseen test data
    model_accuracy = model.test(
        X_test,
        y_test
    )

    print(
        f"{optimiser_class.__name__:<10} "
        f"Test Accuracy: {model_accuracy:.3f}"
    )

    # create log-loss plot
    fig = visualisation_functions.plot_log_loss(loss_data)

    fig.savefig(
        os.path.join(
            visualisations_path,
            f"{optimiser_class.__name__.lower()}_log_loss.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)
