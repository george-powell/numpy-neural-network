"""
How does the choice of optimisation algorithm affect neural-network performance?

This script trains otherwise identical networks on the tabular Iris dataset
from scikit-learn with SGD, Momentum and Adam optimisers and quantifies the
differences in accuracy.

The dataset contains 150 samples of iris flowers with 4 features/inputs:
sepal length, sepal width, petal length and petal width. Given this
information, the network classifies datapoints into one of three classes:
Setosa, Versicolor or Virginica.

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
from sklearn.datasets import load_iris
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


# import Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# parameters
epochs = 500
batch_size = 30
seed = 10

# layer architecture and activations
layer_architecture = [4, 9, 6, 6, 3]
activations = ["gelu", "gelu", "gelu", "softmax"]

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

    for fold, (train_indices, test_indices) in enumerate(skf.split(X, y)):

        # split entire dataset into current training and validation folds
        X_train = X[train_indices]
        y_train = y[train_indices]

        X_test = X[test_indices]
        y_test = y[test_indices]

        # standardise using ONLY the current training fold
        X_train_mean = X_train.mean(axis=0)
        X_train_std = X_train.std(axis=0)

        X_train = (X_train - X_train_mean) / X_train_std
        X_test = (X_test - X_train_mean) / X_train_std

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
            X_train,
            y_train,
            epochs,
            batch_size,
            optimiser_class()
        )

        # evaluate model on validation fold
        model_accuracy = model.test(
            X_test,
            y_test
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

for optimiser, row in optimiser_performance_df.iterrows():

    print(
        f"{optimiser:<10} "
        f"Mean Accuracy: {row['Mean CV Accuracy']:.3f} "
        f"+/- {row['Standard Deviation']:.3f}"
    )

print("\n")
print(optimiser_performance_df)


#################################
# TRAIN / TEST EXPERIMENT
#################################

# shuffle and split the entire dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=seed,
    stratify=y
)

# standardise using ONLY the training data
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)

X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std


#################################
# TRAINING AND TEST RESULTS
#################################

print("Train/Test Results")
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