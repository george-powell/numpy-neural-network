"""
How does the choice of optimisation algorithm affect neural-network performance?

This script compares SGD, Momentum and Adam on the full MNIST handwritten
digits dataset under controlled conditions: identical architecture,
initialisation, mini-batch order, batch size and number of epochs. The only
differences between the three models are the optimisation algorithm and its
learning rate, which is tuned separately for each optimiser.

learning-rate Tuning:
    The 60,000 training images are split into a training set (90%) and a
    validation set (10%). Each optimiser is trained once for every learning
    rate in its own grid and scored on the validation set. The learning rate
    with the highest validation accuracy is selected for each optimiser.
    The test set is not used at any point in this stage.

Optimiser Evaluation:
    Each optimiser is retrained from scratch on all 60,000 training images
    with its selected learning rate, using several different seeds (weight
    initialisation and mini-batch order). Every run is evaluated once on the
    10,000-image test set. The mean and standard deviation of test accuracy
    across seeds are outputted, along with the paired difference between each
    pair of optimisers. Training log-loss plots and a learning-rate
    sensitivity plot are stored in the visualisations folder. Results are 
    saved as csv files in a results folder.

Note: scikit-learn used for creating the stratified train/validation split.
"""

# importing dependencies
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

import src.visualisation_functions as visualisation_functions
from src.neural_network import NeuralNetwork
from src.optimisers import SGD, Momentum, Adam

# set current working directory as base for file paths
cwd = os.getcwd()
visualisations_path = os.path.join(cwd, "visualisations")
results_path = os.path.join(cwd, "results")
os.makedirs(visualisations_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)


# parameters
epochs = 25
batch_size = 100
seed = 10
validation_fraction = 0.1
n_final_runs = 5

# layer architecture and activations
layer_architecture = [784, 128, 64, 10]
activations = ["ReLU", "ReLU", "Softmax"]

optimisers = [SGD, Momentum, Adam]

# learning rates searched for each optimiser (spaced by a factor of ~3).
# Momentum's update is v = beta*v + grad, so with beta = 0.9 its effective step
# is ~10x its learning rate; its grid is shifted down by 10x to cover the same
# effective step sizes as SGD.
learning_rate_grids = {
    "SGD": [0.01, 0.03, 0.1, 0.3, 1.0],
    "Momentum": [0.001, 0.003, 0.01, 0.03, 0.1],
    "Adam": [0.0001, 0.0003, 0.001, 0.003, 0.01],
}


#################################
# HELPER FUNCTIONS
#################################

def standardise(X_fit, *others):

    mean = X_fit.mean()
    std = X_fit.std()

    return [(X - mean) / std for X in (X_fit, *others)]


def train_and_evaluate(optimiser_class, lr, X_train, y_train, X_eval, y_eval, model_seed):

    model = NeuralNetwork(
        layer_architecture,
        activations,
        loss="CCE",
        initialisation="He",
        seed=model_seed
    )

    loss_data, _ = model.fit(
        X_train,
        y_train,
        epochs,
        batch_size,
        optimiser_class(lr=lr)
    )

    accuracy = model.test(X_eval, y_eval)

    return accuracy, loss_data


#################################
# LOAD DATA
#################################

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# reshape X matrix -> vector and scale pixel values to [0, 1]
X_train_full = X_train_full.reshape(X_train_full.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

print("--------------------------------")
print("Full training data shape:", X_train_full.shape)
print("Testing data shape:", X_test.shape)
print("--------------------------------")


#################################
# STAGE 1: LEARNING-RATE TUNING
#################################

# hold out a stratified validation set from the training data
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=validation_fraction,
    stratify=y_train_full,
    random_state=seed
)

# standardise with training-split statistics only
X_tr, X_val = standardise(X_tr, X_val)

print("\nStage 1: Learning-Rate Tuning")
print("Training data shape:", X_tr.shape)
print("Validation data shape:", X_val.shape)
print("--------------------------------")

tuning_records = []

for optimiser_class in optimisers:

    name = optimiser_class.__name__

    for lr in learning_rate_grids[name]:

        # a learning rate that is too large can diverge to inf/nan, which is
        # a valid (poor) result in a sweep, so numerical warnings are silenced
        with np.errstate(all="ignore"):
            val_accuracy, _ = train_and_evaluate(
                optimiser_class,
                lr,
                X_tr,
                y_tr,
                X_val,
                y_val,
                seed
            )

        tuning_records.append({
            "Optimiser": name,
            "Learning Rate": lr,
            "Validation Accuracy": val_accuracy
        })

        print(f"{name:<10} lr={lr:<8} Validation Accuracy: {val_accuracy:.4f}", flush=True)

tuning_df = pd.DataFrame(tuning_records)
tuning_df.to_csv(os.path.join(results_path, "tuning_results.csv"), index=False)

# select the learning rate with the highest validation accuracy per optimiser
best_lr = {}

for optimiser_class in optimisers:

    name = optimiser_class.__name__
    group = tuning_df[tuning_df["Optimiser"] == name]

    best_row = group.loc[group["Validation Accuracy"].idxmax()]
    best_lr[name] = float(best_row["Learning Rate"])

    # a best value on the edge of the grid means the true optimum may lie outside it
    grid = learning_rate_grids[name]
    if best_lr[name] in (grid[0], grid[-1]):
        print(
            f"WARNING: {name} selected the edge of its grid (lr={best_lr[name]}). "
            "Extend the grid and re-run."
        )

print("\nSelected Learning Rates")
print("--------------------------------")
for name, lr in best_lr.items():
    print(f"{name:<10} lr={lr}")

# free the stage 1 copies of the data
del X_tr, X_val, y_tr, y_val


#################################
# STAGE 2: FINAL EVALUATION
#################################

# standardise with full training set statistics only
X_fit, X_eval = standardise(X_train_full, X_test)

print("\nStage 2: Final Evaluation")
print("Training data shape:", X_fit.shape)
print("Testing data shape:", X_eval.shape)
print(f"Runs per optimiser: {n_final_runs}")
print("--------------------------------")

test_accuracy = {}
example_loss_data = {}

for optimiser_class in optimisers:

    name = optimiser_class.__name__
    test_accuracy[name] = []

    for run in range(n_final_runs):

        # the same seeds are used for every optimiser, so run i of each
        # optimiser starts from identical weights and sees identical batches
        run_accuracy, loss_data = train_and_evaluate(
            optimiser_class,
            best_lr[name],
            X_fit,
            y_train_full,
            X_eval,
            y_test,
            seed + run
        )

        test_accuracy[name].append(run_accuracy)

        if run == 0:
            example_loss_data[name] = loss_data

        print(f"{name:<10} run {run + 1}/{n_final_runs} Test Accuracy: {run_accuracy:.4f}", flush=True)


#################################
# FINAL RESULTS
#################################

final_df = pd.DataFrame(
    columns=["Learning Rate", "Mean Test Accuracy", "Standard Deviation", "Min", "Max"]
)
final_df.index.name = "Optimiser"

for name, accuracies in test_accuracy.items():

    accuracies = np.array(accuracies)

    final_df.loc[name] = [
        best_lr[name],
        accuracies.mean(),
        accuracies.std(ddof=1),
        accuracies.min(),
        accuracies.max()
    ]

final_df.to_csv(os.path.join(results_path, "final_results.csv"))

print(f"\nTest Results ({n_final_runs} seeds, tuned learning rates)")
print("--------------------------------")
print(final_df.round(4))

# paired differences: run i of each optimiser shares its initialisation and batch order
print("\nPaired Differences in Test Accuracy (mean ± std across seeds)")
print("--------------------------------")
for name_a, name_b in itertools.combinations(test_accuracy, 2):

    difference = np.array(test_accuracy[name_a]) - np.array(test_accuracy[name_b])
    print(f"{name_a} - {name_b}: {difference.mean():+.4f} ± {difference.std(ddof=1):.4f}")

# the std across seeds only measures variation from initialisation and batch
# order. The finite test set adds a further, separate source of uncertainty.
best_mean = final_df["Mean Test Accuracy"].astype(float).max()
sampling_error = np.sqrt(best_mean * (1 - best_mean) / len(y_test))
print(
    f"\nTest-set sampling error (binomial standard error) at {best_mean:.3f} "
    f"accuracy: ±{sampling_error:.4f}"
)


#################################
# VISUALISATIONS
#################################

# training log-loss plots from the first final run of each optimiser
for name, loss_data in example_loss_data.items():

    fig = visualisation_functions.plot_log_loss(loss_data)

    fig.savefig(
        os.path.join(
            visualisations_path,
            f"{name.lower()}_log_loss.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

# learning-rate sensitivity: validation accuracy against learning rate
fig, axes = plt.subplots(1, len(optimisers), figsize=(12, 3.5), sharey=True)

for ax, optimiser_class in zip(axes, optimisers):

    name = optimiser_class.__name__
    group = tuning_df[tuning_df["Optimiser"] == name]

    ax.semilogx(group["Learning Rate"], group["Validation Accuracy"], marker="o")
    ax.axvline(best_lr[name], linestyle="--", color="grey", label="selected")
    ax.set_title(name)
    ax.set_xlabel("Learning rate")
    ax.set_ylim(0.9, 1.0)

axes[0].set_ylabel("Validation accuracy")
axes[0].legend()
fig.suptitle("Learning-rate sensitivity (values below 0.90 are off-scale)")

fig.savefig(
    os.path.join(visualisations_path, "learning_rate_sensitivity.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)