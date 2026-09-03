"""
The script trains a NumPy-based neural network using Adam optimisation,
then generates a decision-boundary GIF and a training loss plot.
"""

# import dependencies
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

import src.sample_data as sample_data
import src.visualisation_functions as visualisation_functions
from src.neural_network import NeuralNetwork
from src.optimisers import Adam

# set current working directory as base for file paths
pwd = os.getcwd()
visualisations_path = os.path.join(pwd, "visualisations")
os.makedirs(visualisations_path, exist_ok=True)

# import sample data
X = sample_data.X
y = sample_data.y
print("-"*10, "BUILDING AND TRAINING THE MODEL", "-"*10)
# parameters
epochs = 2000
batch_size = 16
seed = 10

# layer architecture and activations
layer_architecture = [2, 8, 4, 1]
activations = ["relu", "relu", "sigmoid"]

# initialising the model
model = NeuralNetwork(
    layer_architecture,
    activations,
    loss = "BCE",
    initialisation = "He",
    seed = seed
)

# training the model on the dataset
loss_data, checkpoints = model.fit(
    X,
    y,
    epochs,
    batch_size,
    optimiser=Adam(),
    checkpoint_interval=50
)
print("-"*10, "BUILDING AND TRAINING COMPLETE", "-"*10)

print("-"*10, "PERFORMING VISUALISATIONS", "-"*10)
# to store plot images for each interval
frames = []

for epoch, checkpoint in checkpoints.items():

    # loads previously saved model weights and biases at the checkpointed epoch
    model.load_checkpoint(checkpoint)

    # create decision boundary at each interval
    fig = visualisation_functions.plot_decision_boundary(
        X,
        y,
        model.forward,
        title=f"Decision Boundary — Epoch {epoch}"
    )
    
    # render the figure
    fig.canvas.draw()

    # convert figure to RBGA pixel data array for imageio
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    frames.append(image)

    plt.close(fig)

# save gif to visualisations folder in cwd
imageio.mimsave(
    os.path.join(visualisations_path, "decision_boundary.gif"),
    frames,
    duration=0.5,
    loop=0
)


# creating log loss plot over training
fig = visualisation_functions.plot_log_loss(loss_data)

fig.savefig(
    os.path.join(visualisations_path, "poc_loss.png"),
    dpi=200
)

plt.close(fig)

print("-"*10, "VISUALISATIONS COMPLETE", "-"*10)