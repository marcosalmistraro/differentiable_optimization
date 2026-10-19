import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import tqdm

from torch import Tensor, nn, optim
from torch.nn import Module

import decoder_functions

# --- Model configuration ---
INPUT_DIM = 2        # Dimensionality of the points sampled from R^2
HIDDEN_DIM = 16      # Width of the encoder's hidden layer
LATENT_DIM = 4       # Size of the learned parameter representation
N_OBJ_PARAMS = 2     # Number of latent entries feeding the objective function

# --- Training configuration ---
N_SAMPLES = 1        # Number of data points in the training set
N_ITERATIONS = 3000  # Number of optimization steps
LEARNING_RATE = 1e-5
SEED = 42            # Seed for the pseudo-random generator

PLOT_SUBTITLE = "Training on one point - Rosenbrock function parameterized by α and β"

class AE(Module):
    """
    Defines the structure of the model, where the encoder stage is constituted
    by a NN outputting a learned representation of the parameters of the NLP
    instance specified as decoder.
    """

    def __init__(self):
        super().__init__()

        # Define encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=INPUT_DIM, out_features=HIDDEN_DIM),
            nn.Tanh(),
            # The encoder learn a compressed representation
            # for parameters a, b, c and d
            nn.Linear(in_features=HIDDEN_DIM, out_features=LATENT_DIM)
        )

        # Define the applied constraints for the NLP instance
        ineq_cons = [decoder_functions.ineq_constraint_1, decoder_functions.ineq_constraint_2]

        # Define decoder according to Rosenbrock formulation
        # See https://en.wikipedia.org/wiki/Test_functions_for_optimization#cite_note-11
        self.decoder = lambda param, input : decoder_functions.DiffCP.apply(param[:N_OBJ_PARAMS], decoder_functions.rosenbrock,
                                                                   None, [],
                                                                   param[-2:], ineq_cons,
                                                                   input)

    def forward(self, input):
        """
        Specifies the forward pass for the model, defining how
        the encoded and decoded representations are expressed for
        a given `input` point.
        """
        encoded = self.encoder(input)
        decoded = self.decoder(encoded, input)
        return decoded


def main():
    # Create model from the custom-defined class
    model = AE()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Define criterion for weight update
    criterion = nn.MSELoss()

    # Create data set from R^2 by sampling uniformly without repetition
    # First set a seed for the pseudo-random generator
    random.seed(SEED)
    # Build data set
    dataset = []
    for i in range(N_SAMPLES):
        sample = tuple(random.uniform(0, 1) for _ in range(2))
        if sample not in dataset:
            dataset.append(sample)
            i += 1
        else:
            continue

    # Train the AE
    losses = []
    idx = np.arange(len(dataset))

    for n_iter in tqdm.tqdm(range(N_ITERATIONS)):

        # Take a data point from dataset
        i = idx[n_iter % len(idx)]
        input = torch.tensor((dataset[i]), dtype=torch.float32, requires_grad=True)

        # Zero-out gradients to avoid accumulation
        optimizer.zero_grad()

        # Compute the decoded version of the input vector as output of the AE
        decoded = model.forward(input)

        # Compute the training loss for the considered data instance
        train_loss = Tensor(criterion(input, decoded))

        # Compute gradients by traversing the graph in a backward fashion
        train_loss.backward()

        # Update parameters to optimize model
        optimizer.step()

        # Append loss to the array of losses for the training phase
        losses.append(train_loss.item())

    # Display the evolution of the loss function over
    # the training phase
    fig = plt.figure()

    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss function')
    fig.suptitle("Loss function evolution for AE with embedded NLP", fontsize=18)
    fig.text(0.5, 0.93, PLOT_SUBTITLE,
             horizontalalignment="center")

    plt.show()


if __name__ == "__main__":
    main()
