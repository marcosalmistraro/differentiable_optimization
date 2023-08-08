import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import tqdm

from torch import Tensor, nn, optim
from torch.nn import Module

import mwe_copy

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
            nn.Linear(in_features=2, out_features=16),
            nn.Tanh(),
            # The encoder learn a compressed representation
            # for parameters a, b, c and d
            nn.Linear(in_features=16, out_features=4)
        )

        # Define the applied constraints for the NLP instance
        ineq_cons = [mwe_copy.ineq_constraint_1, mwe_copy.ineq_constraint_2]

        # Define decoder according to Rosenbrock formulation
        # See https://en.wikipedia.org/wiki/Test_functions_for_optimization#cite_note-11
        self.decoder = lambda param : mwe_copy.DiffCP.apply(param[:2], mwe_copy.rosenbrock,
                                                            None, [],
                                                            param[-2:], ineq_cons)

    def forward(self, input):
        """
        Specifies the forward pass for the model, defining how
        the encoded and decoded representations are expressed for
        a given `input` point.
        """
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded
    

# Create model from the custom-defined class
model = AE()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Define criterion for weight update
criterion = nn.MSELoss()

# Create data set from R^2 by sampling uniformly without repetition
# First set a seed for the pseudo-random generator
random.seed(42)
# Build data set
dataset = []
for i in range(1):
    sample = tuple(random.uniform(0, 1) for _ in range(2))
    if sample not in dataset:
        dataset.append(sample)
        i += 1
    else:
        continue
print(dataset)

# Train the AE
losses = []
idx = np.arange(len(dataset))

for n_iter in tqdm.tqdm(range(3000)):

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
plt.plot(losses)
plt.title('Loss function evolution for AE with embedded NLP')
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.show()