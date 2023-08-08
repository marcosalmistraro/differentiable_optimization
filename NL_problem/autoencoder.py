import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import tqdm

from torch import Tensor, nn, optim
from torch.nn import Module

import mwe

class AE(Module):

    def __init__(self):
        super().__init__()

        # Define encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=2)
        )

        # Define the applied constraints for the NLP instance
        rosenbrock_cons = [lambda x : x[1] - 1 - (x[0] - 1)**2,
                           lambda x : 2 - x[0] - x[1]]

        # Define decoder
        self.decoder = lambda encoded : mwe.DiffCP.apply(encoded, mwe.rosenbrock,
                                                         None, [],
                                                         None, rosenbrock_cons)

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded
    

# Create model from the custom-defined class
model = AE()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

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

# Train the AE
losses = []
idx = np.arange(len(dataset))

for n_iter in tqdm.tqdm(range(1000)):

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

    # Update parameters
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