import argparse
import os
import pickle

import numpy as np
import torch.nn
import tqdm
from matplotlib import pyplot as plt

from simulations.msd import MoNA, MSDProblemInstance

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-regen',
    default=False,
    action='store_true',
    help='(Re-)generate dataset'
)
args = parser.parse_args()

if args.regen:

    # Go to: https://mona.fiehnlab.ucdavis.edu/downloads
    # Download: Libraries > MassBank > CASMI 2016
    # Store the json file in the data/ folder
    mona = MoNA(os.path.join(DATA_FOLDER, 'MoNA-export-CASMI_2016.json'))

    # Generate random problem instances
    data = [mona.random() for _ in range(1000)]

    # Save problem instances
    with open(os.path.join(DATA_FOLDER, 'mona.pickle'), 'wb') as f:
        pickle.dump(data, f)
else:
    # Load problem instances
    if not os.path.exists(os.path.join(DATA_FOLDER, 'mona.pickle')):
        raise FileNotFoundError('Dataset does not exist. Call "python problem1.py -regen" first.')
    with open(os.path.join(DATA_FOLDER, 'mona.pickle'), 'rb') as f:
        data = pickle.load(f)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 1),
            torch.nn.Tanh()
        )
        self.apply(Model.init_weights)

    def forward(self, mu: torch.Tensor, nu: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Deconvolute mass spectra.

        Empirical spectrum is first corrected to account for the distributional shift between
        the domains that produced the empirical and theoretical spectra.
        Indeed, data were produced with different ionization modes.
        Then, the model estimates the contribution of each theoretical spectrum (each row of `mu`)
        to the empirical spectrum `nu`.

        Args:
            mu: Theoretical spectra. `mu[i, j]` is the intensity of compound `i` for mass ratio `j`.
            nu: Empirical spectra. `nu[j]` is the intensity of the mixture for mass ratio `j`.
            s: Mass ratios, expressed in m/z. `s[j]` is the mass ratio `j` for `mu[i, j]` and `nu[j]`.

        See:
            "The Wasserstein Distance as a Dissimilarity Measure for Mass Spectra with Application
            to Spectral Deconvolution"
            by S. Majewski, M. A. Ciach, M. Startek, W. Niemyska, B. Miasojedow and A. Gambin.
        """

        # Preprocessing on empirical spectrum
        corrections = self.mlp(torch.cat((nu.unsqueeze(1), s.unsqueeze(1)), dim=1)) + 1.
        nu = nu * torch.squeeze(corrections)

        # Ensure mass spectra sum to 1 each
        f = torch.cumsum(mu / torch.sum(mu, dim=1).unsqueeze(1), dim=1)
        g = torch.cumsum(nu / torch.sum(nu), dim=0)

        # Objective function:
        #     min_{t, p} np.sum(t[:-1] * d)
        # Constraints:
        #     np.sum(p) == 1
        #     -t + f.T @ p <= g
        #     -t - f.T @ p <= -g

        p = torch.FloatTensor(MSDProblemInstance.solve_with_cvxpy(
            f.cpu().data.numpy(),
            g.cpu().data.numpy(),
            s.cpu().data.numpy(),
        ))  

        # The Problem is currently solved by disconnecting it from the computational graph -> no gradient
        # It needs to be solved in a differentiable fashion instead
        p = p + torch.sum(0 * g)  # Trick to have a gradient

        return p

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
            m.bias.data.fill_(0.0001)


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

split = int(0.6 * len(data))
idx_train = np.arange(split)
idx_test = np.arange(split, len(data))

losses = []
for n_iter in tqdm.tqdm(range(100)):
    optimizer.zero_grad()
    i = idx_train[n_iter % len(idx_train)]
    p = torch.FloatTensor(data[i].p)
    mu = torch.FloatTensor(data[i].mu)
    nu = torch.FloatTensor(data[i].nu)
    s = torch.FloatTensor(data[i].s)
    p_pred = model.forward(mu, nu, s)
    loss = torch.mean(torch.square(p - p_pred))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # Take loss at the last iteration and print to file
    if n_iter == 99:
        dir_path = "/Users/marco/Documents/Repos/differentiable_optimization/data_analysis"
        with open(os.path.join(dir_path, "losses_cvxpy.txt"), "a") as file:
            file.write(f"{loss}\n")

plt.plot(losses)
plt.title("Non-differentiable LP instance")
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.show()
