import argparse
import os
import pickle

import sys
from scipy.optimize import linprog

import numpy as np
import sympy
import torch.nn
from torch import Tensor
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

    def convert(self, mu: torch.Tensor, nu: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
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
        
        # Preprocess the f and g tensors to remove redundant entries
        col_list = []
        index_list = []
        for col_number in range(f.size()[1]):
            col = f[:, col_number].tolist()
            if all(item == 0 for item in col) :
                index_list.append(col_number)
            elif col in col_list:
                index_list.append(col_number)
            else:
                col_list.append(col)

        # Remove redundant columns from f
        new_f = []
        # Iterate through rows of f
        for i in range(f.size()[0]):
            old_row = f[i, :].tolist()
            new_row = []
            for j in range(len(old_row)):
                if j not in index_list:
                    new_row.append(old_row[j])
            new_f.append(new_row)

        new_f = torch.autograd.Variable(torch.Tensor(new_f), requires_grad=True)

        # Equally remove redundant rows from g
        new_g = []
        # Iterate through entries of g
        for i in range(g.size()[0]):
            old_entry = g.tolist()[i]
            if i not in index_list:
                new_g.append(old_entry)

        new_g = torch.autograd.Variable(torch.Tensor(new_g), requires_grad=True)
        
        # Preprocess tensors for solve_with_qpth method
        n = s.size()[0]
        k = f.size()[0]

        d = s[1:] - s[:-1]

        # Construct coefficient vector
        q = torch.cat((d, torch.ones(k + 1)))

        # Construct inequality constraints.
        # Concatenate f and -f matrices column-wise 
        # for incorporating the two inequality constraints
        f_block = torch.cat((new_f, -new_f), 1)
        # Add block of -1 entries to account for -t
        f1 = -torch.ones(n, new_f.size()[1]*2)
        # Create new f matrix by concatenating the two block row-wise
        # This has to be transposed according to the problem formulation
        A_in = torch.cat((f1, f_block), 0)
        # Construct new vector for inequality constraints
        # This has the same shape as the amount of columns in f
        g1 = new_g
        g2 = -new_g
        b_in = torch.cat((g1, g2))

        # Construct equality constraint matrix
        A_eq = torch.cat((torch.zeros(n), torch.ones(k))).unsqueeze(0) 
        # Construct equality constraint vector. The entries of p should sum up to one
        b_eq = Tensor([1])

        # Reduce the transpose of the inequality matrix to row echelon form 
        # in order to avoid collinear constraints. 
        # The transpose of the investigated matrix needs to be plugged in.
        # Identify ids of the lines to be suppressed - to be inserted in the call
        # to the solve_with_qpth method

        # PEARSON / COSINE > 99 %
        _, ids = sympy.Matrix(A_in.detach().numpy()).rref()

        # TODO check if partial collinearity - check condition number
        # print(f"CONDITION NUMBER OF A_in ={torch.linalg.cond(A_in[[ids]])}")
        print(f"CONDITION NUMBER OF A_in ={torch.linalg.cond(A_in)}")
        print(f"A_in={A_in}")
        print(f"A_in.shape={A_in.shape}")
        
        #TODO remove - add random number to diagonal of A_in
        # A_in += torch.rand_like(A_in)
        # new_diag = torch.diagonal(A_in) + torch.rand_like(torch.diagonal(A_in))
        # A_in[range(len(new_diag)), range(len(new_diag))] = new_diag
        # A_in = A_in + torch.rand_like(A_in)
        print(f"A_in = {A_in}")

        # Return tensors according to qpth canonical form
        # return q, A_in.T[[ids]], b_in[[ids]], A_eq, b_eq
        return q, A_in.T[[ids]], b_in[[ids]], A_eq, b_eq

    @staticmethod
    def forward(q : Tensor,
                A_in : Tensor,
                b_in : Tensor,
                A_eq : Tensor,
                b_eq : Tensor) -> Tensor:

        # Solve the problem in a differential fashion
        p = torch.FloatTensor(MSDProblemInstance.solve_with_qpth(q, A_in, b_in, A_eq, b_eq))

        return p.T
                              
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

    q, A_in, b_in, A_eq, b_eq = model.convert(mu, nu, s)

    p_pred = model.forward(q, A_in, b_in, A_eq, b_eq)
    
    loss = torch.mean(torch.square(p - p_pred))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.yscale('log')
plt.show()
