import argparse
import os
import pickle

from LP.LP_linprog import LP_linprog
import numpy as np
from scipy.optimize import linprog
import sympy
import torch.nn
from torch import matmul, Tensor
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

    def convert(self, mu: Tensor, nu : Tensor, s : Tensor):
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

        # Preprocess tensors for solving with solve_with_linprog method
        n = s.size()[0]
        k = f.size()[0]

        d = s[1:] - s[:-1]

        # Construct coefficient vector    
        q = torch.cat((d, torch.ones(k + 1)))

        # Construct inequality constraints. Similar to solve_with_qpth method
        f_block = torch.cat((f, -f), 1)
        f1 = -torch.ones(n, f.size()[1]*2)
        ineq_matrix = torch.cat((f1, f_block), 0)

        # Construct new vector for inequality constraints
        g1 = g
        g2 = -g
        ineq_vector = torch.cat((g1, g2))

        # Construct equality constraint matrix
        eq_matrix = torch.cat((torch.zeros(n), torch.ones(k))).unsqueeze(0) 
        # Construct equality constraint vector. The entries of p should sum up to one
        eq_vector = Tensor([1])

        # Reduce the transpose of ineq_matrix to row echelon form 
        # in order to avoid collinear constraints. 
        # The transpose of the investigated matrix needs to be plugged in.
        # Identify ids of the lines to be suppressed - to be inserted in the call
        # to the linear_linprog method
        _, ids = sympy.Matrix(ineq_matrix.detach().numpy()).rref()

        # Return tensors according to the LP canonical form
        A_in = ineq_matrix
        b_in = ineq_vector
        A_eq = eq_matrix
        b_eq = eq_vector

        # Invert inequality matrix before return statement according
        # to problem formulation. At the same time, apply reduction of 
        # A_in and b_in to linearly independent form by array slicing
        return A_in.T[[ids]], b_in[[ids]], A_eq, b_eq, q
                              
    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
            m.bias.data.fill_(0.0001)

class Linprog_diff(torch.autograd.Function):

    @staticmethod
    def forward(A_in : Tensor,
                b_in : Tensor,
                A_eq : Tensor,
                b_eq : Tensor,
                q : Tensor) -> Tensor:

        # Solve the problem in a differential fashion
        res = MSDProblemInstance.solve_with_linprog(A_in, b_in, A_eq, b_eq, q)
        lamda = Tensor(res.ineqlin.marginals).unsqueeze(0)
        print(f"lamda = {lamda}")
        nu =  Tensor(res.eqlin.marginals).unsqueeze(0)
        print(f"nu = {nu}")
        res = torch.from_numpy(res.x).unsqueeze(0)
        return res.T, lamda, nu
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Method to manage the `ctx` modification.
        `inputs` is the tuple of arguments passed to the `forward` method.
        `output` is the output of `forward`.
        """
        A_in, b_in, A_eq, _, _ = inputs
        res, lamda, nu = output
        ctx.save_for_backward(A_in, b_in, A_eq, res, lamda, nu)
    
    @staticmethod
    def backward(ctx, grad_res, *_):
        
        # Unpack saved tensors
        A_in, b_in, A_eq, res, lamda, nu = ctx.saved_tensors

        # Perturb lamda an nu tensors
        lamda = lamda + torch.rand_like(lamda)
        nu = nu + torch.rand_like(nu)

        # Convert res to float-type tensor
        res = Tensor.float(res)

        # Zero-out gradients
        grad_A_in = grad_b_in = grad_A_eq = grad_b_eq = grad_q = None
        
        # In order to calculate gradients, first solve a linear system for retrieveing dz, dlamda and dnu
        diff_matrix_size = A_in.size()[1] + A_in.size()[0] + A_eq.size()[0]
        diff_matrix = Tensor(diff_matrix_size, diff_matrix_size)

        # Construct (2, 1) block
        diff_matrix[A_in.size()[1]:A_in.size()[1] + A_in.size()[0], :A_in.size()[1]] = A_in
        # Construct (3, 1) block
        diff_matrix[A_in.size()[1] + A_in.size()[0]:, :A_eq.size()[1]] = A_eq
        # Construct (1, 2) block
        top_block = matmul(A_in.T, torch.diag(lamda.squeeze()))
        diff_matrix[:top_block.size()[0], A_in.size()[1]:A_in.size()[1] + top_block.size()[1]] = top_block
        # Construct (1, 3) block
        diff_matrix[:A_eq.size()[1], A_in.size()[1] + top_block.size()[1]:, ] = A_eq.T
        # Construct (2, 2) block
        central_block = torch.diag(((matmul(A_in, res) - b_in.unsqueeze(1))).squeeze())
        diff_matrix[top_block.size()[0]:top_block.size()[0] + central_block.size()[0], 
                    A_in.size()[1]:A_in.size()[1] + central_block.size()[1]] = central_block
        
        # Take the opposite of the differential matrix
        diff_matrix = -diff_matrix

        # Construct vector for linear system
        diff_vector_size = A_in.size()[1] + A_in.size()[0] + A_eq.size()[0]
        diff_vector = torch.Tensor(diff_vector_size).unsqueeze(1)
        
        # Construct (1, 1) entry
        diff_vector[:grad_res.size()[0], :] = grad_res

        # Solve linear problem for differential matrix 
        # differentials = LP_linprog(A_eq=diff_matrix, b_eq=diff_vector)
        # dz, dlamda, dnu = differentials.x

        # TODO Test values - REMOVE LATER
        # Possibly transpose in grad calculations
        dz = torch.rand(1, 145)
        dlamda = torch.rand(1, 6)
        dnu = torch.rand(1, 1)

        # Compute gradients
        grad_q = dz
        grad_A_in =  matmul(torch.diag(lamda.squeeze()), matmul(dlamda.T, res.T)) + matmul(lamda.T, dz)
        grad_b_in = (matmul(-torch.diag(lamda.squeeze()), dlamda.T)).squeeze()
        grad_A_eq = matmul(dnu.T, res.T) + matmul(nu, dz)
        grad_b_eq = -dnu
        
        # Return as many gradients as there are inputs to the forward method.
        # Arguments that do not require a gradient will have a corresponding None value
        return grad_A_in, grad_b_in, grad_A_eq, grad_b_eq, grad_q


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

split = int(0.6 * len(data))
idx_train = np.arange(split)
idx_test = np.arange(split, len(data))

losses = []
for n_iter in tqdm.tqdm(range(100)):
    optimizer.zero_grad()
    i = idx_train[n_iter % len(idx_train)]

    p = torch.autograd.Variable(torch.FloatTensor(data[i].p), requires_grad=True)
    mu = torch.autograd.Variable(torch.FloatTensor(data[i].mu), requires_grad=True)
    nu = torch.autograd.Variable(torch.FloatTensor(data[i].nu), requires_grad=True)
    s = torch.autograd.Variable(torch.FloatTensor(data[i].s), requires_grad=True)

    # Convert the model to canonical form for LP
    A_in, b_in, A_eq, b_eq, q = model.convert(mu, nu, s)

    # Employ the custom-defined function
    function = Linprog_diff.apply

    # Run the model forward
    res, lamda, nu = function(A_in, b_in, A_eq, b_eq, q)

    # Compute loss and back-propagate
    loss = torch.mean(torch.square(p - res))
    loss.backward()

    # Adjust weights
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.show()
