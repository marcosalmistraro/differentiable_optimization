import argparse
import os
import os.path
import pickle

from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor, diag, matmul
import torch.nn
import tqdm

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
        raise FileNotFoundError('Dataset does not exist. Call "python3 problem1_cvxpy_diff.py -regen" first.')
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

    @staticmethod
    def preprocess_data(f: Tensor, g: Tensor):

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

        # Equally remove redundant rows from g
        new_g = []
        # Iterate through entries of g
        for i in range(g.size()[0]):
            old_entry = g.tolist()[i]
            if i not in index_list:
                new_g.append(old_entry)

        # Return preprocessed vectors as new autograd variables with trackable gradient
        new_f = torch.autograd.Variable(Tensor(new_f), requires_grad=True)
        new_g = torch.autograd.Variable(Tensor(new_g), requires_grad=True)

        return new_f, new_g


    def convert(self, mu: Tensor, nu: Tensor, s: Tensor) -> Tensor:
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
        f, g = Model.preprocess_data(f, g)
        
        # Preprocess tensors for solve_with_qpth method
        n = s.size()[0]
        k = f.size()[0]

        d = s[1:] - s[:-1]

        # Construct coefficient vector
        q = torch.cat((d, torch.ones(k + 1)))

        # Construct inequality constraints.
        # Concatenate f and -f matrices column-wise 
        # for incorporating the two inequality constraints
        f_block = torch.cat((f, -f), 1)
        # Add block of -1 entries to account for -t
        f1 = -torch.ones(n, f.size()[1]*2)
        # Create new f matrix by concatenating the two block row-wise
        # This has to be transposed according to the problem formulation
        A_in = torch.cat((f1, f_block), 0)
        # Construct new vector for inequality constraints
        # This has the same shape as the amount of columns in f
        g1 = g
        g2 = -g
        b_in = torch.cat((g1, g2))

        # Construct equality constraint matrix
        A_eq = torch.cat((torch.zeros(n), torch.ones(k))).unsqueeze(0) 
        # Construct equality constraint vector. The entries of p should sum up to one
        b_eq = Tensor([1])

        return q, A_in.T, b_in, A_eq, b_eq
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
            m.bias.data.fill_(0.0001)
    
class CVXPY_diff(torch.autograd.Function):

    @staticmethod
    def forward(A_in : Tensor,
                b_in : Tensor,
                A_eq : Tensor,
                b_eq : Tensor,
                q : Tensor) -> Tensor:

        # Solve the problem in a differential fashion
        res, lamda, nu = MSDProblemInstance.solve_with_cvxpy_diff(q, A_in, b_in, A_eq, b_eq)

        res = torch.from_numpy(res).unsqueeze(0)
        lamda = torch.from_numpy(lamda).unsqueeze(0)
        nu =  torch.from_numpy(nu).unsqueeze(0)

        return res.T, lamda, nu
    
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        """
        Method to manage the `ctx` modification.
        `inputs` is the tuple of arguments passed to the `forward` method.
        `output` is the output of `forward`.
        """
        A_in, b_in, A_eq, _, _ = inputs
        res, lamda, nu = outputs
        ctx.save_for_backward(A_in, b_in, A_eq, res, lamda, nu)

    @staticmethod
    def backward(ctx, grad_res, *_):
        
        # Unpack saved tensors
        A_in, b_in, A_eq, res, lamda, nu = ctx.saved_tensors

        # Convert res, lamda, nu to float-type tensor
        res = Tensor.float(res)
        lamda = Tensor.float(lamda)
        nu = Tensor.float(nu)
        
        # In order to calculate gradients, first solve a linear system for retrieveing dz, dlamda and dnu
        diff_matrix_size = A_in.size()[1] + A_in.size()[0] + A_eq.size()[0]
        diff_matrix = torch.zeros(diff_matrix_size, diff_matrix_size)

        # Construct (1, 0) block
        diff_matrix[A_in.size()[1]:A_in.size()[1] + A_in.size()[0], :A_in.size()[1]] = A_in
        # Construct (2, 0) block
        diff_matrix[A_in.size()[1] + A_in.size()[0]:, :A_eq.size()[1]] = A_eq
        # Construct (0, 1) block
        top_block = matmul(A_in.T, Tensor.float(diag(lamda.squeeze())))
        diff_matrix[:top_block.size()[0], A_in.size()[1]:A_in.size()[1] + top_block.size()[1]] = top_block
        # Construct (0, 2) block
        diff_matrix[:A_eq.size()[1], A_in.size()[1] + top_block.size()[1]:] = A_eq.T
        # Construct (1, 1) block
        central_block = diag(((matmul(A_in, res) - b_in.unsqueeze(1))).squeeze())
        diff_matrix[top_block.size()[0]:top_block.size()[0] + central_block.size()[0], 
                    A_in.size()[1]:A_in.size()[1] + central_block.size()[1]] = central_block
        
        # Take the opposite of the differential matrix
        diff_matrix = -diff_matrix

        # Invert the matrix. First make it invertible by adding a small value to its diagonal
        diff_matrix = diff_matrix + torch.eye(diff_matrix.size()[0])*1e-2
        diff_matrix = torch.linalg.inv(diff_matrix)

        # Construct vector for linear system
        diff_vector_size = A_in.size()[1] + A_in.size()[0] + A_eq.size()[0]
        diff_vector = torch.zeros(diff_vector_size).unsqueeze(1)
        
        # Construct (0, 0) entry for diff_vector
        diff_vector[:grad_res.size()[0], :] = grad_res

        # Solve linear problem for differential matrix 
        differentials = Tensor(np.linalg.solve(diff_matrix, diff_vector))

        dz = differentials[:res.size()[0]]
        
        dlamda = differentials[res.size()[0]:res.size()[0] + lamda.size()[1]]
        
        dnu = differentials[-nu.size()[0]:].squeeze()
        dnu = Tensor([dnu]).unsqueeze(0)

        # Compute gradients
        grad_q = dz
        grad_A_in = matmul(diag(lamda.squeeze()), matmul(dlamda, res.T)) + matmul(lamda.T, dz.T)
        grad_b_in = matmul(-diag(lamda.squeeze()), dlamda).squeeze()
        grad_A_eq = matmul(dnu.T, res.T) + matmul(nu, dz.T)
        grad_b_eq = -dnu
        
        # Return as many gradients as there are inputs to the forward method.
        # Arguments that do not require a gradient need to have a corresponding None value
        return grad_A_in, grad_b_in, grad_A_eq, grad_b_eq, grad_q

# Run model. 
# First regenerate dataset by running "python3 problem1_cvxpy_diff.py -regen"
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

    # Employ the custom-defined function
    function = CVXPY_diff.apply

    p_pred, _, _ = function(A_in, b_in, A_eq, b_eq, q)
    
    loss = torch.mean(torch.square(p - p_pred))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # Take loss at the last iteration and print to file
    if n_iter == 99:
        dir_path = "/Users/marco/Documents/Repos/differentiable_optimization/data_analysis"
        with open(os.path.join(dir_path, "losses_cvxpy_diff.txt"), "a") as file:
            file.write(f"{loss}\n")
    
# Plot loss function evolution
plt.plot(losses)
plt.title('LP problem instance')
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.show()
