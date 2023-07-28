import sys

import cvxpy as cp
import numpy as np
import random
import torch
import tqdm
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from torch import diag, Tensor, matmul, nn, optim
from torch.autograd import Variable
from torch.autograd.functional import jvp
from torch.nn import Module

from NL_approximations.NL_approximations import quadratic_approximation, linear_approximation

class AE(Module):

    def __init__(self, 
                 decoder_function, 
                 decoder_objective_function, 
                 decoder_ineq_constraints, 
                 decoder_eq_constraints):
        super().__init__()

        # Define encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=2)
        )

        # Define decoder through the decoder_function argument, which subclasses
        # the torch.autograd.Function class, allowing for custom forward and
        # backward passes
        self.decoder = decoder_function

        # Define constrained optimization problem implemented in the decoder
        self.decoder_objective_function = decoder_objective_function
        self.decoder_ineq_constraints = decoder_ineq_constraints
        self.decoder_eq_constraints = decoder_eq_constraints

    def forward(self, input):
        encoded = self.encoder(input)
        # The decoder employs a custom-defined function.
        # The original input is passed as argument in order to provide
        # an initial solution to the NLP system that is solved to retrieve
        # the reconstructed vector
        decoded = self.decoder(encoded, 
                               self.decoder_objective_function, 
                               self.decoder_ineq_constraints,
                               self.decoder_eq_constraints)
        return decoded
    

class Decode_diff(torch.autograd.Function):

    @staticmethod
    def convert(encoded, 
                decoder_objective_function, 
                decoder_ineq_constraints, 
                decoder_eq_constraints):
        
        # Compute the A, b, G, h, Q, q parameters for the OptNet representation
        # of the constrained optimization problem for the decoder

        Q, q = quadratic_approximation(encoded, decoder_objective_function)
        
        # Check whether inequality constraints are specified by the inner optimization task
        # If so, form G and h vectors for each single constraint of this type. 
        # First initialize empty list of (G, h) tuples which stays empty in case
        # of no constraints
        ineq_constraints = []
        if len(decoder_ineq_constraints)!=0:
            for ineq_constraint in decoder_ineq_constraints:
                Gi, hi = linear_approximation(encoded, ineq_constraint)
                ineq_constraints.append((Gi, hi))
        # If not, assign an empty matrix and vector to express the lack of constraints
        else:
            G = torch.zeros(1, encoded.size()[0])
            h = torch.zeros(encoded.size()[0])

        # Perform the same check on equality constraints
        eq_constraints = []
        if len(decoder_eq_constraints)!=0:
            for eq_constraint in decoder_eq_constraints:
                Ai, bi = linear_approximation(encoded, eq_constraint)
                eq_constraints.append((Ai, bi))
        else:
            A = torch.zeros(1, encoded.size()[0])
            b = torch.zeros(encoded.size()[0])
            
        # Combine the retrieved inequality approximations into one unique matrix and one unique vector
        if len(ineq_constraints)!=0:
            Gis = tuple(Gi for (Gi, _) in ineq_constraints)
            his = tuple(hi for (_, hi) in ineq_constraints)

            G = torch.block_diag(*Gis)
            h = torch.cat(his)
        
        # Do the same for equality constraints
        if len(eq_constraints)!=0:
            Ais = tuple(Ai for (Ai, _) in eq_constraints)
            bis = tuple(bi for (_, bi) in eq_constraints)

            A = torch.block_diag(*Ais)
            b = torch.cat(bis)

        # Retrieve the Lagrangian multipliers lamda, nu for the same problem.
        # This can be done by solving a two-equation system obtained from the
        # KKT conditions of the QP formulation.

        # Encode MAG matrix for the first equation by first creating the A0 and G0 blocks
        A0 = torch.cat((A.T, torch.zeros_like(G.T)), -1)
        G0 = torch.cat((torch.zeros_like(A.T), G.T), -1)
        # Combine them into MAG matrix to account for the first equation
        MAG = torch.cat((A0, G0), -1)

        # Encode the matrix term for the second equation
        D_temp = diag(matmul(G, encoded) - h)
        # Expand it to account for the nu solution component
        D = torch.cat((torch.zeros(D_temp.size()[0], A.size()[0])), -1)

        # Form the matrix term for the linear solver
        matrix_term = torch.cat((MAG, D), 0)

        # Equally form the vector term
        vector_temp = matmul(-Q, x) - q
        vector_term = torch.cat(vector_temp, torch.zeros(D.size()[0]))

        # Solve the linear system
        multipliers = Tensor(np.linalg.solve(matrix_term, vector_term))

        # Extract the Lagrangian multipliers from the obtained solution
        nu = multipliers[:A0.size()[1]]
        lamda = multipliers[A0.size()[1]:]
        
        A_in = G
        b_in = h
        A_eq = A
        b_eq = b

        return Q, q, A_in, b_in, A_eq, b_eq, lamda, nu
    
    @staticmethod
    def backward_convert(grad_Q, grad_q, grad_A_in, grad_b_in, grad_A_eq, grad_b_eq):

        # Solve the generally non-convex problem defined by the parameters passed as arguments
        x = cp.Variable(grad_Q.size()[0])
        prob = cp.Problem(cp.minimize((1/2) * cp.quad_form(x, grad_Q) + grad_q @ x),
            [grad_A_in @ x <= grad_b_in,
             grad_A_eq @ x == grad_b_eq])
        
        prob.solve()
        
        return x.value
    
    @staticmethod
    def forward(encoded,
                decoder_objective_function, 
                decoder_ineq_constraints,
                decoder_eq_constraints):

        # Traverse the decoder

        # First compute solution to the non-linear system, given the
        # parameters obtained at the encoder's output
        fun = decoder_objective_function # TODO add items from encoded (alpha beta)

        # Check if inequalities constraints are present in the definition of the AE.
        # If so, add them to the list of constraints for the solver
        if len(decoder_ineq_constraints)!=0:
            ineq_cons = list({'type':'ineq', 
                              'fun':item} for item in decoder_ineq_constraints)
        # Otherwise, assign an empty list to cons
        else:
            ineq_cons = []

        # Do the same for equality constraints
        if len(decoder_eq_constraints)!=0:
            eq_cons = list({'type':'eq', 'fun':item} for item in decoder_eq_constraints)
        else:
            eq_cons = []

        # Unify the two lists and convert into a tuple
        cons = tuple(ineq_cons + eq_cons)

        # Exploit initial solution to reconstruct the input features
        reconstructed = minimize(fun, x0=encoded, constraints=cons)

        # Obtain parameters for OptNet approximtion of the problem
        Q, q, A_in, b_in, A_eq, b_eq, lamda, nu = Decode_diff.convert(
            encoded, decoder_objective_function, decoder_ineq_constraints, decoder_eq_constraints)

        # Return decoded version of the input features
        return reconstructed, Q, q, A_in, b_in, A_eq, b_eq, lamda, nu
    
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        encoded, _, _, _, _, _, _ = inputs
        output, Q, q, A_in, b_in, A_eq, b_eq, lamda, nu = outputs
        ctx.save_for_backward(encoded, output, Q, q, A_in, b_in, A_eq, b_eq, lamda, nu)

    @staticmethod
    def backward(ctx, grad_encoded, grad_Q, grad_q, grad_A_in, grad_b_in, grad_A_eq, grad_b_eq, *_):
        # Unpack saved tensors
        encoded, output, Q, q, A_in, b_in, A_eq, b_eq, lamda, nu = ctx.saved_tensors

        # Convert res, lamda, nu to float-type tensor
        encoded = Tensor.float(encoded)
        lamda = Tensor.float(lamda)
        nu = Tensor.float(nu)

        # Zero-out gradients
        grad_Q = grad_q = grad_A_in = grad_b_in = grad_A_eq = grad_b_eq = None
        
        # In order to calculate gradients, first solve a linear system for retrieveing dz, dlamda and dnu
        diff_matrix_size = A_in.size()[1] + A_in.size()[0] + A_eq.size()[0]
        diff_matrix = torch.zeros(diff_matrix_size, diff_matrix_size)

        # Construct (0, 0) block
        diff_matrix[:Q.size()[0], :Q.size()[1]] = Q
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
        central_block = diag(((matmul(A_in, output) - b_in.unsqueeze(1))).squeeze())
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
        diff_vector[:grad_encoded.size()[0], :] = grad_encoded

        # Solve linear problem for differential matrix 
        differentials = Tensor(np.linalg.solve(diff_matrix, diff_vector))

        dz = differentials[:output.size()[0]]
        
        dlamda = differentials[output.size()[0]:output.size()[0] + lamda.size()[1]]
        
        dnu = differentials[-nu.size()[0]:].squeeze()
        dnu = Tensor([dnu]).unsqueeze(0)

        # Compute gradients
        grad_Q = 0.5*(matmul(dz.T, output) + matmul(output, dz.T))
        grad_q = dz
        grad_A_in = matmul(diag(lamda.squeeze()), matmul(dlamda, res.T)) + matmul(lamda.T, dz.T)
        grad_b_in = matmul(-diag(lamda.squeeze()), dlamda).squeeze()
        grad_A_eq = matmul(dnu.T, output.T) + matmul(nu, dz.T)
        grad_b_eq = -dnu
        
        # Return as many gradients as there are inputs to the forward method.
        # Arguments that do not require a gradient need to have a corresponding None value
        grad_encoded = Decode_diff.backward_convert(grad_Q, grad_A_in, grad_b_in, grad_A_eq, grad_b_eq, grad_q)

        return grad_encoded, None, None, None

# Employ the custom-defined function 
decoder_function = Decode_diff.apply

# Create model from the custom-defined class
model = AE(decoder_function, 
           decoder_objective_function = lambda x : (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2,
           decoder_ineq_constraints = [lambda x: x[1] - 1 - (x[0] - 1)**3,
                                       lambda x: 2 - x[0] - x[1]],
           decoder_eq_constraints = [])

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define criterion for weight update
criterion = nn.MSELoss()

# Create data set from R^2 by sampling uniformly without repetition
# First set a seed for the pseudo-random generator
random.seed(42)
# Create data set
dataset = []
for i in range(100):
    sample = tuple(random.uniform(0, 1) for _ in range(2))
    if sample not in dataset:
        dataset.append(sample)
        i += 1
    else:
        continue

# Split data into sets for training and testing
train_set, test_set = train_test_split(dataset, test_size=7, train_size=3)

# Train the AE
losses = []
idx_train = np.arange(len(train_set))

for n_iter in tqdm.tqdm(range(100)):

    # Take a data point from train_set
    i = idx_train[n_iter % len(idx_train)]
    input = torch.tensor((train_set[i]), dtype=torch.float64, requires_grad=True)

    # Zero-out gradients to avoid accumulation
    optimizer.zero_grad()

    # Compute reconstruction through the forward pass
    output = model.forward(input.float())

    # Compute the training loss for the considered data instance
    train_loss = Tensor(criterion(output, input))

    # Compute gradients by traversing the graph in a backward fashion
    # according to the custom-defined backward method
    train_loss.backward()

    # Update parameters
    optimizer.step()

    # Append loss to the array of losses for the training phase
    losses.append(train_loss.item())

# Display the evolution of the loss function over 
# the training phase
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.show()