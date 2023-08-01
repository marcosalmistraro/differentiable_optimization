import sys

import numpy as np
import random
import torch
import tqdm
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from torch import diag, Tensor, matmul, nn, optim
from torch.nn import Module

from rosenbrock_NL_approximations import *

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
    
    def forward(self, input, eps):
        encoded = self.encoder(input)
        decoded, Q, q, G, h, A, b, lamda, nu = self.decoder(input, encoded, eps)
        return encoded, decoded, Q, q, G, h, A, b, lamda, nu
    
    def convert(self, encoded, decoded):

        # TODO decoded does not require grad?
        
        # Extract quadratic approximation for the objective function
        Q, q = quad_approximation_objective(decoded, encoded)

        # Compute linear approximation for the first constraint
        G1, h1 = lin_approximation_cons_1(decoded)

        # Do the same for the second constraint
        G2, h2 = lin_approximation_cons_2(decoded)

        # Form G matrix and h vector for OptNet problem formulation
        G = torch.stack((G1, G2), dim=-1)
        h = torch.cat((h1.unsqueeze(0), h2.unsqueeze(0)), 0)

        # Equally form the A matrix and b vector for equality constraints
        A = torch.zeros(1, 2)
        b = torch.zeros(1)

        # Retrieve the Lagrangian multipliers lamda, nu for the same problem.
        # This can be done by solving a two-equation system obtained from the
        # KKT conditions of the QP formulation.

        # Encode MAG matrix for the first equation by combining the A.T and G.T blocks
        MAG = torch.cat((A.T, G.T), dim=-1)

        # Equally form the vector term for the first equation
        vector_temp = -matmul(Q, decoded) - q
        
        # Encode the matrix term for the second equation
        D_temp = diag(matmul(G, decoded) - h)
        
        # Expand it to account for the nu solution component
        D = torch.cat((torch.zeros(D_temp.size()[0], A.size()[0]), D_temp), -1)

        # Form the matrix term for the linear solver
        matrix_term = torch.cat((MAG, D), 0)

        # Equally form the vector term for the solver
        vector_term = torch.cat((vector_temp, torch.zeros(matrix_term.size()[0] - vector_temp.size()[0])), -1)

        # Solve the linear system to obain Lagrangian multipliers
        multipliers = torch.linalg.lstsq(matrix_term, vector_term, rcond=None)[0]

        # Extract the Lagrangian multipliers from the obtained solution
        nu = multipliers[:A.size()[0]]
        lamda = multipliers[A.size()[0]:]

        return Q, q, G, h, A, b, lamda, nu
    
    def backward_convert(Q : Tensor, q : Tensor, encoded : Tensor):
        return backward_quad_approximation(Q, q, encoded)


class Decode_diff(torch.autograd.Function):
    
    @staticmethod
    def forward(input, encoded, _):

        # Traverse the decoder by computing solution to the non-linear system, given the
        # parameters obtained at the encoder's output

        # Built dictionary for different constraints. No equality constraints are present
        cons = [{'type': 'ineq',
                'fun': lambda x : -(x[0] - 1)**3 + x[1] - 1,
                'jac': None},
                {'type': 'ineq',
                'fun': lambda x : -x[0] -x[1] + 2,
                'jac': None}]

        # Exploit initial solution to reconstruct the input features.
        # First cast the output of decoder_objective_function as float to make it
        # compliant with the form requested by SLSQP
        decoder_objective_function = lambda x : float(model.decoder_objective_function(x, encoded))
        decoded = minimize(fun = decoder_objective_function, 
                           x0=input,
                           method='SLSQP', 
                           jac=None,
                           constraints=cons)
        
        # Return resulting solution
        print(decoded.x)

        decoded = torch.tensor(decoded.x)
        Q, q, G, h, A, b, lamda, nu = model.convert(encoded, decoded)

        return decoded, Q, q, G, h, A, b, lamda, nu

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        input, encoded, eps = inputs
        decoded, Q, q, G, h, A, b, lamda, nu = outputs
        ctx.save_for_backward(input, encoded, eps, decoded, Q, q, G, h, A, b, lamda, nu)

    @staticmethod
    def backward(ctx, grad_decoded, *_):
        # Unpack saved tensors 
        # TODO change names
        input, encoded, eps, decoded, Q, q, G, h, A, b, lamda, nu = ctx.saved_tensors

        # In order to calculate gradients, first solve a linear system for retrieveing dz, dlamda and dnu
        diff_matrix_size = G.size()[1] + G.size()[0] + A.size()[0]
   
        diff_matrix = torch.zeros(diff_matrix_size, diff_matrix_size)
        # Construct (0, 0) block
        diff_matrix[:Q.size()[0], :Q.size()[1]] = Q
        # Construct (1, 0) block
        diff_matrix[G.size()[1]:G.size()[1] + G.size()[0], :G.size()[1]] = G
        # Construct (2, 0) block
        diff_matrix[G.size()[1] + G.size()[0]:, :A.size()[1]] = A
        # Construct (0, 1) block
        top_block = matmul(G.T, (diag(lamda.squeeze())))
        diff_matrix[:top_block.size()[0], G.size()[1]:G.size()[1] + top_block.size()[1]] = top_block
        # Construct (0, 2) block
        diff_matrix[:A.size()[1], G.size()[1] + top_block.size()[1]:] = A.T
        # Construct (1, 1) block
        central_block = diag(((matmul(G, decoded.double()) - h.unsqueeze(1))).squeeze())
        diff_matrix[top_block.size()[0]:top_block.size()[0] + central_block.unsqueeze(0).size()[0], 
                    G.size()[1]:G.size()[1] + central_block.unsqueeze(0).size()[1]] = central_block

        # Take the opposite of the differential matrix
        diff_matrix = -diff_matrix

        # TODO Invert the matrix. First make it invertible by adding a small value to its diagonal.
        # This value can be specified through the define_eps method of the Decode_diff class
        diff_matrix = diff_matrix + torch.eye(diff_matrix.size()[0])*eps
        diff_matrix = torch.linalg.inv(diff_matrix)

        # Construct vector for linear system
        diff_vector_size = G.size()[1] + G.size()[0] + A.size()[0]
        diff_vector = torch.zeros(diff_vector_size).unsqueeze(1)
        
        # Construct (0, 0) entry for diff_vector
        diff_vector[:grad_decoded.size()[0], :] = grad_decoded.unsqueeze(-1)

        # Solve linear problem for differential matrix 
        differentials = Tensor(np.linalg.solve(diff_matrix, diff_vector))

        dz = differentials[:decoded.size()[0]]

        # dlamda = differentials[encoded.size()[0]:encoded.size()[0] + lamda.size()[1]]
        
        dnu = differentials[-nu.size()[0]:].squeeze()
        dnu = Tensor([dnu]).unsqueeze(0)

        # Compute gradients
        grad_Q = 0.5*(matmul(dz, (encoded.unsqueeze(0))) + matmul(encoded.unsqueeze(-1), dz.T))
        grad_q = dz
        
        # grad_G = matmul(diag(lamda.squeeze()), matmul(dlamda, res.T)) + matmul(lamda.T, dz.T)
        # grad_h = matmul(-diag(lamda.squeeze()), dlamda).squeeze()
        # grad_A = matmul(dnu.T, output.T) + matmul(nu, dz.T)
        # grad_b_eq = -dnu
        
        # Return as many gradients as there are inputs to the forward method.
        # Arguments that do not require a gradient need to have a corresponding None value
        grad_encoded = AE.backward_convert(grad_Q, grad_q, encoded)

        return grad_encoded, None, None
    
    @staticmethod
    def define_eps(eps : float = 1e-2):
        return torch.Tensor([eps])

# Employ the custom-defined function 
decoder_function = Decode_diff.apply

# Create model from the custom-defined class
model = AE(decoder_function, 
           decoder_objective_function = lambda x, encoded : encoded[0]*(1 - x[0])**2 + encoded[1]*100*(x[1] - x[0]**2)**2,
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

# Train the AE
losses = []
idx_train = np.arange(len(dataset))

for n_iter in tqdm.tqdm(range(100)):

    # Take a data point from train_set
    i = idx_train[n_iter % len(idx_train)]
    input = torch.tensor((dataset[i]), dtype=torch.float64, requires_grad=True)

    # Zero-out gradients to avoid accumulation
    optimizer.zero_grad()

    # Define eps value for inverting the differential matrix during the backward pass
    eps = Decode_diff.define_eps()

    # First compute the encoded representation at the output of the NN
    encoded, decoded, Q, q, G, h, A, b, lamda, nu = model.forward(input.float(), eps)

    # Compute the training loss for the considered data instance
    train_loss = Tensor(criterion(input, decoded))
    print(train_loss)

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