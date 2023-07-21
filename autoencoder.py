import numpy as np
import random
import torch
import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from torch import diag, Tensor, matmul, nn, optim
from torch.nn import Module

from NLOpt.NL_problem import quadratic_approximation, linear_approximaton


class AE(Module):

    # TODO fix 'input_shape' kwarg
    def __init__(self, 
                 decoder_function, 
                 decoder_objective_function, 
                 decoder_ineq_constraints, 
                 decoder_eq_constraints,
                 **kwargs):
        super().__init__()

        # Define encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=kwargs['input_shape'], out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=2)
        )

        # Define decoder through the decoder_function argument, which subclasses
        # the torch.autograd.Function class, allowing for custom forward and
        # backward passes
        self.decoder = decoder_function(input_features=2, output_features=2)

        # Define constrained optimization problem implemented in the decoder
        self.decoder_objective_function = decoder_objective_function
        self.decoder_ineq_constraints = decoder_ineq_constraints
        self.decoder_eq_constraints = decoder_eq_constraints


    def forward(self, features):
        encoded = self.encoder(features)
        # The decoder employs a custom-defined function.
        # The original input is passed as argument in order to provide
        # an initial solution to the NLP system that is solved to retrieve
        # the reconstructed vector
        decoded = self.decoder(features, encoded)
        Q, q, A_in, b_in, A_eq, b_eq = self.convert(encoded)
        return decoded
    

class Decode_diff(torch.autograd.Function):

    @staticmethod
    def convert(x0, 
                decoder_objective_function, 
                decoder_ineq_constraints, 
                decoder_eq_constraints):
        # Compute the A, b, G, h, Q, q parameters for the OptNet representation
        # of the constrained optimization problem for the decoder
        Q, q = quadratic_approximation(x0, decoder_objective_function)

        # Check whether inequality constraints are specified by the inner optimization task
        # If so, form G and h vectors for each single constraint of this type. 
        # First initialize empty list of (G, h) tuples which stays empty in case
        # of no constraints
        ineq_constraints = []
        if len(decoder_ineq_constraints)!=0:
            for ineq_constraint in decoder_ineq_constraints:
                Gi, hi = linear_approximaton(x0, ineq_constraint)
                ineq_constraints.append((Gi, hi))
        # If not, assign an empty matrix and vector to express the lack of constraints
        else:
            G = torch.zeros(1, x0.size()[0])
            h = torch.zeros(x0.size()[0])

        # Perform the same check on equality constraints
        eq_constraints = []
        if len(decoder_eq_constraints)!=0:
            for eq_constraint in decoder_eq_constraints:
                Ai, bi = linear_approximaton(x0, eq_constraint)
                eq_constraints.append((Ai, bi))
        else:
            A = torch.zeros(1, x0.size()[0])
            b = torch.zeros(x0.size()[0])
            
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

            A = torch.block_diag(Ais)
            b = torch.cat(bis)

        # Retrieve the Lagrangian multipliers lamda, nu for the same problem.
        # This can be done by solving a two-equation system obtained from the
        # KKT conditions of the QP formulation.
        
        # First create the tensor that will store the optimal values for lamda and nu
        multipliers = Tensor(A.size()[0] + G.size()[0])

        # Encode MAG matrix for the first equation
        MAG = torch.zeros(A.size()[1] + G.size()[1], A.size()[0] + G.size()[0])
        MAG[:A.size()[1], :A.size()[0]] = A.T
        MAG[A.size()[1]:, A.size()[1]:] = G.T

        # TODO clarify how to wrtite D(lamda)

        # Solve the linear system
        lamda = multipliers[]
        nu = multipliers[]

        return Q, q, A_in, b_in, A_eq, b_eq
    
    @staticmethod
    def forward(encoded, 
                x0,
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
            ineq_cons = list({'type':'ineq', 'fun':item} for item in decoder_ineq_constraints)
        # Otherwise, assign an empty list to cons
        else:
            ineq_cons = []

        # Do the same for equality constraints
        if len(decoder_eq_constraints)!=0:
            eq_cons = list({'type':'eq', 'fun':item} for item in decoder_eq_constraints)
        else:
            eq_cons = []

        # Unify the two lists and convert into a tuple
        cons = tuple(ineq_cons.extend(eq_cons))

        # Exploit initial solution to reconstruct the input features
        reconstructed = minimize(fun, x0=x0, cons=cons)

        # Obtain parameters for OptNet approximtion of the problem
        Q, q, A_in, b_in, A_eq, b_eq, lamda, nu = Decode_diff.convert(x0, decoder_objective_function, decoder_ineq_constraints, decoder_eq_constraints)

        # Return decoded version of the input features
        return reconstructed, Q, q, A_in, b_in, A_eq, b_eq, lamda, nu
    
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        encoded, _, _, _, _, _, _ = inputs
        _, Q, q, A_in, b_in, A_eq, b_eq, lamda, nu = outputs
        ctx.save_for_backward(encoded, Q, q, A_in, b_in, A_eq, b_eq, lamda, nu)

    @staticmethod
    def backward(ctx, grad_reconstructed, grad_Q, grad_q, grad_A_in, grad_b_in, grad_A_eq, grad_b_eq, *_):
        # Unpack saved tensors
        encoded, Q, q, A_in, b_in, A_eq, res, lamda, nu = ctx.saved_tensors

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
        diff_vector[:grad_reconstructed.size()[0], :] = grad_reconstructed

        # Solve linear problem for differential matrix 
        differentials = Tensor(np.linalg.solve(diff_matrix, diff_vector))

        dz = differentials[:res.size()[0]]
        
        dlamda = differentials[res.size()[0]:res.size()[0] + lamda.size()[1]]
        
        dnu = differentials[-nu.size()[0]:].squeeze()
        dnu = Tensor([dnu]).unsqueeze(0)

        # Compute gradients
        grad_Q = 0.5(matmul(dz.T, res) + matmul(res, dz.T))
        grad_q = dz
        grad_A_in = matmul(diag(lamda.squeeze()), matmul(dlamda, res.T)) + matmul(lamda.T, dz.T)
        grad_b_in = matmul(-diag(lamda.squeeze()), dlamda).squeeze()
        grad_A_eq = matmul(dnu.T, res.T) + matmul(nu, dz.T)
        grad_b_eq = -dnu
        
        # Return as many gradients as there are inputs to the forward method.
        # Arguments that do not require a gradient need to have a corresponding None value
        # return grad_A_in, grad_b_in, grad_A_eq, grad_b_eq, grad_q

        return _, grad_Q, grad_q, grad_A_in, grad_b_in, grad_A_eq, grad_b_eq

# Employ the custom-defined function 
decoder_function = Decode_diff.apply

# Create model from the custom-defined class
model = AE(decoder_function, 
           decoder_objective_function = lambda x : (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2,
           decoder_ineq_constraints = [lambda x: x[1] - 1 - (x[0] - 1)**3,
                                       lambda x: 2 - x[0] - x[1]],
           decoder_eq_constraints = [],
           input_shape=2)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define criterion for weight update
criterion = nn.MSELoss()

# Create data set from R^2 by sampling uniformly without repetition
# First set a seed for the pseudo-random generator
random.seed(42)
# Create data set
dataset = []
for i in range(10):
    sample = (random.uniform(1, 100), random.uniform(1, 100))
    if sample not in dataset:
        dataset.append(sample)
        i += 1
    else:
        continue

print(dataset)
print(len(dataset))

# Split data into sets for training and testing
train_set, test_set = train_test_split(dataset, test_size=7, train_size=3)

# Train the AE
losses = []
for n_iter in tqdm.tqdm(range(100)):

    # Take a data point from train_set
    input_features = pass

    # Zero-out gradients to avoid accumulation
    optimizer.zero_grad()

    # 

    # Compute reconstruction through the forward pass
    output_features = model.forward(input_features,
                                    model.decoder_objective_function, 
                                    model.decoder_ineq_constraints, 
                                    model.decoder_eq_constraints)

    # Compute the training loss for the considered data instance
    train_loss = Tensor(criterion(output_features, input_features))

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