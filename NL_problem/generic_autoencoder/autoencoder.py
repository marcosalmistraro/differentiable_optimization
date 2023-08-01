import sys

import numpy as np
import random
import torch
import tqdm
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from torch import diag, Tensor, matmul, nn, optim
from torch.autograd import Variable
from torch.nn import Module

from NL_approximations.generic_NL_approximations import quadratic_approximation, linear_approximation, calculate_gradient, calculate_hessian



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
    
    def forward_input(self, input):
        encoded = self.encoder(input)
        return encoded
    
    def forward_encoded(self, input, encoded, eps):
        # The decoder employs a custom-defined function.
        # The original input is passed as argument in order to provide
        # an initial solution to the NLP system that is solved to retrieve
        # the reconstructed vector
        decoded = self.decoder(input, encoded, eps)
        return decoded
    
    def convert(self, encoded, decoded):

        decoder_objective_function = lambda x : model.decoder_objective_function(x, encoded)

        # Extract linear and quadratic approximations
        Q, q = quadratic_approximation(decoded, decoder_objective_function)
        Q.requires_grad=True
        
        # Check whether inequality constraints are specified by the inner optimization task
        # If so, form G and h vectors for each single constraint of this type. 
        # First initialize empty list of (G, h) tuples which stays empty in case
        # of no constraints
        ineq_constraints = []
        if len(model.decoder_ineq_constraints)!=0:
            for ineq_constraint in model.decoder_ineq_constraints:
                Gi, hi = linear_approximation(encoded, ineq_constraint)
                ineq_constraints.append((Gi, hi))
        # If not, assign an empty matrix and vector to express the lack of constraints
        else:
            G = torch.tensor(torch.zeros(1, decoded.size()[0]), requires_grad=True)
            h = torch.tensor(torch.zeros(decoded.size()[0]), requires_grad=True)

        # Perform the same check on equality constraints
        eq_constraints = []
        if len(model.decoder_eq_constraints)!=0:
            for eq_constraint in model.decoder_eq_constraints:
                Ai, bi = linear_approximation(decoded, eq_constraint)
                eq_constraints.append((Ai, bi))
        else:
            A = torch.tensor(torch.zeros(1, decoded.size()[0]), requires_grad=True)
            b = torch.tensor(torch.zeros(decoded.size()[0]), requires_grad=True)
        
        #TODO express these terms as cat
        # Combine the retrieved inequality approximations into one unique matrix and one unique vector
        if len(ineq_constraints)!=0:
            Gis = tuple(Gi for (Gi, _) in ineq_constraints)
            his = tuple(hi for (_, hi) in ineq_constraints)

            G = torch.tensor(torch.cat(tuple(Gi.unsqueeze(dim=-1) for Gi in Gis), dim=-1), requires_grad=True)
            h = torch.tensor(torch.cat(tuple(hi.unsqueeze(dim=-1) for hi in his)), requires_grad=True)        

        # Do the same for equality constraints
        if len(eq_constraints)!=0:
            Ais = tuple(Ai for (Ai, _) in eq_constraints)
            bis = tuple(bi for (_, bi) in eq_constraints)

            A = torch.tensor(torch.cat(tuple(Ai.unsqueeze(dim=-1) for Ai in Ais), dim=-1), requires_grad=True)
            b = torch.tensor(torch.cat(tuple(bi.unsqueeze(dim=-1) for bi in bis)), requires_grad=True)

        #TODO check differentiability formation G, H, A, b
        # TODO check unsqueeze() maybe it's dim=0 since they get transposed later

        # Retrieve the Lagrangian multipliers lamda, nu for the same problem.
        # This can be done by solving a two-equation system obtained from the
        # KKT conditions of the QP formulation.

        # Encode MAG matrix for the first equation by combining the A.T and G.T blocks
        MAG = torch.cat((A.T, G.T), dim=-1)

        # Equally form the vector term for the first equation
        vector_temp = -matmul(Q, decoded) - q
        
        # Encode the matrix term for the second equation
        D_temp = diag(matmul(G, decoded.float()) - h)
        
        # Expand it to account for the nu solution component
        D = torch.cat((torch.zeros(D_temp.size()[0], A.size()[0]), D_temp), -1)

        # Form the matrix term for the linear solver
        matrix_term = torch.cat((MAG, D), 0)

        # Equally form the vector term for the solver
        vector_term = torch.cat((vector_temp, torch.zeros(matrix_term.size()[0] - vector_temp.size()[0])), -1).float()
        
        # Solve the linear system to obain Lagrangian multipliers
        multipliers = torch.linalg.lstsq(matrix_term, vector_term, rcond=None)[0]

        # Extract the Lagrangian multipliers from the obtained solution
        nu = multipliers[:A.size()[0]]
        lamda = multipliers[A.size()[0]:]

        optnet_parameters = Q, q, G, h, A, b, lamda, nu

        return optnet_parameters
    
    def backward_convert():
        pass
    

class Decode_diff(torch.autograd.Function):
    
    @staticmethod
    def forward(input, encoded, _):

        # Traverse the decoder by computing solution to the non-linear system, given the
        # parameters obtained at the encoder's output
        # TODO add items from encoded (alpha beta)

        # Check if inequalities constraints are present in the definition of the AE.
        # If so, add them to the list of constraints for the solver
        # Jacobians for the constrqints are computed by means of a Jacobian-vector product
        if len(model.decoder_ineq_constraints)!=0:
            ineq_cons = list({'type':'ineq', 
                              'fun':item,
                              'jac':None}
                              for item in model.decoder_ineq_constraints)
        # Otherwise, assign an empty list to cons
        else:
            ineq_cons = []

        # Do the same for equality constraints
        if len(model.decoder_eq_constraints)!=0:
            eq_cons = list({'type':'eq', 
                            'fun':item,
                            'jac':None} 
                            for item in model.decoder_eq_constraints)
        else:
            eq_cons = []

        # Unify the two lists and convert into a tuple
        cons = tuple(ineq_cons + eq_cons)
        print(cons)

        # Exploit initial solution to reconstruct the input features.
        # First cast the output of decoder_objective_function as float to make it
        # compliant with the form requested by SLSQP
        decoder_objective_function = lambda x : float(model.decoder_objective_function(x, encoded))
        # decoder_objective_function = lambda x : float(
        #     encoded[0]*(1 - x[0])**2 + encoded[1]*100*(x[1] - x[0]**2)**2)
        decoded = minimize(fun = decoder_objective_function, 
                           x0=input,
                           method='SLSQP', 
                           jac=None,
                           constraints=cons)
        
        # Return resulting solution
        print(decoded.x)

        decoded = torch.tensor(decoded.x)

        return decoded

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, encoded, eps = inputs
        decoded = output
        ctx.save_for_backward(input, encoded, eps, decoded)

    @staticmethod
    def backward(ctx, grad_decoded, *_):
        # Unpack saved tensors 
        # TODO change names
        input, encoded, eps, decoded = ctx.saved_tensors

        # Convert res, lamda, nu to float-type tensor
        encoded = Tensor.float(encoded)
        lamda = Tensor.float(lamda)
        nu = Tensor.float(nu)
        
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

        # TODO Invert the matrix. First make it invertible by adding a small value to its diagonal.
        # This value can be specified through the define_eps method of the Decode_diff class
        diff_matrix = diff_matrix + torch.eye(diff_matrix.size()[0])*eps
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

    # First compute the encoded representation at the output of the NN
    encoded = model.forward_input(input.float())

    # TODO find better way to express jacobians
    # Call the forward pass for the decoder stage. 
    # The emplyed SLSQP method does not require an explicit formulation
    # of the Jacobian of the objective function, which can be estimated
    # with a 2-point finite difference estimation otherwise. Same for
    # the constraints, which can use JVP.

    # Define eps value for inverting the differential matrix during the backward pass
    eps = Decode_diff.define_eps()
    decoded = model.forward_encoded(input, encoded, eps)
    sys.exit() 

    # Extract the OptNet parameters corresponding to the decoder formulation
    # and the encoded vector
    optnet_parameters = model.convert(encoded, decoded)

    # Compute the training loss for the considered data instance
    train_loss = Tensor(criterion(input, decoded))
    
    # Compute gradients by traversing the graph in a backward fashion
    # according to the custom-defined backward method
    train_loss.backward()
    sys.exit()

#     # Update parameters
#     optimizer.step()

#     # Append loss to the array of losses for the training phase
#     losses.append(train_loss.item())

# Display the evolution of the loss function over 
# the training phase
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.show()