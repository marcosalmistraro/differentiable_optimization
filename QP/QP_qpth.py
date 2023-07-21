# Script to implement a qpth-bqsed solver.
# Application of the original OptNet formulation to the LP qnd QP cases
# 
# LP
# 1. Linear objective function
# 2. Linear constraints
#
# QP
# 1. Covex quadratic function
# 2. Linear constraints
#
# Treating LP as a special case of QP, employing qpth as a solver.

from qpth.qp import QPFunction
import torch
from torch import Tensor

# Allocate to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Memory allocation to {device}")

def qpth(q=Tensor(),
        **kwargs) -> Tensor:
    """
    The method returns the solution to the LP constrained optimization problem
    as output. It treats the linear instance of the problem as a special case of QP.
    If equality or inequality constraints are not specified it employs empty tensors 
    as default values. If the `x_gt0_constraint` keyword is set to `True` then 
    a corresponding constraint is added for the optimization variable. 
    """
    # Check if inequalities constraints have to be added.
    # If not, automatically plug empty tensors into QPFunction
    if 'A_in' in kwargs:
        A_in = kwargs['A_in']
        b_in = kwargs['b_in']
    else:
        A_in = torch.empty(1, q.size()[0])
        b_in = torch.empty(1)
    A_in.to(device)
    b_in.to(device)

    # Check if equalities constraints have to be added.
    # If not, automatically plug empty tensors into QPFunction
    if 'A_eq' in kwargs:
        A_eq = kwargs['A_eq']
        b_eq = kwargs['b_eq']
    else:
        A_eq = Tensor()
        b_eq = Tensor()
    A_eq.to(device)
    b_eq.to(device)

    # If Q is not specified, treat as linear case
    if 'Q' not in kwargs:
        # Set Q as null matrix; employ small epsilon value as a form of L2 regularization
        # in order to avoid numerical issues with qpth's solver.
        # Ensure that the matrix be SDP in order to make the problem convex
        Q = torch.randn(len(q), len(q))
        Q = Q @ Q.mT
        eps = kwargs['eps'] if 'eps' in kwargs else 1e-10
        Q *= eps
    else:
        Q = kwargs['Q']
    Q.to(device)

    # Incorporate the requirement for x >= 0 into the A matrix and b vector.
    # This can be done by adding a row to both thensors. 
    # Only do this if specified as a keyword argument
    if 'x_gt0_constraint' in kwargs:
        tensor_ones = torch.mul(-1, torch.ones(A_in.size(0)))
        A_in = torch.concat(A_in, tensor_ones, 0)
        b_in = torch.concat(b_in, torch.zeros(1), 1)

    # Apply QP Function from qpth
    return QPFunction(verbose=False)(Q, q, A_in, b_in, A_eq, b_eq)
    