# Script to implement a linear solver.
# Application of the original OptNet formulation to the LP case.
#
# 1. Linear objective function
# 2. Linear constraints
#
# Employing the linprog function from the scipy.optimize library.

from scipy.optimize import linprog
import torch
from torch import Tensor

# Allocate to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Memory allocation to {device}")

def LP_linprog(A_in=Tensor(), 
            b_in=Tensor(), 
            A_eq=Tensor(), 
            b_eq=Tensor(),
            coeff=Tensor(), 
            bounds = None) -> Tensor:
    """
    The method returns the solution to the LP constrained optimization problem
    as output. It employs the linprog method from the scipy.optimize package.
    `x_bounds` indicates intervals for each entry in the optimization vector.
    As per default, this is set to `None`. The `coeff` parameter represents the 
    coefficient vector for the optimization variable; if not specified this is set
    to a 1-D array of 1 entries. The implementation makes use of the `highs-ipm` method,
    corresponding to a HiGHS interior point solver.
    """

    # Consider case in which inequalities are absent 
    if A_in.size()[0] == 0:
        A_in = torch.zeros_like(A_eq)
        b_in = torch.zeros_like(b_eq)
        # Also account for the case where coefficients
        # are not specified
        if coeff.size()[0] == 0:
            coeff = torch.ones(A_eq.size()[1])
    
    # Do the same for equalities
    elif A_eq.size()[0] == 0:
        A_eq = torch.zeros_like(A_in)
        b_eq = torch.zeros_like(b_in)
        if coeff.size()[0] == 0:
            coeff = torch.ones(A_in.size()[1])
    
    # Create the coefficient vector in case both equalities and inequalities are present
    elif coeff.size()[0] == 0:
        coeff = torch.ones(A_in.size()[1])

    # Detach tensors and create NumPy arrays for the linprog method
    A_in = A_in.detach().numpy()
    b_in = b_in.detach().numpy()
    A_eq = A_eq.detach().numpy()
    b_eq = b_eq.detach().numpy()
    coeff = coeff.detach().numpy()

    res = linprog(c=coeff, A_ub=A_in, b_ub=b_in,
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ipm')
    
    return res
