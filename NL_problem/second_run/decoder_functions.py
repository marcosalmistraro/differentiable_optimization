import os
import sys

from torch import Tensor

# The DiffCP implementation is shared across all runs and lives in the
# parent NL_problem directory. Make it importable when this module is used
# from within the run folder.
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from diffcp import DiffCP  # noqa: E402  (re-exported for use as decoder_functions.DiffCP)

def rosenbrock(x: Tensor, param: Tensor) -> Tensor:
    """
    Specifies an expression for the Rosenbrock function parameterized
    by tensors `a` and `b` and evaluated at point `x`.
    """
    a = param[0]
    b = param[1]
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

def ineq_constraint_1(x: Tensor, param: Tensor) -> Tensor:
    """
    Specifies an expression for an inequality constraint to be applied to
    the non-linear constrained task in the decoder stage of the AE. It takes
    tensor `x` as input argument and `param` as the tensor specifying the parameters
    for the inequality.
    """
    _ = param[0]
    _ = param[1]
    return x[1] - 1 - (x[0] - 1)**3

def ineq_constraint_2(x: Tensor, param: Tensor) -> Tensor:
    """
    Specifies an expression for an inequality constraint to be applied to
    the non-linear constrained task in the decoder stage of the AE. It takes
    tensor `x` as input argument and `param` as the tensor specifying the parameters
    for the inequality.
    """
    c = param[0]
    _ = param[1]
    return 2 - (x[0] + x[1])
