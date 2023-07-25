from torch.autograd import grad
from torch.autograd.functional import hessian
from torch import Tensor, matmul
import torch

def calculate_gradient(x0 : Tensor, function):
    """
    Calculates the gradient of a function passed as argument w.r.t.
    a specified Tensor.
    """
    y = function(x0)
    x0_grad = torch.autograd.grad(y, x0)
    return x0_grad[0]

def calculate_hessian(x0 : Tensor, function):
    """
    Calculates the hessian of a function passed as argument w.r.t.
    a specified Tensor.
    """
    return hessian(function, x0)

def linear_approximation(x0 : Tensor, function):
    """
    Calculates the linear approximation of a given function passed
    as argument w.r.t. a specified Tensor. It outputs the parameters 
    that are used to express the approximation as a linear system.
    """
    grad = calculate_gradient(x0, function)
    A = grad
    b = function(x0) + matmul(grad, x0)
    return A, b

def quadratic_approximation(x0 : Tensor, function):
    """
    Calculates the quadratic approximation of a given function passed
    as argument w.r.t. a specified Tensor. It outputs the parameters 
    that are used to express the approximation according to the canonical QP form.
    """
    grad = calculate_gradient(x0, function)
    hessian = calculate_hessian(x0, function)
    Q = hessian

    # Construct the q coefficient vector
    q1 = grad
    q2 = -0.5*(matmul(hessian, x0))
    q3 = -0.5*matmul(x0, matmul(hessian, x0))
    q = q1 + q2 + q3

    return Q, q