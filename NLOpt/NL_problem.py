# Rosenbrock function constrained with a cubic line

import torch
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch import Tensor, matmul

def constraint_1(input : Tensor):
    return input[0] + input[1] -2

def constraint_2(input : Tensor):
    return (input[0] - 1)**3 - input[1]**2 + 1

def calculate_gradient(input : Tensor, function):
    """
    Calculates the gradient of a function passed as argument w.r.t.
    a specified Tensor.
    """
    coordinates = Variable(Tensor(input), requires_grad=True)
    z = function(coordinates)
    z.backward()
    return(coordinates.grad)

def calculate_hessian(input : Tensor, function):
    """
    Calculates the hessian of a function passed as argument w.r.t.
    a specified Tensor.
    """
    return hessian(function, input)

def linear_approximaton(x0 : Tensor, function):
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
    print(grad.size())
    print(hessian)

    # Construct the q coefficient matrix
    q = torch.Tensor((grad.unsqueeze(0).size()[0])*3, 
                     grad.unsqueeze(0).size()[1])
    

    q[0, :] = grad
    # Exploit the property xy = y.Tx for generic vectors
    q[1, :] = -matmul(hessian, x0).T
    q[2, :] = -matmul(x0, hessian)
    print(q)

    #return Q, q

def test():
    x0 = Tensor([1, 2])
    quadratic_approximation(x0, constraint_2)

test()