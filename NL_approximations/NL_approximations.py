from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch import Tensor, matmul

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

    # Construct the q coefficient vector
    q1 = grad
    q2 = -0.5*(matmul(hessian, x0))
    q3 = -0.5*matmul(x0, matmul(hessian, x0))
    q = q1 + q2 + q3

    return Q, q

def test():
    g = calculate_gradient(Tensor([1, 2, 40]), lambda x : 1000*x[0]**x[1]*x[2])
    print(g)

test()