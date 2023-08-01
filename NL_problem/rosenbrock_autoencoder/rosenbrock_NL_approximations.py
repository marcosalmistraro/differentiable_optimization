import math
from torch import Tensor, matmul
import torch

def gradient_objective(x0 : Tensor, coeff : Tensor):
    grad_1 = coeff[0]*(2*x0[0] - 2) - 400*coeff[1]*x0[0]*(-x0[0]**2 + x0[1])
    grad_2 = 100*coeff[1]*(-2*x0[0]**2 + 2*x0[1])
    grad = torch.cat((grad_1.unsqueeze(0), grad_2.unsqueeze(0)), -1)
    return grad

def hessian_objective(x0 : Tensor, coeff : Tensor):
    hess_1 = 2*coeff[0] + 800*coeff[1]*x0[0]**2 - 400*coeff[1]*(-x0[0]**2 + x0[1])
    hess_2 = -400*coeff[1]*x0[0]
    hess_3 = -400*coeff[1]*x0[0]
    hess_4 = 200*coeff[1]
    tmp_hess_1 = torch.cat((hess_1.unsqueeze(0), hess_2.unsqueeze(0)), -1)
    tmp_hess_2 = torch.cat((hess_3.unsqueeze(0), hess_4.unsqueeze(0)), -1)
    hess = torch.stack((tmp_hess_1, tmp_hess_2), -1)
    return hess

def quad_approximation_objective(x0 : Tensor, coeff : Tensor):
    grad = gradient_objective(x0, coeff)
    hessian = hessian_objective(x0, coeff)
    Q = hessian

    # Construct the q coefficient vector
    q1 = grad
    q2 = -0.5*(matmul(hessian, x0))
    q3 = -0.5*matmul(x0, matmul(hessian, x0))
    q = q1 + q2 + q3

    return Q, q

def gradient_cons_1(x0 : Tensor):
    grad_1 = 3*(x0[0] - 1)**2
    # Implement -1 entry in a differentiable manner
    grad_2 = -torch.div(x0, x0)[0]
    grad = torch.cat((grad_1.unsqueeze(0), grad_2.unsqueeze(0)), -1)
    return grad

def lin_approximation_cons_1(x0 : Tensor):
    grad = gradient_cons_1(x0)
    G = grad
    h = -matmul(grad, x0) + (x0[0] - 1)**3 - x0[1] + 1
    return G, h

def gradient_cons_2(x0 : Tensor):
    grad_1 = -torch.div(x0, x0)[0]
    grad_2 = -torch.div(x0, x0)[0]
    grad = torch.cat((grad_1.unsqueeze(0), grad_2.unsqueeze(0)), -1)
    return grad

def lin_approximation_cons_2(x0 : Tensor):
    grad = gradient_cons_2(x0)
    G = grad
    h = -matmul(grad, x0) + x0[0] + x0[1] - 2
    return G, h

def backward_quad_approximation(Q : Tensor, q : Tensor, decoded : Tensor):
    # Exploit the formulation for entry [0, 1] of the Q matrix, given decoded vector
    beta = -Q[0][1]/(400*decoded[0])
    # Plug the obtained value into the expression for the [0, 0] entry of the Q matrix
    alpha = (Q[0][0] - 800*beta*(decoded[0]**2) + 400*beta*(-decoded[0]**2 + decoded[1]))/2
    # Use trick to maintain differentiability of the obtained tensor
    return torch.cat((q.squeeze(1), alpha.unsqueeze(0), beta.unsqueeze(0)), -1)[:-2]


def test():
    pass

test()

