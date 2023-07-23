from NL_approximations.NL_approximations import calculate_gradient, calculate_hessian, quadratic_approximation, linear_approximaton
import torch

x0 = torch.Tensor([1, 3, 32])
fun = lambda x : x[0]**x[1] + x[2]

g = calculate_gradient(x0, fun)
print(g)