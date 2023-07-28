import torch

a = torch.rand(2,requires_grad=True)

def fun1(x):
    return x*x

def fun2(x):
    return x[0]**2 + x[1]

fun3 = lambda x : (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

print(fun2(a))

g = torch.autograd.grad(fun3(a), inputs=a, create_graph=True)
print(g)
print(fun2(a).requires_grad)