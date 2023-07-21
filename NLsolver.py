from scipy.optimize import minimize
import torch

x0 = torch.rand(2)

fun = lambda x : (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

cons = ({'type': 'ineq', 'fun': lambda x: x[1] - 1 - (x[0] - 1)**3},
        {'type': 'ineq', 'fun': lambda x: 2 - x[0] - x[1]})

bounds = ((None, None), (None, None))

def test():
    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons)
    print(res)
    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons)
    print(res)
    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons)
    print(res)

test()
