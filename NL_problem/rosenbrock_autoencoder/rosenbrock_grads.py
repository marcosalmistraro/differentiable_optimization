from sympy import *

x = Symbol('x')
y = Symbol('y')
alpha = Symbol('alpha')
bet = Symbol('bet')

f = alpha * (1 - x)**2 + bet * 100 * (y - x**2)**2
c1 = (x - 1)**3 - y + 1
c2 = x + y -2

grad_f = [f.diff(x) for x in [x, y]]
grad_c1 = [c1.diff(x) for x in [x, y]]
grad_c2 = [c2.diff(x) for x in [x, y]]
print(grad_c1)
print(grad_c2)
print(f.diff(x).diff(y))