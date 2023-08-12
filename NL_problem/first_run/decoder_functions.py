import numpy as np
import scipy.optimize
import torch

from torch import Tensor, tensor
from typing import Any, Callable, List, Tuple

USE_VJP_BASED_BACKWARD_IMPLEMENTATION = True

class DiffCP(torch.autograd.Function):
    """
    Defines the differentiable function for implementing non-linear constrained
    optimization problems into end-to-end learnable architectures, such as AEs.
    Contains custom-defined `forward` and `backward` passes, along with methods
    for calculating a quadratic approximation of the NLP objective function and
    linear approximations for the corresponding constraints.
    """

    @staticmethod
    def forward(ctx: Any,
                obj_param: Tensor, obj: Callable,
                eq_param: Tensor, eqs: List[Callable],
                ineq_param: Tensor, ineqs: List[Callable],
                input: Tensor) -> Tensor:
        """
        Computes the output of the forward pass for the AE's decoder.
        It takes as arguments the tensors containing the parameters to 
        be estimated with regards to the objective function, the inequality 
        and the equality constraints. The last `input` arguments is used as
        an initial solution estimate for the SLSQP solver. It returns a decoded
        version of the input tensor at the output of the decoder stage.
        """
        
        @torch.no_grad()
        def fun(x: np.ndarray) -> float:
            """
            Defines the objective function for the decoder.
            """
            x = tensor(x)
            return obj(x, obj_param).item()

        def jac(x: np.ndarray) -> np.ndarray:
            """
            Outputs the gradient of the objective function 
            w.r.t. a candidate solution `x` passed as argument.
            """
            x = tensor(x, requires_grad=True)
            f = lambda x_: obj(x_, obj_param)
            grad = torch.autograd.functional.jacobian(f, (x,), create_graph=True)[0]
            return grad.cpu().data.numpy()

        # Build dicts for the specified constraints. 
        # These are used in the call to scipy.optimize.minimize()
        eq_cons = list({'type':'eq', 'fun': lambda x : item(x, eq_param)} for item in eqs)
        ineq_cons = list({'type':'ineq', 'fun': lambda x: item(x, ineq_param)} for item in ineqs)
        cons = eq_cons + ineq_cons
        
        # Solve NLP to obtain the decoded representation of the input
        # at the output of the AE's decoder stage
        x0 = np.array(input)
        res = scipy.optimize.minimize(fun, x0, jac=jac, constraints=cons)
        x_star = Tensor(res.x)

        # Compute Lagrangian multipliers from the KKT conditions

        # First retrieve Q from quadratic approximation of the objective function
        Q, q = DiffCP.quadratic_approximation(x_star, lambda x: obj(x, obj_param))

        # Check whether inequality constraints are specified by the inner optimization task.
        # If so, form G and h vectors for each single constraint of this type. 
        # First initialize empty list of (G, h) tuples which stays empty in case
        # of no constraints
        ineq_cons_approx = []
        if len(ineq_cons)!=0:
            for cons in ineq_cons:
                Gi, hi = DiffCP.linear_approximation(x_star, cons['fun'])
                ineq_cons_approx.append((Gi, hi))

        # If not, assign an empty matrix and vector to express the lack of constraints
        else:
            G = torch.zeros(1, x_star.size()[0]).clone().detach().requires_grad_()
            h = torch.zeros(x_star.size()[0]).clone().detach().requires_grad_()

        # Perform the same check on equality constraints
        eq_cons_approx = []
        if len(eq_cons)!=0:
            for cons in eq_cons:
                Ai, bi = DiffCP.linear_approximation(x_star, cons['fun'])
                eq_cons_approx.append((Ai, bi))
        else:
            A = torch.zeros(1, x_star.size()[0]).clone().detach().requires_grad_()
            b = torch.zeros(x_star.size()[0]).clone().detach().requires_grad_()

        # Combine the retrieved inequality approximations into one unique matrix and one unique vector
        if len(ineq_cons_approx)!=0:
            Gis = tuple(Gi for (Gi, _) in ineq_cons_approx)
            his = tuple(hi for (_, hi) in ineq_cons_approx)

            G = torch.cat(tuple(Gi.unsqueeze(dim=-1) for Gi in Gis), dim=-1).clone().detach().requires_grad_()
            h = torch.cat(tuple(hi.unsqueeze(dim=-1) for hi in his)).clone().detach().requires_grad_()

        # Do the same for equality constraints
        if len(eq_cons_approx)!=0:
            Ais = tuple(Ai for (Ai, _) in eq_cons_approx)
            bis = tuple(bi for (_, bi) in eq_cons_approx)

            A = torch.cat(tuple(Ai.unsqueeze(dim=-1) for Ai in Ais), dim=-1).clone().detach().requires_grad_()
            b = torch.cat(tuple(bi.unsqueeze(dim=-1) for bi in bis)).clone().detach().requires_grad_()

        # Encode MAG matrix for the first system equation by combining the A.T and G.T blocks
        MAG = torch.cat((A.T, G.T), dim=-1)

        # Equally form the vector term for the first equation
        vector_temp = -torch.mv(Q, x_star) - q
        
        # Encode the matrix term for the second equation
        D_temp = torch.diag(torch.mv(G, x_star) - h)

        # Expand it to account for the nu solution component
        D = torch.cat((torch.zeros(MAG.size()[1] - D_temp.size()[0]), D_temp), -1).unsqueeze(0)

        # Form the matrix term for the linear solver
        matrix_term = torch.cat((MAG, D), 0)        

        # Equally form the vector term for the solver
        vector_term = torch.cat((vector_temp, torch.zeros(matrix_term.size()[0] - vector_temp.size()[0])), -1).float()
        
        # Solve the linear system to obtain Lagrangian multipliers
        multipliers = torch.linalg.lstsq(matrix_term, vector_term, rcond=None)[0]
        # Extract the Lagrangian multipliers from the obtained solution
        nu_star = Tensor(multipliers[:A.size()[0]])
        lambda_star = Tensor(multipliers[A.size()[0]:])

        # Save inputs and solution for backward pass
        ctx.save_for_backward(x_star, nu_star, lambda_star, G, h, A, b)
        ctx.obj_param = obj_param
        ctx.obj = obj
        ctx.eq_param = eq_param
        ctx.eqs = eqs
        ctx.ineq_param = ineq_param
        ctx.ineqs = ineqs

        return x_star

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None, 
                                                         Tensor, None, 
                                                         Tensor, None]:
        """
        Computes the backward pass for differentiable back-propagation
        of the gradient of the AE's decoded output. The method outputs as
        many objects as there are inputs to the forward pass, with corresponding
        `None` values for inputs not requiring a gradient. In this case, the lists
        `obj`, `eqs` and `ineqs` do not have a correspoding gradient, as they are just
        lists of functions for the NLP instance.
        """

        # Load inputs and solution
        x_star, _, lambda_star, G, h, A, _ = ctx.saved_tensors
        obj_param = ctx.obj_param
        obj = ctx.obj
        ineq_param = ctx.ineq_param
        ineqs = ctx.ineqs
        _ = ctx.eq_param
        _ = ctx.eqs
        
        # Locally assume the problem is a QP around the optimal solution
        x_star = torch.clone(x_star).detach()
        x_star.requires_grad_()
        obj_param = torch.clone(obj_param).detach()
        obj_param.requires_grad_()
        Q, _ = DiffCP.quadratic_approximation(
            x_star,
            lambda x_: obj(x_, obj_param)
        )

        # Compute gradients using OptNet
        d_z = -torch.linalg.solve(Q, grad_output)
        d_lambda = -torch.linalg.lstsq(G, grad_output)[0]
        d_nu = -torch.linalg.lstsq(A.T, grad_output)[0]

        grad_Q = torch.outer(d_z, x_star)
        grad_Q = 0.5 * (grad_Q + grad_Q.t())
        grad_q = d_z
        
        tmp1 = torch.diag(lambda_star)
        tmp2 = torch.matmul(d_lambda.unsqueeze(-1), x_star.unsqueeze(0))
        grad_G = torch.matmul(tmp1, tmp2) + torch.matmul(lambda_star.unsqueeze(-1), d_z.unsqueeze(0))

        grad_h = -torch.mv(torch.diag(lambda_star), d_lambda)
        grad_A = torch.outer(d_nu, x_star)
        grad_A = grad_A + grad_A.t()
        grad_b = -d_nu
    
        # Detach obtained gradients 
        grad_Q = grad_Q.detach()
        grad_q = grad_q.detach()
        grad_G = grad_G.detach()
        grad_h = grad_h.detach()
        grad_A = grad_A.detach()
        grad_b = grad_b.detach()

        # Back-propagate through the QP approximation function
        obj_param = obj_param.detach().requires_grad_()
        x_star = x_star.detach().requires_grad_()

        if USE_VJP_BASED_BACKWARD_IMPLEMENTATION:
            grads = torch.autograd.functional.vjp(
                lambda obj_param: DiffCP.quadratic_approximation(x_star, lambda x: obj(x, obj_param)),
                (obj_param,),
                v=(grad_Q, grad_q),
                create_graph=True,
                strict=True
            )
            grad_obj_param = grads[1][0]
        # Another solution is possible, although drastically less efficient
        else:  
            jacobians = torch.autograd.functional.jacobian(
                lambda obj_param: DiffCP.quadratic_approximation(x_star, lambda x: obj(x, obj_param)),
                (obj_param,),
                create_graph=True
            )
            jac_Q = jacobians[0][0]
            jac_q = jacobians[1][0]
            grad_obj_param = torch.einsum('ijk,ij->k', jac_Q, grad_Q.double())
            grad_obj_param += torch.einsum('jk,j->k', jac_q, grad_q.double())

        # Print gradient estimation for the parameters of the objective function.
        # Useful as a means to track convergency at runtime
        print(grad_obj_param)

        # Back-propagate through the linear approximation for ineq constraints
        ineq_param = ineq_param.detach().requires_grad_()

        Gis = []
        his = []
        for ineq in ineqs:
            G, h = DiffCP.linear_approximation(x_star, lambda x : ineq(ineq_param, x))
            Gis.append(G)
            his.append(h)

        matrix_term = torch.cat(tuple((Gi.unsqueeze(0)) for Gi in Gis), 0)
        vector_term = torch.cat(tuple((hi) for hi in his), 0)

        grad_ineq_param =  torch.linalg.lstsq(matrix_term, vector_term, rcond=None)[0]
        
        # Set grad_eq_param to None, as these are not present in the NLP formulation 
        grad_eq_param = None

        # Return the obtained gradients for the different sets of parameters
        # employed in the definition of the NLP task
        return grad_obj_param, None, grad_eq_param, None, grad_ineq_param, None, None

    @staticmethod
    def quadratic_approximation(x_star: Tensor, func: Callable):
        """
        Outputs the `Q` matrix and `q` vector associated with the quadratic approximation
        of a specified `func` function, with respect to a given `x_star` point.
        """
        grad = torch.autograd.functional.jacobian(func, (x_star,), create_graph=True)[0]
        Q = torch.autograd.functional.hessian(func, (x_star,), create_graph=True)[0][0]
        q = grad - torch.mv(Q, x_star)
        return Q, q
    
    @staticmethod
    def linear_approximation(x_star: Tensor, func: Callable):
        """
        Outputs the `A` matrix and `b` vector associated with the linear approximation
        of a specified `func` function, with respect to a given `x_star` point.
        """
        grad = torch.autograd.functional.jacobian(func, (x_star,), create_graph=True)[0]
        A = grad
        b = func(x_star) + torch.mv(grad.unsqueeze(0), x_star)
        return A, b

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
    _ = param[0]
    _ = param[1]
    return 2 - (x[0] + x[1])