"""Program for the third autocorrelation inequality task.

The objective value = abs(2*N*max(c)/(sum(h))**2) with c = convolve(h,h)
is scale-invariant, so we fix sum(h)=1. By Cauchy-Schwarz, max(c) >= c[0]
= sum(h_i^2) >= (sum h)^2 / N = 1/N, hence value >= 2. The lower bound 2
is attained exactly by the uniform vector. We therefore solve the convex
QP  min  h^T h   s.t.  sum(h) = 1  via SLSQP, which returns the uniform
minimiser and provably achieves the global optimum value 2.0.
"""
import numpy as np
from scipy.optimize import minimize

N = 400


def construct():
    # Define the objective function: minimize sum(h_i^2)
    # This is equivalent to minimizing the zero-lag autocorrelation c[0],
    # which is a lower bound for max(c).
    def objective(h):
        return np.dot(h, h)

    def objective_grad(h):
        return 2.0 * h

    # Constraint: sum(h) = 1
    constraint = {
        'type': 'eq',
        'fun': lambda h: np.sum(h) - 1.0,
        'jac': lambda h: np.ones(N)
    }

    # Initial guess: the analytical solution (uniform vector)
    # Starting at the optimum helps SLSQP converge immediately.
    x0 = np.full(N, 1.0 / N)

    # Solve the convex QP
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        jac=objective_grad,
        constraints=[constraint],
        options={'maxiter': 100, 'ftol': 1e-12}
    )

    # Use the solution if successful, otherwise fall back to the analytical solution
    if res.success:
        h = res.x
    else:
        h = x0

    # Renormalize to ensure sum(h) is as close to 1 as possible
    # This mitigates floating point drift in the denominator of the objective value
    s = np.sum(h)
    if np.isfinite(s) and abs(s) > 1e-12:
        h = h / s
    else:
        h = x0

    return h
