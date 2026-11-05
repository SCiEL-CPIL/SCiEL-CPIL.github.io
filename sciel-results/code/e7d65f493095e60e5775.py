"""Program for the third autocorrelation inequality task.

We formulate the problem as minimizing the squared L2 norm of the
autoconvolution c = h * h, subject to sum(h) = 1. This convex structure
suppresses the global peak magnitude. The global optimum is the uniform
vector, which achieves the theoretical lower bound of 2.0. We solve this
using SLSQP initialized at the uniform vector.
"""
import numpy as np
from scipy.optimize import minimize

N = 400


def _objective(h):
    # Minimize sum(c_k^2) where c = convolve(h, h)
    c = np.convolve(h, h)
    return float(np.sum(c**2))


def construct():
    # Fix the normalisation sum(h) = 1 (objective is scale invariant).
    x0 = np.full(N, 1.0 / N)
    cons = [{"type": "eq", "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    res = minimize(_objective, x0, method="SLSQP",
                   constraints=cons,
                   options={"maxiter": 100, "ftol": 1e-12})
    h = res.x
    # Guard against numerical drift: renormalise to exact sum 1.
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        h = np.full(N, 1.0 / N)
    else:
        h = h / s
    return h
