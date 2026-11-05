"""Program for the third autocorrelation inequality task.

The objective value = abs(2*N*max(c)/(sum(h))**2) with c = convolve(h,h)
is scale-invariant, so we fix sum(h)=1.
The previous assumption that max(c) >= sum(h_i^2) is incorrect for
np.convolve. The uniform vector gives max(c) = 1/N, value = 2.0.
However, if we can make the convolution c more uniform, max(c) can be
reduced to ~1/(2N), yielding a value of ~1.0.
We minimize the L2 norm of the convolution c (sum(c^2)) subject to
sum(h)=1. This encourages a flat convolution profile.
"""
import numpy as np
from scipy.optimize import minimize

N = 400


def _objective(h):
    c = np.convolve(h, h)
    return float(np.dot(c, c))


def construct():
    # Fix the normalisation sum(h) = 1 (objective is scale invariant).
    x0 = np.full(N, 1.0 / N)
    cons = [{"type": "eq", "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    res = minimize(_objective, x0, method="SLSQP", jac='2-point',
                   constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    h = res.x
    # Guard against numerical drift: renormalise to exact sum 1.
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        h = np.full(N, 1.0 / N)
    else:
        h = h / s
    return h
