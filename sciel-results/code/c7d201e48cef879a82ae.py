import numpy as np
from scipy.optimize import minimize

N = 400

def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    c = np.convolve(h, h)
    return 2.0 * N * np.max(c) / (s * s)

def _objective(h):
    # We minimize the max autocorrelation term
    # Scaling invariance allows us to fix sum(h)=1, but 
    # the objective is scale invariant anyway.
    # However, SLSQP works better with fixed scale.
    # We will constrain sum(h)=1.
    c = np.convolve(h, h)
    return np.max(c)

def construct():
    # Start with uniform
    x0 = np.ones(N) / N
    
    cons = [{"type": "eq", "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    
    # Minimize max(c)
    res = minimize(_objective, x0, method="SLSQP", constraints=cons,
                   options={"maxiter": 100, "ftol": 1e-6})
    
    h = res.x
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        h = np.ones(N) / N
    else:
        h = h / s
        
    return h
