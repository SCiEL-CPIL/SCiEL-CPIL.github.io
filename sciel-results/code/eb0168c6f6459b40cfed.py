import numpy as np
from scipy.optimize import minimize

N = 400

def _max_approx(c, tau=0.1):
    c_max = np.max(c)
    return c_max + tau * np.log(np.mean(np.exp((c - c_max) / tau)))

def _objective(h, tau=0.1):
    c = np.convolve(h, h)
    return _max_approx(c, tau)

def _grad_objective(h, tau=0.1):
    c = np.convolve(h, h)
    c_max = np.max(c)
    p = np.exp((c - c_max) / tau)
    p_mean = np.mean(p)
    # grad of max_approx = (1/tau) * p_k / p_mean ?
    # Let f(c) = c_max + tau log(mean(exp((c-c_max)/tau)))
    # df/dc_k = (1/tau) * exp((c_k-c_max)/tau) / mean(exp(...))
    # So grad_h = sum_k (df/dc_k) * dc_k/dh
    # dc_k/dh_j = h_{k-j} + h_{j-k}
    
    weights = (1.0 / tau) * p / p_mean
    # weights has length 2N-1
    grad = np.zeros(N)
    for j in range(N):
        s = 0.0
        for k in range(len(c)):
            if 0 <= k - j < N:
                s += weights[k] * h[k - j]
            if 0 <= j - k < N:
                s += weights[k] * h[j - k]
        grad[j] = s
    return grad

def construct():
    x0 = np.full(N, 1.0 / N)
    cons = [{"type": "eq", "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    res = minimize(_objective, x0, method="SLSQP", 
                   jac=_grad_objective,
                   constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-8})
    h = res.x
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        h = np.full(N, 1.0 / N)
    else:
        h = h / s
    return h
