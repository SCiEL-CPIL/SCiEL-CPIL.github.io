import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconv(x):
    """Compute autoconvolution of x (linear convolution, centered)."""
    # x has length N, autoconv has length 2N-1
    # (x*x)_k = sum_i x_i * x_{k-i} with appropriate indexing
    # For zero-based: (x*x)[j] = sum_i x[i] * x[j-i] for valid indices
    n = len(x)
    ac = np.zeros(2*n - 1)
    for i in range(n):
        if x[i] != 0:
            # x[i] contributes to ac[i+j] for j=0..n-1
            ac[i:i+n] += x[i] * x
    return ac

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s
    best_x = None
    best_score = -np.inf

    # Initial guess: spread out to reduce autoconv peak
    # Uniform distribution gives low autoconv peak
    x0 = np.ones(N) / N

    def objective(x):
        # Minimize negative of ||x||_2^2
        return -np.dot(x, x)

    def autoconv_peak(x):
        ac = _autoconv(x)
        return np.max(ac)

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: 1.0 - autoconv_peak(x)},  # peak <= 1
    ]
    bounds = [(0.0, 1.0)] * N

    # Try multiple initializations
    inits = [
        np.ones(N) / N,
        np.maximum(0, np.cos(np.linspace(0, np.pi, N))) / np.sum(np.maximum(0, np.cos(np.linspace(0, np.pi, N)))),
        np.ones(N) / N,  # duplicate for different method
    ]

    # Add some random inits
    rng = np.random.default_rng(42)
    for _ in range(3):
        r = rng.random(N)
        r = np.maximum(0, r - 0.3)
        if r.sum() > 0:
            r /= r.sum()
            inits.append(r)

    for x_init in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, x_init, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            if res.success or res.fun < best_score:
                x = res.x
                x = np.maximum(0, x)
                x_sum = x.sum()
                if x_sum > 0:
                    x = x / x_sum
                    ac = _autoconv(x)
                    peak = np.max(ac)
                    if peak > 0:
                        score = np.dot(x, x) / (x.sum() * peak)
                        if score > best_score:
                            best_score = score
                            best_x = x.copy()
        except Exception:
            pass

    if best_x is None:
        return [1.0] * N

    return list(best_x)
