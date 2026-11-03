"""Minimize normalized max autoconvolution via sparse polynomial magnitude-squared."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600
DEADLINE = 150.0

def _compute_score(h):
    """Compute the normalized maximum autoconvolution score."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return np.inf
    energy = np.sum(h ** 2)
    if energy < 1e-15:
        return np.inf
    max_val = -np.inf
    for lag in range(N):
        if lag == 0:
            val = np.sum(h * h)
        else:
            val = np.sum(h[:N-lag] * h[lag:])
        if val > max_val:
            max_val = val
    return max_val / energy

def construct():
    t0 = time.time()
    best_score = np.inf
    best_heights = np.ones(N)
    
    # Order-1: P(z) = 1 + r*e^{iφ}*z
    def score1(params):
        r = params[0]
        phi = params[1]
        k = np.arange(N)
        theta = 2 * np.pi * k / N
        h = 1.0 + r**2 + 2*r*np.cos(theta - phi)
        if np.any(h < 0):
            return np.inf
        return _compute_score(h)
    
    try:
        res1 = differential_evolution(score1, [(0.0, 1.0), (0.0, 2*np.pi)],
                                       maxiter=50, popsize=20, seed=42, tol=1e-4, polish=True)
        if res1.fun < best_score:
            best_score = res1.fun
            r, phi = res1.x
            k = np.arange(N)
            theta = 2 * np.pi * k / N
            best_heights = 1.0 + r**2 + 2*r*np.cos(theta - phi)
    except Exception:
        pass
    
    if time.time() - t0 > DEADLINE - 15:
        return list(best_heights)
    
    # Order-2: P(z) = 1 + a*z + b*z^2 (real a, b)
    def score2(params):
        a = params[0]
        b = params[1]
        k = np.arange(N)
        theta = 2 * np.pi * k / N
        z = np.exp(1j * theta)
        h = np.abs(1.0 + a*z + b*z**2)**2
        if np.any(h < 0):
            return np.inf
        return _compute_score(h)
    
    try:
        res2 = differential_evolution(score2, [(-1.0, 1.0), (-1.0, 1.0)],
                                       maxiter=30, popsize=15, seed=42, tol=1e-4, polish=True)
        if res2.fun < best_score:
            best_score = res2.fun
            a, b = res2.x
            k = np.arange(N)
            theta = 2 * np.pi * k / N
            z = np.exp(1j * theta)
            best_heights = np.abs(1.0 + a*z + b*z**2)**2
    except Exception:
        pass
    
    if time.time() - t0 > DEADLINE - 5:
        return list(best_heights)
    
    return list(best_heights)
