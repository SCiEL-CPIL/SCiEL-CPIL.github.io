import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _objective(x):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin
    
    # Start with a smooth initial guess
    best_h = np.ones(N)
    best_val = 2.0
    
    # Try several initializations and optimize
    inits = []
    # Linear ramp
    inits.append(np.linspace(1, -1, N))
    # Sine wave
    inits.append(np.sin(np.pi * np.arange(N) / N))
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2 / 0.1))
    # Triangular
    inits.append(1.0 - np.abs(2.0 * np.arange(N) / N - 1.0))
    # Cosine window
    inits.append(0.5 * (1 + np.cos(np.pi * np.arange(N) / N)))
    
    for h0 in inits:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(_objective, h0, method='BFGS', 
                          options={'maxiter': 200, 'gtol': 1e-8})
            if res.success or res.fun < best_val:
                h_new = res.x
                val = _objective(h_new)
                if val < best_val and np.all(np.isfinite(h_new)) and abs(np.sum(h_new)) > 1e-12:
                    best_h = h_new
                    best_val = val
        except Exception:
            pass
    
    # Local refinement with smaller steps
    if time.time() < deadline - 10:
        try:
            res = minimize(_objective, best_h, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                h_new = res.x
                val = _objective(h_new)
                if val < best_val and np.all(np.isfinite(h_new)) and abs(np.sum(h_new)) > 1e-12:
                    best_h = h_new
                    best_val = val
        except Exception:
            pass
    
    # Verify
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    return best_h
