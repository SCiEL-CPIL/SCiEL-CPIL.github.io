"""Program exploring non-constant step heights to reduce the third autocorrelation constant."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _objective(h):
    s = np.sum(h)
    if s == 0:
        return 1e12
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s * s))

def construct():
    # Start from a non-trivial profile: a cosine that goes negative in the middle.
    # This is distinct from the constant-vector optimum and explores a different basin.
    t = np.linspace(-1.0, 1.0, N)
    h0 = 1.0 + 0.5 * np.cos(np.pi * t)
    
    # Use L-BFGS-B for bounded optimization; allow heights in [-2, 2] to stay finite.
    bounds = [(-2.0, 2.0)] * N
    result = minimize(_objective, h0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
    
    h_opt = result.x
    # Ensure sum is not zero (shouldn't happen with this start, but guard anyway)
    if np.sum(h_opt) == 0:
        h_opt = h0
    return h_opt
# EVOLVE-BLOCK-END
