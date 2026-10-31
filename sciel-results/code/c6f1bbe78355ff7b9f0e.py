import numpy as np
from scipy.optimize import differential_evolution

N = 50

def _score(x):
    f = np.asarray(x, dtype=float)
    f = np.maximum(f, 0)
    l1 = np.sum(f)
    if l1 < 1e-12:
        return 0.0
    f = f / l1
    l2_sq = np.sum(f**2)
    conv = np.convolve(f, f, mode='full')
    amax = np.max(conv)
    if amax < 1e-12:
        return 0.0
    return l2_sq / amax

def _objective(x):
    # We want to minimize -score
    return -_score(x)

def construct():
    # Define bounds: [0, 1] is sufficient due to homogeneity, but DE works better with finite bounds.
    # Since scaling doesn't matter, we can restrict search to [0, 1] or [0, 2].
    bounds = [(0, 1)] * N
    
    # Run Differential Evolution
    # tol=1e-6, maxiter=50, popsize=20
    # This should be fast enough.
    try:
        res = differential_evolution(
            _objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20, 
            tol=1e-6, 
            polish=True # Use L-BFGS-B for final polish if polish=True
        )
        f_opt = res.x
    except Exception:
        # Fallback to a simple random or uniform
        f_opt = np.ones(N)
        
    # Ensure non-negative and normalize
    f_opt = np.maximum(f_opt, 0)
    l1 = np.sum(f_opt)
    if l1 > 1e-12:
        f_opt = f_opt / l1
    else:
        f_opt = np.ones(N) / N
        
    return list(f_opt)
