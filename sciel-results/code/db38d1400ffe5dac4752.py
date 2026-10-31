"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 16
D = 2

# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import minimize
import time
import random

def _pairwise_distances_sq(points):
    """Compute squared pairwise distances for an (n, d) array."""
    # Using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    return dist_sq

def _score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    dist_sq = _pairwise_distances_sq(points)
    # Exclude diagonal
    n = len(points)
    mask = ~np.eye(n, dtype=bool)
    d_sq = dist_sq[mask]
    if len(d_sq) == 0:
        return float('inf')
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq <= 1e-12:
        return float('inf')
    ratio = max_d_sq / min_d_sq
    return ratio

def _objective(x):
    """Objective function for optimization: squared ratio."""
    points = x.reshape(-1, D)
    return _score(points)

def _gradient(x):
    """
    Numerical gradient for the objective.
    Note: The objective is non-smooth at min/max, but we can use a smooth surrogate
    or just use numerical gradient which works reasonably well in practice for local search.
    Actually, let's use a smooth surrogate for better optimization behavior.
    We want to minimize max(d^2) / min(d^2).
    Let f = log(max(d^2)) - log(min(d^2)).
    max(d^2) can be approximated by soft_max, min by soft_min.
    soft_max(v, t) = t * log(sum(exp(v_i/t)))
    soft_min(v, t) = -t * log(sum(exp(-v_i/t)))
    So objective approx = soft_max(d^2, t) / soft_min(d^2, t) or log ratio.
    Let's use log ratio for numerical stability.
    obj = log(soft_max(d^2, t)) - log(soft_min(d^2, t))
    """
    points = x.reshape(-1, D)
    dist_sq = _pairwise_distances_sq(points)
    n = len(points)
    mask = ~np.eye(n, dtype=bool)
    d_sq = dist_sq[mask]
    
    # Smooth max and min
    t = 0.1  # Temperature for smoothing
    # To avoid overflow in exp, shift by max
    d_max = np.max(d_sq)
    d_min = np.min(d_sq)
    
    # Soft max
    exp_d = np.exp((d_sq - d_max) / t)
    soft_max_val = d_max + t * np.log(np.sum(exp_d))
    
    # Soft min
    exp_neg_d = np.exp(-(d_sq - d_min) / t)
    soft_min_val = d_min - t * np.log(np.sum(exp_neg_d))
    
    # Objective is log(soft_max) - log(soft_min)
    # But we need gradient. Let's just use finite difference or scipy's minimize with BFGS/CG.
    # Since we are rewriting, let's stick to a robust numerical approach or analytic if possible.
    # Actually, minimizing the ratio directly is hard.
    # Let's try to minimize the ratio directly using a custom gradient or just use Nelder-Mead which doesn't need gradient.
    # Nelder-Mead is robust for non-smooth functions.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial guess: Regular grid or circle
    # Let's start with a circle or a perturbed grid
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    x0 = np.array(grid, dtype=float).flatten()
    
    # Add some random perturbations to avoid local minima
    best_x = x0.copy()
    best_score = _score(best_x.reshape(-1, D))
    
    # Multi-start Nelder-Mead
    n_restarts = 10
    rng = np.random.default_rng(42)
    
    for i in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate initial point
        if i == 0:
            x_init = x0 + rng.normal(0, 0.1, size=x0.shape)
        else:
            # Random start or perturbation of best
            if i % 2 == 0:
                x_init = x0 + rng.normal(0, 0.5, size=x0.shape)
            else:
                x_init = best_x + rng.normal(0, 0.2, size=best_x.shape)
        
        # Optimize
        try:
            res = minimize(_objective, x_init, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time remains
    if time.time() < deadline - 2.0:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass
    
    points = best_x.reshape(-1, D)
    return points
# EVOLVE-BLOCK-END
