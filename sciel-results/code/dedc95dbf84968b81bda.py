"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _softmin(v, beta):
    """
    Vectorized softmin.
    v: array of shape (..., k)
    returns: array of shape (...)
    approximates min(v, axis=-1)
    """
    m = np.min(v, axis=-1, keepdims=True)
    exp_v = np.exp(-beta * (v - m))
    sum_exp = np.sum(exp_v, axis=-1)
    return m - np.log(sum_exp) / beta

def _vec_objective(x, W, H, beta=50.0):
    """
    Vectorized objective with soft-min approximation.
    x: [x_0..x_{n-1}, y_0..y_{n-1}, W]
    H is derived from W: H = 2.0 - W
    """
    n = N
    cx = x[:n]
    cy = x[n:2*n]
    w = x[2*n]
    h = 2.0 - w
    
    if w <= 0.01 or h <= 0.01:
        return 1e10

    # Compute pairwise distances
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    D = np.sqrt(dx*dx + dy*dy)
    
    # Mask diagonal
    idx_diag = np.arange(n)
    D[idx_diag, idx_diag] = 1e6
    
    # Terms for each circle i:
    # Cols 0-3: Wall distances
    # Cols 4-4+n-1: Pairwise distances / 2
    terms = np.empty((n, n+3))
    terms[:, 0] = cx
    terms[:, 1] = w - cx
    terms[:, 2] = cy
    terms[:, 3] = h - cy
    terms[:, 4:] = D / 2.0
    
    # Apply softmin over the last axis
    radii = _softmin(terms, beta)
    
    return -np.sum(radii)

def _hard_objective(x, W, H):
    """
    Exact objective for final validation and polish.
    """
    n = N
    cx = x[:n]
    cy = x[n:2*n]
    
    radii = np.full(n, np.inf)
    radii = np.minimum(radii, cx)
    radii = np.minimum(radii, W - cx)
    radii = np.minimum(radii, cy)
    radii = np.minimum(radii, H - cy)
    
    # Vectorized pairwise
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    D = np.sqrt(dx*dx + dy*dy)
    idx = np.arange(n)
    D[idx, idx] = np.inf
    min_d = np.min(D, axis=1)
    radii = np.minimum(radii, min_d / 2.0)
    
    return -np.sum(radii)

def _optimize_smooth_aspect(W, H, n, time_limit=10, beta=50.0):
    """
    Optimize using L-BFGS-B with smooth surrogate.
    """
    # Initial guess: Grid
    n_cols = int(math.ceil(math.sqrt(n * W / H)))
    n_rows = int(math.ceil(n / n_cols))
    dx = W / n_cols
    dy = H / n_rows
    x0 = np.zeros(2*n + 1)
    for i in range(n):
        c = i % n_cols
        r = i // n_rows
        x0[i] = (c + 0.5) * dx
        x0[n+i] = (r + 0.5) * dy
    x0[2*n] = W
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.8))
    for i in range(n):
        bounds.append((0.0, 1.8))
    bounds.append((0.3, 1.7))
    
    x0[2*n] = np.clip(x0[2*n], 0.3, 1.7)
    
    def obj_full(x):
        w = x[2*n]
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return 1e10
        return _vec_objective(x, w, h, beta)
    
    try:
        result = minimize(obj_full, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
        return result.x
    except Exception:
        return x0

def _polish(x, n, time_limit=5):
    """
    Final polish using Nelder-Mead on the hard objective.
    """
    def obj_hard(x):
        w = x[2*n]
        h = 2.0 - w
        if w <= 0.01 or h <= 0.01:
            return 1e10
        return _hard_objective(x, w, h)
    
    try:
        result = minimize(obj_hard, x, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        return result.x
    except Exception:
        return x

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    n = N
    best_x = None
    best_val = -1e10
    
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.9, 1.1]
    
    # Phase 1: Smooth Optimization for multiple aspects
    for aspect in aspects:
        if time.time() > deadline - 20:
            break
        W0 = 2.0 * aspect / (1.0 + aspect)
        H0 = 2.0 / (1.0 + aspect)
        
        x_opt = _optimize_smooth_aspect(W0, H0, n, time_limit=10)
        
        w = x_opt[2*n]
        h = 2.0 - w
        val = -_hard_objective(x_opt, w, h)
        
        if val > best_val:
            best_val = val
            best_x = x_opt.copy()
            
    # Phase 2: Polish the best solution
    if best_x is not None and time.time() < deadline - 10:
        x_polished = _polish(best_x, n, time_limit=10)
        w = x_polished[2*n]
        h = 2.0 - w
        val = -_hard_objective(x_polished, w, h)
        if val > best_val:
            best_val = val
            best_x = x_polished
    
    # Build output
    if best_x is None:
        W = 1.0
        H = 1.0
        n_cols = int(math.ceil(math.sqrt(n * W / H)))
        dx = W / n_cols
        dy = H / n_cols
        cx = np.array([(i % n_cols + 0.5) * dx for i in range(n)])
        cy = np.array([(i // n_cols + 0.5) * dy for i in range(n)])
        radii = np.full(n, 0.5 * min(dx, dy) - 0.005)
        return np.column_stack([cx, cy, radii])
        
    cx = best_x[:n]
    cy = best_x[n:2*n]
    w = best_x[2*n]
    h = 2.0 - w
    
    # Compute final radii (Hard)
    radii = np.full(n, np.inf)
    radii = np.minimum(radii, cx)
    radii = np.minimum(radii, w - cx)
    radii = np.minimum(radii, cy)
    radii = np.minimum(radii, h - cy)
    
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    D = np.sqrt(dx*dx + dy*dy)
    idx = np.arange(n)
    D[idx, idx] = np.inf
    min_d = np.min(D, axis=1)
    radii = np.minimum(radii, min_d / 2.0)
    
    radii = np.maximum(radii, 0.0)
    
    # Clamp
    for i in range(n):
        r = radii[i]
        cx[i] = np.clip(cx[i], r, w - r)
        cy[i] = np.clip(cy[i], r, h - r)
        
    # Recompute radii after clamp
    radii = np.full(n, np.inf)
    radii = np.minimum(radii, cx)
    radii = np.minimum(radii, w - cx)
    radii = np.minimum(radii, cy)
    radii = np.minimum(radii, h - cy)
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    D = np.sqrt(dx*dx + dy*dy)
    idx = np.arange(n)
    D[idx, idx] = np.inf
    min_d = np.min(D, axis=1)
    radii = np.minimum(radii, min_d / 2.0)
    radii = np.maximum(radii, 0.0)
    
    return np.column_stack([cx, cy, radii])

# EVOLVE-BLOCK-END
