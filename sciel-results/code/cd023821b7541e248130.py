"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite_physicists(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2 * x
    for k in range(2, n + 1):
        h_n = 2 * x * h_nm1 - 2 * (k - 1) * h_nm2
        h_nm2 = h_nm1
        h_nm1 = h_n
    return h_nm1

def _precompute_matrices():
    """Precompute matrices using Gauss-Hermite Quadrature."""
    n_nodes = 20 
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    
    degrees = [0, 4, 8, 12]
    n_basis = len(degrees)
    
    M_G = np.zeros((n_basis, n_basis))
    M_X = np.zeros((n_basis, n_basis))
    M_Q = np.zeros((n_basis, n_basis))
    
    H_vals = {}
    Q_vals = {}
    
    for d in degrees:
        H_vals[d] = _hermite_physicists(d, nodes)
        if d == 0:
            Q_vals[d] = -nodes 
        else:
            H_prev = _hermite_physicists(d-1, nodes)
            Q_vals[d] = 2 * d * H_prev - nodes * H_vals[d]
            
    for i, di in enumerate(degrees):
        for j, dj in enumerate(degrees):
            M_G[i, j] = np.sum(weights * H_vals[di] * H_vals[dj])
            M_X[i, j] = np.sum(weights * nodes**2 * H_vals[di] * H_vals[dj])
            M_Q[i, j] = np.sum(weights * Q_vals[di] * Q_vals[dj])
            
    return M_G, M_X, M_Q

_M_G, _M_X, _M_Q = None, None, None

def _get_matrices():
    global _M_G, _M_X, _M_Q
    if _M_G is None:
        _M_G, _M_X, _M_Q = _precompute_matrices()
    return _M_G, _M_X, _M_Q

def _compute_uncertainty_product(c0, c4, c8):
    """Compute the uncertainty product using analytic matrices."""
    M_G, M_X, M_Q = _get_matrices()
    
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
    
    c_full = np.array([c0, c4, c8, c12])
    
    norm_sq = c_full @ M_G @ c_full
    if norm_sq < 1e-15:
        return 1e10
        
    x2_mean = (c_full @ M_X @ c_full) / norm_sq
    p2_mean = (c_full @ M_Q @ c_full) / norm_sq
    
    sigma_x2 = x2_mean
    sigma_p2 = p2_mean
    
    if sigma_x2 < 0 or sigma_p2 < 0:
        return 1e10
        
    return np.sqrt(sigma_x2 * sigma_p2)

def _objective(params):
    """Objective function for optimization."""
    c0, c4, c8 = params
    try:
        val = _compute_uncertainty_product(c0, c4, c8)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    _get_matrices()
    
    best_params = np.array([-113/100, 1/25, 1/3240])
    best_val = _objective(best_params)
    
    rng = np.random.default_rng(42)
    
    n_starts = 100
    for i in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
            
        c0 = rng.uniform(-5, 5)
        c4 = rng.uniform(-0.5, 0.5)
        c8 = rng.uniform(-0.05, 0.05)
        
        p = np.array([c0, c4, c8])
        
        try:
            res = minimize(_objective, p, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
            res = differential_evolution(_objective, bounds, seed=123, maxiter=100, 
                                         tol=1e-12, polish=True, workers=1, popsize=20)
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    return np.array([best_params[0], best_params[1], best_params[2]])
# EVOLVE-BLOCK-END
