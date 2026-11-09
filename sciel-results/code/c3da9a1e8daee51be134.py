"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import genlaguerre
from numpy.polynomial.laguerre import laggauss

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 30.0

def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 moment ratio for the Cohn-Goncalves Laguerre construction.
    |Psi(x)|^2 = e^{-x} [P(x)]^2 where P(x) = prod (x - x_i)
    C4 = M_4 / M_2^2 where M_k = integral x^k e^{-x} P(x)^2 dx
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e10
    
    # Check validity
    if np.any(roots <= 0):
        return 1e10
    if not np.all(np.diff(np.sort(roots)) > 1e-6):
        return 1e10
    
    # Use Gauss-Laguerre quadrature with enough nodes
    num_nodes = max(2 * m + 1, 100)
    try:
        nodes, weights = laggauss(num_nodes)
    except Exception:
        num_nodes = 2 * m + 1
        nodes, weights = laggauss(num_nodes)
    
    # Compute P(x)^2 = prod (x - x_i)^2 at each node
    diff = nodes[:, None] - roots[None, :]
    
    # Use log to avoid overflow
    log_p_sq = 2.0 * np.sum(np.log(np.abs(diff)), axis=1)
    log_p_sq = np.clip(log_p_sq, -700, 700)
    p_sq = np.exp(log_p_sq)
    
    # Compute moments
    m2 = np.sum(weights * nodes**2 * p_sq)
    m4 = np.sum(weights * nodes**4 * p_sq)
    
    if m2 <= 0 or not np.isfinite(m2) or not np.isfinite(m4):
        return 1e10
    
    c4 = m4 / (m2 ** 2)
    
    if not np.isfinite(c4):
        return 1e10
    
    return c4

def construct():
    start_time = time.time()
    
    m = 12
    n = 1
    
    # Initial guess: Zeros of L_m^{(1)}(x)
    try:
        L_init = genlaguerre(m, 1)
        init_roots = L_init.roots()
        init_roots = np.real(init_roots)
        init_roots = np.sort(init_roots)
        
        # Ensure they are positive
        init_roots = init_roots[init_roots > 0]
        
        if len(init_roots) != m:
            init_roots = np.array([1.0 + i * 2.0 for i in range(m)])
    except Exception:
        init_roots = np.array([1.0 + i * 2.0 for i in range(m)])
        
    best_roots = init_roots.copy()
    best_score = _compute_c4_bound(best_roots, m)
    
    bounds = [(1e-6, 1000.0)] * m
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed + _SAFETY_MARGIN > _DEADLINE:
            return 1e10
        return _compute_c4_bound(params, m)
    
    try:
        res = minimize(
            objective,
            best_roots,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < best_score:
            best_roots = res.x
            best_score = res.fun
    except Exception:
        pass
    
    if not _deadline_reached():
        best_roots = np.sort(best_roots)
        
        try:
            res2 = minimize(
                objective,
                best_roots,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res2.fun < best_score:
                best_roots = res2.x
                best_score = res2.fun
        except Exception:
            pass
    
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    
    for i in range(len(best_roots) - 1):
        if best_roots[i+1] - best_roots[i] < 1e-5:
            best_roots[i+1:] += 1e-5
    
    return [float(x) for x in best_roots]

# EVOLVE-BLOCK-END
