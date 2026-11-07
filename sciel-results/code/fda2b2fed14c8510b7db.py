import time
import numpy as np
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 150.0
SAFETY = 10.0

def compute_value(h):
    """
    Computes the objective value for a given height array h.
    Vectorized for speed.
    """
    h = np.asarray(h, dtype=float)
    # Ensure non-negative and finite
    h = np.clip(h, 0, None)
    if not np.all(np.isfinite(h)):
        return 0.0
    
    g = np.convolve(h, h)
    m = len(g)
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Spacing w
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    y_curr = y[:-1]
    y_next = y[1:]
    term = y_curr**2 + y_curr * y_next + y_next**2
    l2 = (w / 3.0) * np.sum(term)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def gmm_from_params(params):
    """
    Convert parameters (A, c, sigma) for K Gaussians to a 50-dim height vector.
    params shape: (3*K,) -> [A1, c1, s1, A2, c2, s2, ...]
    """
    K = len(params) // 3
    h = np.zeros(N)
    x = np.linspace(0, 1, N)
    for k in range(K):
        A = params[3*k]
        c = params[3*k + 1]
        s = params[3*k + 2]
        if s < 1e-6:
            s = 1e-6
        h += A * np.exp(-((x - c)**2) / (2*s**2))
    return np.maximum(h, 0)

def neg_value_gmm(params):
    """
    Objective for parametric optimization.
    Returns negative value (for minimization).
    """
    h = gmm_from_params(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    return -compute_value(h)

def _soft_norm_inf(g, alpha=50.0):
    """
    Smooth surrogate for max(abs(g)) using log-sum-exp.
    """
    abs_g = np.abs(g)
    if len(abs_g) == 0:
        return 0.0
    max_val = np.max(abs_g)
    if max_val < 1e-12:
        return 0.0
    # log-sum-exp trick for stability
    shifted = abs_g - max_val
    log_sum = max_val + np.log(np.sum(np.exp(alpha * shifted)) / alpha)
    return log_sum

def neg_value_gmm_smooth(params, alpha=50.0):
    """
    Objective for parametric optimization using smooth norm_inf surrogate.
    """
    h = gmm_from_params(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    y_curr = y[:-1]
    y_next = y[1:]
    term = y_curr**2 + y_curr * y_next + y_next**2
    l2 = (w / 3.0) * np.sum(term)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    if norm_1 < 1e-12:
        return 1e10
    
    smooth_inf = _soft_norm_inf(g, alpha)
    if smooth_inf < 1e-12:
        return 1e10
        
    return -(l2 / (norm_1 * smooth_inf))

def _neg_value_full(h):
    """
    Objective for full 50-dim optimization.
    """
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return 1e10
    return -compute_value(h)

def _make_initial_guesses(rng, K, n_guesses):
    """
    Generate diverse initial parameter guesses.
    Includes Gaussian, spike, and flat profiles.
    """
    guesses = []
    
    # Standard Gaussian mixtures
    for i in range(n_guesses // 3):
        params = np.zeros(3 * K)
        for k in range(K):
            params[3*k] = rng.uniform(0.1, 3.0)
            params[3*k+1] = rng.uniform(0, 1)
            params[3*k+2] = rng.uniform(0.02, 0.4)
        guesses.append(params)
    
    # Spike-like profiles (narrow Gaussians with high amplitude)
    for i in range(n_guesses // 3):
        params = np.zeros(3 * K)
        for k in range(K):
            params[3*k] = rng.uniform(1.0, 5.0)
            params[3*k+1] = rng.uniform(0, 1)
            params[3*k+2] = rng.uniform(0.01, 0.08)
        guesses.append(params)
    
    # Flatter profiles
    for i in range(n_guesses // 3):
        params = np.zeros(3 * K)
        for k in range(K):
            params[3*k] = rng.uniform(0.5, 1.5)
            params[3*k+1] = rng.uniform(0, 1)
            params[3*k+2] = rng.uniform(0.15, 0.45)
        guesses.append(params)
    
    return guesses

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - SAFETY
    rng = np.random.default_rng(12345)

    # Initial seed: uniform
    best_h = np.ones(N)
    best_v = compute_value(best_h)
    
    # Parametric search with K=4 Gaussians (12 parameters)
    K = 4
    dim = 3 * K
    
    # Define bounds for parametric space
    bounds = [(0.01, 5.0), (0.0, 1.0), (0.005, 0.5)] * K
    
    # Phase 1: Global search with smooth surrogate for better gradients
    n_restarts = 80
    guesses = _make_initial_guesses(rng, K, n_restarts)
    
    for i, params0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break
        
        try:
            # Use smooth norm_inf surrogate for better gradient behavior
            res = minimize(
                neg_value_gmm_smooth,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 80
                }
            )
            
            if res.x is not None:
                h_opt = gmm_from_params(res.x)
                val_opt = compute_value(h_opt)
                
                if val_opt > best_v:
                    best_v = val_opt
                    best_h = h_opt
                    
        except Exception:
            pass
    
    # Phase 2: Polish the best parametric solution using exact (nonsmooth) objective
    if time.time() < deadline - 15:
        try:
            # Re-optimize the best parameters with exact objective
            best_params = None
            best_param_val = -1e10
            
            # Try to recover good parameters from best_h by fitting
            # Use the last successful params or do a local search
            for trial in range(5):
                if time.time() > deadline - 15:
                    break
                # Perturb around current best
                if best_params is None:
                    # Initialize from a random guess close to uniform
                    p_init = np.zeros(dim)
                    for k in range(K):
                        p_init[3*k] = 1.0
                        p_init[3*k+1] = 0.5
                        p_init[3*k+2] = 0.2
                else:
                    p_init = best_params + rng.normal(0, 0.05, size=dim)
                    p_init = np.clip(p_init, 
                                    [b[0] for b in bounds], 
                                    [b[1] for b in bounds])
                
                res2 = minimize(
                    neg_value_gmm,
                    p_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxls': 100
                    }
                )
                
                if res2.x is not None:
                    h_opt = gmm_from_params(res2.x)
                    val_opt = compute_value(h_opt)
                    
                    if val_opt > best_v:
                        best_v = val_opt
                        best_h = h_opt
                        best_params = res2.x
                        
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish on full 50-dim
    if time.time() < deadline - 5:
        try:
            bounds_full = [(0.0, None)] * N
            res3 = minimize(
                _neg_value_full,
                best_h,
                method='L-BFGS-B',
                bounds=bounds_full,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxls': 200
                }
            )
            if res3.x is not None:
                h_polished = np.clip(res3.x, 0, None)
                val_polished = compute_value(h_polished)
                if val_polished > best_v:
                    best_v = val_polished
                    best_h = h_polished
        except Exception:
            pass
    
    # Final safety: ensure finite and non-negative
    best_h = np.maximum(best_h, 0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return [float(x) for x in best_h]
