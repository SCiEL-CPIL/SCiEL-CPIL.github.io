import time
import numpy as np
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 150.0
SAFETY = 10.0
START_TIME = time.time()

def _value(h):
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

def _neg_value(h):
    """
    Objective function for minimization.
    Returns negative value.
    """
    return -_value(h)

def _params_to_h(params):
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
        if s < 1e-6: s = 1e-6
        h += A * np.exp(-((x - c)**2) / (2*s**2))
    return np.maximum(h, 0)

def _neg_value_params(params):
    """
    Objective for parametric optimization.
    """
    h = _params_to_h(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    return -_value(h)

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - SAFETY
    rng = np.random.default_rng(12345)

    # Initial seed: uniform
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    # Parametric search setup
    # We use K=3 Gaussians. This gives 9 parameters.
    # A > 0, 0 <= c <= 1, 0 < sigma <= 0.5
    K = 3
    dim = 3 * K
    
    # Define bounds for parametric space
    # A: [0, 5], c: [0, 1], sigma: [0.01, 0.5]
    bounds = [(0, 5), (0, 1), (0.01, 0.5)] * K
    
    # Global search over parameters using random restarts + L-BFGS-B
    n_restarts = 100
    
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Random initial parameters
        params0 = np.zeros(dim)
        for k in range(K):
            params0[3*k] = rng.uniform(0.1, 2.0)
            params0[3*k+1] = rng.uniform(0, 1)
            params0[3*k+2] = rng.uniform(0.05, 0.3)
            
        try:
            res = minimize(
                _neg_value_params,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.x is not None:
                h_opt = _params_to_h(res.x)
                val_opt = _value(h_opt)
                
                if val_opt > best_v:
                    best_v = val_opt
                    best_h = h_opt
                    
        except Exception:
            pass

    # Final polish on the best solution found in full dimension
    if time.time() < deadline - 2:
        try:
            bounds_full = [(0, None)] * N
            res = minimize(
                _neg_value, 
                best_h, 
                method='L-BFGS-B',
                bounds=bounds_full,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 100
                }
            )
            if res.x is not None:
                h_polished = np.clip(res.x, 0, None)
                val_polished = _value(h_polished)
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
