import numpy as np
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 140.0

def _softmax(x):
    # Stable softmax
    x = x - np.max(x)
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

def _compute_ratio(h):
    """Compute the raw score: L2^2 / (L1 * PSL)."""
    l1 = np.sum(h)
    if l1 < 1e-10:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    # Off-diagonal indices
    # Full conv length is 2N-1. Lag 0 is at index N-1.
    off_diag = np.concatenate([conv[:N-1], conv[N:]])
    psl = np.max(off_diag) if len(off_diag) > 0 else 0.0
    if psl < 1e-10:
        return l2_sq / l1 # Infinite case, treat as large
    return l2_sq / (l1 * psl)

def _cost_fn(x, tau=10.0):
    """
    Cost function to minimize.
    x: parameters for softmax.
    Returns LSE(off_diag) / L2^2.
    """
    h = _softmax(x)
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    off_diag = np.concatenate([conv[:N-1], conv[N:]])
    # Log-Sum-Exp for max approximation
    # max(off_diag) approx 1/tau * log(sum(exp(tau * off_diag)))
    # We want to minimize this ratio.
    # To avoid overflow in exp, shift by max
    max_off = np.max(off_diag)
    if max_off < 1e-10:
        lse_val = max_off
    else:
        lse_val = max_off + (1.0/tau) * np.log(np.sum(np.exp(tau * (off_diag - max_off))))
    
    return lse_val / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5.0 # Safety margin
    
    best_h = np.ones(N) / N
    best_score = _compute_ratio(best_h)
    best_x = np.zeros(N)
    
    # Strategy: Multi-start local optimization using Softmax parameterization
    # This allows the support to shift and weights to adjust continuously.
    
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on time
    # Each minimize call takes ~0.1-0.5s.
    # We can afford ~200 restarts in 135s? Maybe less.
    # Let's do 100 restarts, break if time is up.
    
    max_restarts = 150
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialization strategies
        if i % 3 == 0:
            # Random sparse: pick a few random indices, set x high there
            x = np.zeros(N)
            k = rng.integers(5, 15)
            indices = rng.choice(N, size=k, replace=False)
            x[indices] = 5.0
            x += rng.normal(0, 0.1, N)
        elif i % 3 == 1:
            # Random continuous
            x = rng.normal(0, 1.0, N)
        else:
            # Structured: Sidon-like or periodic
            # Try to initialize with a known good pattern
            # e.g. every k-th index
            k = rng.integers(5, 10)
            x = np.zeros(N)
            x[::k] = 3.0
            x += rng.normal(0, 0.1, N)
            
        # Optimize
        try:
            res = minimize(
                _cost_fn, 
                x, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            h_opt = _softmax(res.x)
            score = _compute_ratio(h_opt)
            
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
                best_x = res.x.copy()
                
        except Exception:
            pass
            
    return [float(x) for x in best_h]
