"""Optimized program: using scipy optimization with soft-max surrogate to beat the constant lower bound."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft

N = 400

# EVOLVE-BLOCK-START
def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n
    H = fft(h, n=size)
    conv = np.real(ifft(H * H, n=size))
    return conv[:2*n-1]

def _soft_max(x, beta):
    """Soft approximation of max(x) using log-sum-exp."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _objective(params):
    """Objective: 2*N*max(autoconv)/sum(h)^2, using soft-max for differentiability."""
    h = np.tanh(params)  # bounded to [-1, 1]
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e6
    conv = _autoconv_fft(h)
    # Use exact max for final evaluation accuracy
    max_conv = np.max(conv)
    val = 2.0 * N * max_conv / (s * s)
    return abs(val)

def _objective_soft(params, beta):
    """Soft-max version for gradient-based optimization."""
    h = np.tanh(params)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e6
    conv = _autoconv_fft(h)
    smax = _soft_max(conv, beta)
    val = 2.0 * N * smax / (s * s)
    return abs(val)

def construct():
    t0 = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    # Start from a non-trivial initialization
    # Try a vector that's mostly 1s with some -1s to break symmetry
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _objective(np.zeros(N))  # value for all-ones via tanh(0)=0... 
    # Actually tanh(0)=0, so let's initialize properly
    # We want h close to ones, so params should be large positive
    best_params = np.full(N, 3.0)  # tanh(3) ≈ 0.995
    best_val = _objective(best_params)
    
    # Phase 1: Soft-max optimization with low beta (broad landscape)
    beta1 = 5.0
    res1 = minimize(
        lambda p: _objective_soft(p, beta1),
        best_params,
        method='L-BFGS-B',
        options={'maxiter': 200, 'maxfun': 5000, 'gtol': 1e-6}
    )
    if time.time() - t0 > deadline:
        return np.tanh(best_params)
    
    if res1.fun < best_val:
        best_params = res1.x
        best_val = _objective(best_params)
    
    # Phase 2: Soft-max optimization with higher beta (sharper max)
    beta2 = 50.0
    res2 = minimize(
        lambda p: _objective_soft(p, beta2),
        best_params,
        method='L-BFGS-B',
        options={'maxiter': 300, 'maxfun': 10000, 'gtol': 1e-7}
    )
    if time.time() - t0 > deadline:
        return np.tanh(best_params)
    
    if res2.fun < best_val:
        best_params = res2.x
        best_val = _objective(best_params)
    
    # Phase 3: Coordinate descent polish with exact objective
    h = np.tanh(best_params)
    improved = True
    while improved and (time.time() - t0) < deadline - 5:
        improved = False
        for i in range(N):
            if time.time() - t0 > deadline - 5:
                break
            old_val = _objective(best_params)
            # Try small perturbations
            for delta in [0.1, -0.1, 0.05, -0.05, 0.2, -0.2]:
                test_params = best_params.copy()
                test_params[i] += delta
                test_val = _objective(test_params)
                if test_val < old_val - 1e-10:
                    best_params = test_params
                    old_val = test_val
                    improved = True
                    break
    
    # Final: try a few random restarts if time permits
    if time.time() - t0 < deadline - 10:
        for restart in range(5):
            if time.time() - t0 > deadline - 10:
                break
            # Random perturbation of best
            noise = rng.normal(0, 0.5, N)
            p_try = best_params + noise
            res_r = minimize(
                lambda p: _objective_soft(p, beta2),
                p_try,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 3000, 'gtol': 1e-6}
            )
            if res_r.fun < best_val:
                best_params = res_r.x
                best_val = _objective(best_params)
    
    h = np.tanh(best_params)
    
    # Safety check: ensure finite and sum != 0
    if not np.all(np.isfinite(h)) or abs(np.sum(h)) < 1e-10:
        return np.ones(N)
    
    return h
# EVOLVE-BLOCK-END
