"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_value(h):
    conv = np.convolve(h, h)
    s = np.sum(h)
    if s == 0:
        return float('inf')
    return abs(2 * N * np.max(conv) / s**2)

def _smooth_objective(x, beta=100.0):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    # Soft-max approximation of max(conv)
    m = np.max(conv)
    # Use log-sum-exp for smooth max
    shifted = conv - m
    log_sum_exp = m + np.log(np.sum(np.exp(shifted)))
    return 2 * N * log_sum_exp / s**2

def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Start with a smooth window (e.g., raised cosine / Hann)
    # This should have lower peak autocorrelation than uniform
    t = np.linspace(0, 1, N)
    h0 = 0.5 * (1 + np.cos(2 * np.pi * t))  # Hann window, non-negative
    
    best_h = h0.copy()
    best_val = _compute_value(best_h)
    
    # Try different initializations
    inits = [
        h0.copy(),
        np.sin(np.pi * t),  # Sine window
        np.ones(N),  # Seed
        np.cos(np.pi * t),  # Cosine (has negative part)
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_objective,
                init.copy(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8},
                beta=None  # Can't pass beta directly, use closure
            )
            # Actually, let me fix the smooth objective to use a fixed beta
            if time.time() > deadline:
                break
        except:
            continue
    
    # Better approach: use a proper smooth objective with fixed beta
    def obj(x):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        m = np.max(conv)
        # Smooth max with beta
        beta = 50.0
        shifted = (conv - m) / beta
        log_sum_exp = m + beta * np.log(np.sum(np.exp(shifted)))
        return 2 * N * log_sum_exp / s**2
    
    best_h = h0.copy()
    best_val = _compute_value(best_h)
    
    inits = [
        h0.copy(),
        np.sin(np.pi * t),
        np.ones(N) * 0.01,  # Small perturbation from uniform
        np.cos(np.pi * t),
        np.ones(N) + 0.1 * np.random.randn(N),
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                obj,
                init.copy(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
        except Exception:
            continue
    
    # Polish with higher beta (closer to true max)
    if time.time() < deadline - 20:
        def obj_high_beta(x):
            h = x
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h)
            m = np.max(conv)
            beta = 200.0
            shifted = (conv - m) / beta
            log_sum_exp = m + beta * np.log(np.sum(np.exp(shifted)))
            return 2 * N * log_sum_exp / s**2
        
        try:
            res = minimize(
                obj_high_beta,
                best_h.copy(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.success:
                val = _compute_value(res.x)
                if val < best_val:
                    best_h = res.x.copy()
                    best_val = val
        except Exception:
            pass
    
    # Final evaluation with true max
    final_val = _compute_value(best_h)
    
    # Ensure sum is not zero and finite
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
