"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconv_max(h):
    """Compute supremum norm of autoconvolution."""
    c = np.correlate(h, h, mode='full')
    return np.max(c)

def _ratio_normalized(h):
    """Compute ratio with L1 norm fixed to 1. Returns ||h||_2^2 / ||h*h||_inf."""
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    max_conv = _autoconv_max(h_norm)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / max_conv

def _find_sidon_set(n, size, seed=42):
    """Find a Sidon set (B2 sequence) of given size within [0, n)."""
    rng = np.random.default_rng(seed)
    for attempt in range(200):
        support = set()
        candidates = list(range(n))
        rng.shuffle(candidates)
        
        for idx in candidates:
            if len(support) >= size:
                break
            can_add = True
            new_support = support | {idx}
            s = sorted(new_support)
            sums = {}
            for i in range(len(s)):
                for j in range(i, len(s)):
                    total = s[i] + s[j]
                    if total in sums:
                        if sums[total] != (i, j):
                            can_add = False
                            break
                    else:
                        sums[total] = (i, j)
                if not can_add:
                    break
            if can_add:
                support.add(idx)
        
        if len(support) >= size:
            return sorted(list(support))[:size]
    
    return list(range(0, n, n // size))[:size]

def _optimize_weights_slsqp(support, deadline):
    """Optimize weights on a fixed support using SLSQP with L1=1 constraint."""
    n = len(support)
    
    def objective(weights):
        h = np.zeros(N)
        h[support] = weights
        return -_ratio_normalized(h)
    
    # Constraints: sum(weights) = 1, weights >= 0
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]
    bounds = [(0, 1)] * n
    
    # Initial guess: equal weights
    w0 = np.ones(n) / n
    
    try:
        result = minimize(objective, w0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-10})
        if result.success or result.fun < objective(w0):
            best_w = np.maximum(result.x, 0)
            best_w /= np.sum(best_w)
            return best_w
    except Exception:
        pass
    
    return w0

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_val = _ratio_normalized(best_h)
    
    # Try various support sizes with Sidon-like structures
    for support_size in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random seeds for diversity
        for seed in [42, 123, 777, 2024]:
            if time.time() > deadline - 10:
                break
            
            support = _find_sidon_set(N, support_size, seed=seed)
            if len(support) < 2:
                continue
            
            # Optimize weights on this support
            weights = _optimize_weights_slsqp(support, deadline)
            
            h = np.zeros(N)
            h[support] = weights
            
            val = _ratio_normalized(h)
            if val > best_val:
                best_val = val
                best_h = h.copy()
    
    # Final global optimization from best point
    if time.time() < deadline - 5:
        def global_objective(h):
            return -_ratio_normalized(h)
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}
        ]
        bounds = [(0, 1)] * N
        
        try:
            result = minimize(global_objective, best_h, method='SLSQP',
                             bounds=bounds, constraints=constraints,
                             options={'maxiter': 1000, 'ftol': 1e-12})
            if result.success or result.fun < -best_val:
                best_h = np.maximum(result.x, 0)
                best_h /= np.sum(best_h)
                best_val = _ratio_normalized(best_h)
        except Exception:
            pass
    
    return list(best_h)
# EVOLVE-BLOCK-END
