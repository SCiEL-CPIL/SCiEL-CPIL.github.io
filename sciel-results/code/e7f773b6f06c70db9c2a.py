"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / (2 * N)

def _is_golomb(ruler):
    """Check if the set of indices forms a Golomb ruler (unique differences)."""
    diffs = set()
    for i in range(len(ruler)):
        for j in range(i):
            d = ruler[i] - ruler[j]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _find_golomb_ruler(n, target_size):
    """Find a Golomb ruler of target_size in range(n)."""
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        # Try to add i
        new_ruler = ruler + [i]
        if _is_golomb(new_ruler):
            ruler = new_ruler
            if len(ruler) >= target_size:
                break
    return ruler

def _compute_ratio(h):
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    if l1 == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    max_val = max_conv * DELTA
    if max_val == 0:
        return 0.0
    return l2_sq / (l1 * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    best_h = None
    best_ratio = 0.0
    
    # Try to find a good Golomb ruler
    # Larger K might be better if it allows for better weight distribution?
    # But R ~ 1/(K * Delta * avg_w) if uniform.
    # If we optimize weights, spiky is better.
    # Let's try a few sizes.
    
    for target_size in [10, 8, 6]:
        if time.time() > deadline:
            break
            
        ruler = _find_golomb_ruler(N, target_size)
        if len(ruler) < 4:
            continue
            
        K = len(ruler)
        # Initial weights: uniform
        h0 = np.zeros(N)
        for idx in ruler:
            h0[idx] = 1.0 / K
            
        # Objective: maximize ratio. Minimize -ratio.
        # Constraints: sum(h) = 1, h >= 0.
        # We optimize on the support indices.
        
        def objective(w):
            h = np.zeros(N)
            for i, idx in enumerate(ruler):
                h[idx] = w[i]
            return -_compute_ratio(h)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(K)]
        
        # Run optimization
        try:
            res = minimize(objective, h0[ruler], method='L-BFGS-B', bounds=bounds, constraints=constraints, options={'maxiter': 50, 'ftol': 1e-6})
            if res.success:
                w_opt = res.x
                h_opt = np.zeros(N)
                for i, idx in enumerate(ruler):
                    h_opt[idx] = w_opt[i]
                
                ratio = _compute_ratio(h_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h_opt.copy()
        except Exception:
            pass

    if best_h is None:
        # Fallback to single spike
        best_h = np.zeros(N)
        best_h[0] = 1.0
        
    return [float(x) for x in best_h]
