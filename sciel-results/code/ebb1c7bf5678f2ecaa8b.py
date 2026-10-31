"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 100

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_h(w, c):
    """
    Construct h given width w and center c.
    h[i] = 0.5 + 0.5 * tanh((i - c) / w)
    """
    i = np.arange(N)
    if w < 1e-6:
        w = 1e-6
    h = 0.5 + 0.5 * np.tanh((i - c) / w)
    return h

def _solve_c(w):
    """
    Find c such that sum(h) = N/2.
    sum(h) is monotonic in c.
    """
    target_sum = N / 2.0
    
    def sum_diff(c):
        h = _construct_h(w, c)
        return np.sum(h) - target_sum
        
    lo = 0.0
    hi = float(N)
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_mid = _construct_h(w, mid)
        s_mid = np.sum(h_mid)
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    c_opt = (lo + hi) / 2.0
    return c_opt

def _evaluate_w(w):
    """Evaluate the objective for a specific width w."""
    c = _solve_c(w)
    h = _construct_h(w, c)
    
    # Correct sum to be exactly N/2
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    if abs(diff) > 1e-10:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        # Re-fix sum if clipping was aggressive
        for _ in range(20):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = h < 1.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = min(1.0, h[idx] + step)
            else:
                mask = h > 0.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = max(0.0, h[idx] + step)
                    
    return _max_overlap(h), h

def construct():
    """
    Construct the sequence of step heights.
    """
    # Grid search over log(w)
    # w ranges from very small (sharp step) to large (smooth)
    # log(w) from -5 to 5
    w_logs = np.linspace(-5, 5, 200)
    
    best_score = float('inf')
    best_h = None
    best_w_log = 0
    
    for wl in w_logs:
        w = np.exp(wl)
        if w < 1e-6: continue
        score, h = _evaluate_w(w)
        if score < best_score:
            best_score = score
            best_h = h
            best_w_log = wl
            
    # Refine around the best w_log found
    if best_w_log > -5 and best_w_log < 5:
        # Local refinement
        def obj(wl):
            w = np.exp(wl)
            if w < 1e-6: return 1e9
            score, _ = _evaluate_w(w)
            return score
            
        res = minimize_scalar(obj, bounds=(max(-5, best_w_log - 0.5), min(5, best_w_log + 0.5)), method='bounded', options={'xatol': 1e-5})
        if res.fun < best_score:
            best_w_log = res.x
            best_w = np.exp(best_w_log)
            best_score, best_h = _evaluate_w(best_w)
            
    return best_h

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
