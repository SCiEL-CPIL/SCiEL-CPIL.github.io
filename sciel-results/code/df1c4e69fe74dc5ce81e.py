"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_linear_overlaps(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
    Returns an array of overlap values C_k for k != 0.
    C_k = sum_i h_i * (1 - h_{i+k})
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    c_neg = c[:N-1]
    c_pos = c[N:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _get_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return np.max(overlaps)

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses iterative water-filling style projection.
    """
    h = np.clip(h, 0, 1).copy()
    target = TARGET_SUM
    current = np.sum(h)
    
    if abs(current - target) < 1e-12:
        return h
    
    for _ in range(100):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        
        if diff > 0:
            mask = h < 1.0
            if not np.any(mask):
                break
            count = np.sum(mask)
            step = diff / count
            h[mask] += step
            h = np.clip(h, 0, 1)
        else:
            mask = h > 0.0
            if not np.any(mask):
                break
            count = np.sum(mask)
            step = (-diff) / count
            h[mask] -= step
            h = np.clip(h, 0, 1)
    
    return h

def _objective_lse(x, beta, lam_sum):
    """
    Objective function using LogSumExp approximation of the max overlap.
    """
    h = np.clip(x, 0, 1)
    overlaps = _get_linear_overlaps(h)
    
    if len(overlaps) == 0:
        max_ov = 0.0
    else:
        max_ov = np.max(overlaps)
        
    if max_ov > 1000:
        max_ov = 1000
        
    if beta > 0:
        shifted = overlaps - max_ov
        # Clip to prevent overflow in exp
        shifted = np.clip(shifted, -50/beta, 0)
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        lse = max_ov
    
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return lse + sum_penalty

def _objective_epigraph(x, beta, lam_sum):
    """
    Objective function using epigraph formulation.
    x = [h_0, ..., h_{N-1}, t]
    Minimize t + sum_penalty
    Subject to: overlaps <= t (approximated via log-sum-exp)
    """
    h = np.clip(x[:N], 0, 1)
    t = x[N]
    
    overlaps = _get_linear_overlaps(h)
    
    if len(overlaps) == 0:
        lse = 0.0
    else:
        max_ov = np.max(overlaps)
        if max_ov > 1000:
            max_ov = 1000
        shifted = overlaps - max_ov
        shifted = np.clip(shifted, -50/beta, 0)
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    # Penalize if lse > t
    if lse > t:
        penalty = (lse - t) ** 2
    else:
        penalty = 0.0
    
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return t + penalty + sum_penalty

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Uses epigraph formulation with auxiliary variable t for smoother optimization.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = np.random.default_rng(42)
    
    # Start from a structured initial guess
    best_h = np.full(N, 0.5)
    t = np.arange(N)
    best_h += 0.3 * np.sin(2 * np.pi * 3 * t / N)
    best_h = _project_to_feasible(best_h)
    best_score = _max_overlap(best_h)
    
    # Initial t value
    t_init = best_score + 0.1
    
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds_t = [(0.0, 10.0)]
    bounds = bounds_h + bounds_t
    
    # Phase 1: Coarse optimization with moderate beta
    beta_schedule = [10, 30, 100, 300]
    
    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break
        
        # Build initial x vector
        x_init = np.concatenate([best_h, [t_init]])
        
        def obj_func(x):
            return _objective_epigraph(x, beta, 50.0)
        
        try:
            result = minimize(
                obj_func, x_init, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
                t_init = result.x[N]
        except Exception:
            pass
    
    # Final precision polish with very high beta
    if time.time() < deadline - 10.0:
        x_init = np.concatenate([best_h, [t_init]])
        
        def obj_polish(x):
            return _objective_epigraph(x, 5000.0, 5000.0)
        
        try:
            result = minimize(
                obj_polish, x_init, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
            )
            h_polished = _project_to_feasible(result.x[:N])
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # Ensure final validity
    best_h = _project_to_feasible(best_h)
    
    # Final precision polish: ensure sum is exactly N/2
    sum_h = np.sum(best_h)
    diff = TARGET_SUM - sum_h
    if abs(diff) > 1e-12:
        if best_h[0] + diff <= 1.0 and best_h[0] + diff >= 0.0:
            best_h[0] += diff
        else:
            best_h = best_h * (TARGET_SUM / sum_h)
            best_h = np.clip(best_h, 0.0, 1.0)
            best_h = _project_to_feasible(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
