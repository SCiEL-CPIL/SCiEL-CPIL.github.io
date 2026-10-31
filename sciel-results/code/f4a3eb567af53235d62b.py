"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar

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
    """
    target_sum = N / 2.0
    lo = -10.0
    hi = float(N) + 10.0
    
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

def _evaluate_params(w, c):
    """
    Evaluate the objective for given w and c.
    Returns (overlap, h)
    """
    h = _construct_h(w, c)
    current_sum = np.sum(h)
    if abs(current_sum - N/2.0) > 1e-6:
        diff = N/2.0 - current_sum
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
    ov = _max_overlap(h)
    return ov, h

def _objective_1d(w_log):
    """
    Objective function for 1D optimization of log(w).
    """
    w = np.exp(w_log)
    if w < 1e-3:
        w = 1e-3
    if w > N:
        w = N
        
    c = _solve_c(w)
    ov, _ = _evaluate_params(w, c)
    return ov

def construct():
    """
    Construct the sequence of step heights.
    """
    # Use differential evolution for global search over log(w)
    # Bounds for log(w): w in [0.01, 100] -> log(w) in [-4.6, 4.6]
    bounds = [(-4.6, 4.6)]
    
    # Differential evolution
    res_de = differential_evolution(
        _objective_1d, 
        bounds=bounds, 
        seed=42, 
        maxiter=100, 
        popsize=15,
        tol=1e-5
    )
    
    best_w_log = res_de.x[0]
    best_score = res_de.fun
    
    # Refine with minimize_scalar
    lower_bound = max(-4.6, best_w_log - 0.5)
    upper_bound = min(4.6, best_w_log + 0.5)
    
    res_refine = minimize_scalar(
        _objective_1d, 
        bounds=(lower_bound, upper_bound), 
        method='bounded', 
        options={'xatol': 1e-6}
    )
    
    if res_refine.fun < best_score:
        best_score = res_refine.fun
        best_w_log = res_refine.x
    
    # Final construction
    best_w = np.exp(best_w_log)
    c = _solve_c(best_w)
    
    h = _construct_h(best_w, c)
    
    # Final correction for exact sum
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    
    if abs(diff) > 1e-10:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
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

    return h
