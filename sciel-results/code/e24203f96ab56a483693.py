import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_linear_overlaps(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
    Returns an array of overlap values C_k for k = 1..N-1 (both positive and negative).
    C_k = sum_i h_i * (1 - h_{i+k})  (with appropriate truncation)
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # c has length 2N-1. c[N-1] is the zero-shift correlation.
    # We want shifts k != 0.
    # Negative shifts: c[0 : N-1]
    # Positive shifts: c[N : 2N-1]
    c_neg = c[:N-1]
    c_pos = c[N:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _compute_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return np.max(overlaps)

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses iterative water-filling style projection.
    """
    h = np.clip(h, 0, 1).copy()
    target = N / 2.0
    current = np.sum(h)
    
    if abs(current - target) < 1e-12:
        return h
    
    for _ in range(200):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        
        if diff > 0:
            mask = h < 1.0
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] += step
            h = np.clip(h, 0, 1)
        else:
            mask = h > 0.0
            if not np.any(mask):
                break
            step = (-diff) / np.sum(mask)
            h[mask] -= step
            h = np.clip(h, 0, 1)
    
    return h

def _objective(h, beta, lam_sum, lam_var):
    """
    Objective function:
    1. LogSumExp approximation of max overlap (smooth max).
    2. Variance of overlaps (to flatten the correlation profile).
    3. Sum constraint penalty.
    """
    h = np.clip(h, 0, 1)
    overlaps = _compute_linear_overlaps(h)
    
    if len(overlaps) == 0:
        max_ov = 0.0
        var_ov = 0.0
    else:
        max_ov = np.max(overlaps)
        var_ov = np.var(overlaps)
    
    # Clip for numerical stability in exp
    if max_ov > 500:
        max_ov = 500
    
    if beta > 0:
        shifted = overlaps - max_ov
        # LSE approximation of max
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        lse = max_ov
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    var_penalty = lam_var * var_ov
    
    return lse + var_penalty + sum_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initial best: uniform
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    candidates.append(np.full(N, 0.5))
    
    # Sinusoidal with various parameters - these often have good autocorrelation properties
    t = np.arange(N)
    for freq in range(1, 20):
        for amp in [0.1, 0.2, 0.3, 0.4]:
            for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(trial.copy())
    
    # Random
    for i in range(50):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
        
    # Evaluate all candidates
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # Optimization phases with increasing beta
    # Beta controls the sharpness of the max approximation.
    # Low beta: smooth, convex, easy to optimize.
    # High beta: sharp, closer to true max, harder to optimize.
    beta_schedule = [5, 10, 20, 50, 100, 200, 400, 800]
    
    # We also tune the variance penalty. High variance penalty encourages flat correlations.
    # We start with higher variance penalty to find a good shape, then reduce it to focus on the peak.
    var_penalty_schedule = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]
    
    for i, beta in enumerate(beta_schedule):
        if time.time() > deadline - 10.0:
            break
        
        lam_var = var_penalty_schedule[i]
        lam_sum = 100.0
        
        def obj(x):
            return _objective(x, beta, lam_sum, lam_var)
        
        # Try from best solution
        x0 = best_h.copy()
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B', 
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        # Random restarts to escape local minima
        num_restarts = 5 if beta < 50 else 2
        for _ in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Diverse random initialization
            if rng.random() < 0.5:
                # Sinusoidal random
                freq = rng.integers(1, 30)
                amp = rng.uniform(0.1, 0.4)
                phase = rng.uniform(0, 2*np.pi)
                x0 = 0.5 + amp * np.sin(2*np.pi*freq*np.arange(N)/N + phase)
            else:
                # Random
                x0 = rng.uniform(0.0, 1.0, size=N)
            
            x0 = _project_to_feasible(x0)
            
            try:
                result = minimize(
                    obj, x0, method='L-BFGS-B', 
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_new = _project_to_feasible(result.x)
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    # Final polish with very high beta and no variance penalty (focus purely on min max)
    if time.time() < deadline - 5.0:
        def obj_polish(x):
            return _objective(x, 1000.0, 1000.0, 0.0)
        
        try:
            result = minimize(
                obj_polish, best_h, method='L-BFGS-B', 
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_polished = _project_to_feasible(result.x)
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # Ensure final validity
    best_h = _project_to_feasible(best_h)
    
    return best_h
