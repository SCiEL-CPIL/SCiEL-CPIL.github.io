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
    
    for _ in range(100):
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

def _objective_lse(x, beta, lam_sum):
    """
    Objective function using LogSumExp approximation of the max overlap.
    """
    h = np.clip(x, 0, 1)
    overlaps = _compute_linear_overlaps(h)
    
    if len(overlaps) == 0:
        max_ov = 0.0
    else:
        max_ov = np.max(overlaps)
    
    if max_ov > 500:
        max_ov = 500
    if beta > 0:
        shifted = overlaps - max_ov
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        lse = max_ov
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def _objective_epigraph(x, beta, lam_sum):
    """
    Epigraph formulation: minimize t + penalty
    subject to all overlaps <= t (smoothed)
    """
    h = np.clip(x[:N], 0, 1)
    t = x[N]
    overlaps = _compute_linear_overlaps(h)
    
    if len(overlaps) == 0:
        return t + lam_sum * (np.sum(h) - N / 2.0) ** 2
    
    # Smooth constraint: sum exp(beta*(overlap - t)) should be small
    shifted = overlaps - t
    # Avoid overflow
    max_shifted = np.max(shifted)
    if max_shifted > 50:
        # If t is too small, penalize heavily
        return t + 1000.0 * max_shifted + lam_sum * (np.sum(h) - N / 2.0) ** 2
    
    # Smooth max approximation: t + (1/beta)*log(sum(exp(beta*(overlap-t))))
    # But we want to minimize t such that all overlaps <= t
    # So minimize t + (1/beta)*log(sum(exp(beta*(overlap-t))))
    constraint_term = np.log(np.sum(np.exp(beta * shifted))) / beta
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    
    return t + constraint_term + sum_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds_full = bounds_h + [(-10.0, 10.0)]  # t can be negative initially
    
    rng = np.random.default_rng(123)
    
    # Initial best: uniform
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    candidates.append(np.full(N, 0.5))
    
    # Alternating patterns
    for offset in range(4):
        h_alt = np.zeros(N)
        h_alt[offset::4] = 0.7
        h_alt[(offset+1)::4] = 0.3
        h_alt[(offset+2)::4] = 0.5
        h_alt[(offset+3)::4] = 0.5
        candidates.append(h_alt)
    
    # Sinusoidal with various parameters
    t = np.arange(N)
    for freq in range(1, 30):
        for amp in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            for phase in [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(trial.copy())
    
    # Random
    for i in range(100):
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
    beta_schedule = [5, 10, 20, 50, 100, 200, 400]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        # Try from best solution using LSE
        x0 = best_h.copy()
        
        def obj_lse(x):
            return _objective_lse(x, beta, 100.0)
        
        try:
            result = minimize(
                obj_lse, x0, method='L-BFGS-B', bounds=bounds_h,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        # Try epigraph formulation from best solution
        x0_ep = np.concatenate([best_h, [best_score]])
        
        def obj_ep(x):
            return _objective_epigraph(x, beta, 100.0)
        
        try:
            result = minimize(
                obj_ep, x0_ep, method='L-BFGS-B', bounds=bounds_full,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        # Random restarts
        num_restarts = 8 if beta <= 50 else 5
        for _ in range(num_restarts):
            if time.time() > deadline - 20.0:
                break
            
            # Diverse random initialization
            if rng.random() < 0.3:
                # Sinusoidal random
                freq = rng.integers(1, 50)
                amp = rng.uniform(0.05, 0.4)
                phase = rng.uniform(0, 2*np.pi)
                x0 = 0.5 + amp * np.sin(2*np.pi*freq*np.arange(N)/N + phase)
            elif rng.random() < 0.6:
                # Random
                x0 = rng.uniform(0.0, 1.0, size=N)
            else:
                # Perturb best
                x0 = best_h + rng.normal(0, 0.1, size=N)
            
            x0 = _project_to_feasible(x0)
            
            def obj_rand(x):
                return _objective_lse(x, beta, 100.0)
            
            try:
                result = minimize(
                    obj_rand, x0, method='L-BFGS-B', bounds=bounds_h,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_new = _project_to_feasible(result.x)
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    # Final polish with high beta
    if time.time() < deadline - 10.0:
        # LSE polish
        def obj_polish_lse(x):
            return _objective_lse(x, 500.0, 1000.0)
        
        try:
            result = minimize(
                obj_polish_lse, best_h, method='L-BFGS-B', bounds=bounds_h,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_polished = _project_to_feasible(result.x)
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
        
        # Epigraph polish
        if time.time() < deadline - 5.0:
            x0_ep = np.concatenate([best_h, [best_score]])
            
            def obj_polish_ep(x):
                return _objective_epigraph(x, 500.0, 1000.0)
            
            try:
                result = minimize(
                    obj_polish_ep, x0_ep, method='L-BFGS-B', bounds=bounds_full,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
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
    
    return best_h
