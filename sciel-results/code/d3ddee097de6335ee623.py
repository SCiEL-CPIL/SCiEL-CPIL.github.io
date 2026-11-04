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
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        lse = max_ov
    
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return lse + sum_penalty

def _generate_h_from_theta(theta):
    """
    Generate h from parameters theta.
    theta = [A1, f1, p1, A2, f2, p2, A3, f3, p3]
    h_i = 0.5 + sum(A_k * sin(2*pi*f_k*i/N + p_k))
    Then project to feasible set.
    """
    t = np.arange(N)
    h = np.full(N, 0.5)
    
    # 3 sinusoids
    for k in range(3):
        A = theta[k*3]
        f = theta[k*3 + 1]
        p = theta[k*3 + 2]
        h += A * np.sin(2 * np.pi * f * t / N + p)
        
    h = _project_to_feasible(h)
    return h

def _objective_theta(theta):
    """
    Objective for low-dimensional parameter optimization.
    Uses a moderate beta for LSE to be smooth enough for derivative-free or simple gradient methods.
    """
    h = _generate_h_from_theta(theta)
    # Use a fixed beta for this stage, e.g., 50
    return _objective_lse(h, 50.0, 0.0)

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Strategy:
    1. Low-dimensional search over sinusoidal parameters to find a good structured starting point.
    2. Full-dimensional L-BFGS-B refinement from that point.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Low-dimensional search
    best_h_low_dim = None
    best_score_low_dim = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Parameters: [A1, f1, p1, A2, f2, p2, A3, f3, p3]
    # A in [0, 0.4], f in [0.5, 20], p in [0, 2*pi]
    bounds_theta = [
        (0, 0.4), (0.5, 20), (0, 2*np.pi),
        (0, 0.4), (0.5, 20), (0, 2*np.pi),
        (0, 0.4), (0.5, 20), (0, 2*np.pi)
    ]
    
    # Try multiple random starts for the low-dim optimization
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline - 30.0:
            break
            
        # Random start
        theta0 = np.array([
            rng.uniform(0, 0.3), rng.uniform(1, 10), rng.uniform(0, 2*np.pi),
            rng.uniform(0, 0.3), rng.uniform(1, 10), rng.uniform(0, 2*np.pi),
            rng.uniform(0, 0.3), rng.uniform(1, 10), rng.uniform(0, 2*np.pi)
        ])
        
        try:
            # Use Powell or Nelder-Mead for robustness to non-smooth projection
            res = minimize(
                _objective_theta, 
                theta0, 
                method='Powell', 
                bounds=bounds_theta,
                options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            h_cand = _generate_h_from_theta(res.x)
            score_cand = _max_overlap(h_cand)
            
            if score_cand < best_score_low_dim:
                best_score_low_dim = score_cand
                best_h_low_dim = h_cand
        except Exception:
            pass

    # If low-dim search failed, fallback to simple sinusoid
    if best_h_low_dim is None:
        best_h_low_dim = _project_to_feasible(0.5 + 0.3 * np.sin(2 * np.pi * 5 * np.arange(N) / N))
        best_score_low_dim = _max_overlap(best_h_low_dim)

    # 2. Full-dimensional refinement
    best_h = best_h_low_dim.copy()
    best_score = best_score_low_dim
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Refinement phases with increasing beta
    beta_schedule = [20, 50, 100, 200, 500]
    
    for beta in beta_schedule:
        if time.time() > deadline - 10.0:
            break
        
        def obj_func(x):
            return _objective_lse(x, beta, 100.0)
        
        try:
            result = minimize(
                obj_func, best_h, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass

    # Final precision polish
    if time.time() < deadline - 5.0:
        def obj_polish(x):
            return _objective_lse(x, 1000.0, 1000.0)
        
        try:
            result = minimize(
                obj_polish, best_h, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
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
