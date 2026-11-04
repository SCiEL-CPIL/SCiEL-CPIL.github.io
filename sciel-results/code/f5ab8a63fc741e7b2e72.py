"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

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
    return float(np.max(overlaps))

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses bisection on a uniform shift for exact sum.
    """
    h = np.clip(h, 0.0, 1.0).copy()
    target = TARGET_SUM
    current = np.sum(h)
    
    if abs(current - target) < 1e-14:
        return h
    
    # Bisection to find delta such that sum(clip(h + delta, 0, 1)) = target
    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        h_test = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_test)
        if s < target:
            lo = mid
        else:
            hi = mid
        if abs(s - target) < 1e-14:
            break
    
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    # Final exact correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        # Distribute remaining diff to non-bound elements
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    
    return h

def _objective_epigraph(x, beta, lam_sum):
    """
    Epigraph objective. x = [h_0..h_{N-1}, t].
    Minimize t + penalty for overlaps > t + sum penalty.
    Uses log-sum-exp for smooth approximation of max.
    """
    h = np.clip(x[:N], 0.0, 1.0)
    t = x[N]
    
    overlaps = _get_linear_overlaps(h)
    
    if len(overlaps) == 0:
        lse = 0.0
    else:
        max_ov = np.max(overlaps)
        if max_ov > 1000:
            max_ov = 1000
        shifted = overlaps - max_ov
        # Clip to prevent overflow
        shifted = np.clip(shifted, -50.0 / beta, 0.0)
        lse = max_ov + logsumexp(beta * shifted) / beta
    
    # Penalty if lse > t
    if lse > t:
        penalty = (lse - t) ** 2
    else:
        penalty = 0.0
    
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return t + penalty + sum_penalty

def _coordinate_descent_polish(h, max_iter=2000, tol=1e-10, deadline=None):
    """
    Coordinate descent local search to polish the solution.
    For each coordinate, try small perturbations and keep if it reduces max overlap.
    """
    h = h.copy()
    best_score = _max_overlap(h)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        improved = False
        for i in range(N):
            # Try small perturbations around current value
            base = h[i]
            for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                new_val = base + delta
                if 0.0 <= new_val <= 1.0:
                    h[i] = new_val
                    h = _project_to_feasible(h)
                    score = _max_overlap(h)
                    if score < best_score - tol:
                        best_score = score
                        improved = True
                        break
            h[i] = base  # restore
        
        if not improved:
            # Try smaller steps
            for i in range(N):
                base = h[i]
                for delta in [0.001, -0.001, 0.0005, -0.0005, 0.0001, -0.0001]:
                    new_val = base + delta
                    if 0.0 <= new_val <= 1.0:
                        h[i] = new_val
                        h = _project_to_feasible(h)
                        score = _max_overlap(h)
                        if score < best_score - tol:
                            best_score = score
                            improved = True
                            break
                h[i] = base
            
            if not improved:
                break
    
    return h, best_score

def _generate_chaotic_candidate(rng):
    """
    Generate a candidate using logistic map dynamics.
    h_i = 0.5 + A * (x_i - 0.5) where x_i is logistic map output.
    Optimize r (growth rate) and A (amplitude) to minimize overlap.
    """
    best_h = None
    best_score = float('inf')
    
    # Try various logistic map parameters
    for r in [3.2, 3.5, 3.8, 3.9, 3.99, 3.999]:
        for A in [0.3, 0.5, 0.7, 0.9, 1.0]:
            x = 0.5
            h = np.zeros(N)
            for i in range(N):
                x = r * x * (1 - x)
                h[i] = 0.5 + A * (x - 0.5)
            
            h = _project_to_feasible(h)
            score = _max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    return best_h, best_score

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Uses epigraph formulation with log-sum-exp smooth approximation,
    multiple diverse initializations including chaotic sequences,
    and coordinate descent polish.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Start from multiple structured initial guesses
    best_h = None
    best_score = float('inf')
    
    # Initial guess 1: Sturmian / Quasi-periodic (Golden Ratio)
    alpha = (np.sqrt(5.0) - 1.0) / 2.0
    t = np.arange(N)
    h_init = np.zeros(N)
    for i in range(N):
        if (i * alpha) % 1.0 < 0.5:
            h_init[i] = 1.0
        else:
            h_init[i] = 0.0
    h_init = _project_to_feasible(h_init)
    score = _max_overlap(h_init)
    if score < best_score:
        best_score = score
        best_h = h_init.copy()
        
    # Initial guess 2: Low-frequency sinusoidal (exploits positive autocorrelation)
    h_init2 = 0.5 + 0.3 * np.sin(2 * np.pi * 1 * t / N)
    h_init2 = _project_to_feasible(h_init2)
    score2 = _max_overlap(h_init2)
    if score2 < best_score:
        best_score = score2
        best_h = h_init2.copy()

    # Initial guess 3: Higher frequency sinusoidal
    h_init3 = 0.5 + 0.3 * np.sin(2 * np.pi * 7 * t / N)
    h_init3 = _project_to_feasible(h_init3)
    score3 = _max_overlap(h_init3)
    if score3 < best_score:
        best_score = score3
        best_h = h_init3.copy()
        
    # Initial guess 4: Random Binary
    h_init4 = rng.choice([0.0, 1.0], size=N)
    h_init4 = _project_to_feasible(h_init4)
    score4 = _max_overlap(h_init4)
    if score4 < best_score:
        best_score = score4
        best_h = h_init4.copy()
    
    # Initial guess 5: Chaotic logistic map
    if time.time() < deadline - 30.0:
        h_chaotic, score_chaotic = _generate_chaotic_candidate(rng)
        if h_chaotic is not None and score_chaotic < best_score:
            best_score = score_chaotic
            best_h = h_chaotic
    
    # Optimization Loop with increasing beta (LSE sharpness)
    # Low beta = smoother, easier to escape local minima
    # High beta = sharper, better approximation of true max
    beta_schedule = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    bounds = [(0.0, 1.0)] * N + [(0.0, 10.0)]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        t_init = best_score + 0.1
        x_init = np.concatenate([best_h, [t_init]])
        lam_sum = 50.0 + 10.0 * beta
        
        def obj_func(x, b=beta, ls=lam_sum):
            return _objective_epigraph(x, b, ls)
        
        try:
            result = minimize(
                obj_func, x_init, method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Final Polish: Coordinate Descent
    if time.time() < deadline - 5.0:
        h_cd, score_cd = _coordinate_descent_polish(best_h, max_iter=1000, tol=1e-10, deadline=deadline)
        if score_cd < best_score:
            best_score = score_cd
            best_h = h_cd
    
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
