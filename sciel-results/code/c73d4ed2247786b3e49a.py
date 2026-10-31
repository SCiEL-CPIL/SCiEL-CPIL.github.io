"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h for all cyclic shifts."""
    n = len(h)
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for shift in range(n):
        shifted = np.roll(one_minus_h, shift)
        overlap = np.sum(h * shifted)
        if overlap > max_ov:
            max_ov = overlap
    return max_ov


def _score(h):
    """The objective to minimize: maximum overlap."""
    return _max_overlap(h)


def _project_to_valid(h):
    """Project h to satisfy constraints: sum = N/2, each in [0, 1]."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to make sum exactly N/2
    target_sum = N / 2.0
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    # If clipping changed the sum, do a second pass
    diff2 = target_sum - np.sum(h)
    if abs(diff2) > 1e-12:
        # Distribute remaining difference among elements not at bounds
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h[mask] += diff2 / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with the uniform solution
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Try periodic patterns
    for period in range(1, N + 1):
        if time.time() > deadline:
            break
        if N % period != 0:
            continue
        n_reps = N // period
        # Optimize one period
        x0 = np.full(period, 0.5)
        
        def objective(x):
            h = np.tile(x, n_reps)
            return _max_overlap(h)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            h_candidate = np.tile(result.x, n_reps)
            h_candidate = _project_to_valid(h_candidate)
            score_candidate = _score(h_candidate)
            if score_candidate < best_score:
                best_score = score_candidate
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Try non-periodic optimization with a few random restarts
    np.random.seed(42)
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial point
        h0 = np.random.uniform(0, 1, N)
        h0 = _project_to_valid(h0)
        
        def objective(x):
            h = _project_to_valid(x)
            return _max_overlap(h)
        
        try:
            result = minimize(objective, h0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
            h_candidate = _project_to_valid(result.x)
            score_candidate = _score(h_candidate)
            if score_candidate < best_score:
                best_score = score_candidate
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        def objective(x):
            h = _project_to_valid(x)
            return _max_overlap(h)
        
        try:
            result = minimize(objective, best_h, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            h_candidate = _project_to_valid(result.x)
            score_candidate = _score(h_candidate)
            if score_candidate < best_score:
                best_score = score_candidate
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
