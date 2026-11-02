"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_overlaps(h):
    """Compute all overlaps of h with shifted (1-h)."""
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        shifted_1mh = 1.0 - np.roll(h, -k)
        overlaps[k] = np.dot(h, shifted_1mh)
    return overlaps

def _max_overlap(h):
    return np.max(_compute_overlaps(h))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: Low frequency cosine to encourage positive autocorrelation
    x = np.linspace(0, 1, N, endpoint=False)
    # Target variance ~ 0.05, so amplitude ~ sqrt(0.1)
    A = 0.316
    best_h = 0.5 + A * np.cos(2 * np.pi * x)
    best_h = np.clip(best_h, 0, 1)
    
    # Normalize to sum exactly N/2
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h * (N / 2.0) / s
    best_h = np.clip(best_h, 0, 1)
    
    # Correct sum again
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.where(best_h < 1.0)[0]
        else:
            idx = np.where(best_h > 0.0)[0]
        if len(idx) > 0:
            best_h[idx[0]] += diff
            best_h = np.clip(best_h, 0, 1)

    best_val = _max_overlap(best_h)
    
    def soft_max_objective(h):
        overlaps = _compute_overlaps(h)
        beta = 50.0
        m = np.max(overlaps)
        soft_max_val = m + np.log(np.sum(np.exp((overlaps - m) * beta)) / N) / beta
        
        penalty = 0.0
        s = np.sum(h)
        penalty += 1000.0 * (s - N / 2.0) ** 2
        penalty += 100.0 * np.sum(np.maximum(0, h - 1.0) ** 2)
        penalty += 100.0 * np.sum(np.maximum(0, -h) ** 2)
        
        return soft_max_val + penalty

    bounds = [(0.0, 1.0)] * N
    
    # Optimization
    h_opt = best_h.copy()
    try:
        result = minimize(
            soft_max_objective,
            h_opt,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if time.time() < deadline - 20.0:
            h_candidate = np.clip(result.x, 0, 1)
            s = np.sum(h_candidate)
            if s > 0:
                h_candidate = h_candidate * (N / 2.0) / s
                h_candidate = np.clip(h_candidate, 0, 1)
            
            diff = N / 2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-12:
                if diff > 0:
                    idx = np.where(h_candidate < 1.0)[0]
                else:
                    idx = np.where(h_candidate > 0.0)[0]
                if len(idx) > 0:
                    h_candidate[idx[0]] += diff
                    h_candidate = np.clip(h_candidate, 0, 1)
            
            val = _max_overlap(h_candidate)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_candidate.copy()
    except Exception:
        pass

    # Local search polish
    if time.time() < deadline - 10.0:
        np.random.seed(42)
        for trial in range(50):
            if time.time() > deadline - 5.0:
                break
            h = best_h + np.random.randn(N) * 0.01
            h = np.clip(h, 0, 1)
            s = np.sum(h)
            if s > 0:
                h = h * (N / 2.0) / s
                h = np.clip(h, 0, 1)
            diff = N / 2.0 - np.sum(h)
            if diff > 0:
                idx = np.where(h < 1.0)[0]
                if len(idx) > 0:
                    h[idx] += diff / len(idx)
            else:
                idx = np.where(h > 0.0)[0]
                if len(idx) > 0:
                    h[idx] += diff / len(idx)
            
            val = _max_overlap(h)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h.copy()

    # Final cleanup
    best_h = np.clip(best_h, 0, 1)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h * (N / 2.0) / s
        best_h = np.clip(best_h, 0, 1)
    
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.where(best_h < 1.0)[0]
        else:
            idx = np.where(best_h > 0.0)[0]
        if len(idx) > 0:
            best_h[idx[0]] += diff
            best_h = np.clip(best_h, 0, 1)

    return best_h
# EVOLVE-BLOCK-END
