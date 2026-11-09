"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 120.0  # seconds
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _deadline_ok(margin=5.0):
    return _elapsed() < _DEADLINE - margin

def _compute_overlap(h):
    """
    Compute the overlap profile between h and shifted (1-h).
    For a step function h on [0,2] with N steps of width 2/N,
    the overlap at shift s is sum_i h_i * (1 - h_{i+s}) for valid indices.
    We want to minimize max_s overlap(s).
    """
    n = len(h)
    max_overlap = 0.0
    # Overlap at shift k (in steps): sum over i of h[i] * (1 - h[(i+k) % n])
    # But we need to be careful about the shift interpretation.
    # For shift by k steps, overlap = sum_i h[i] * (1 - h[(i+k) % n])
    # We consider all shifts k = 1, 2, ..., n-1 (shift by 0 is trivial)
    for k in range(1, n):
        shifted_h = np.roll(h, -k)
        overlap = np.sum(h * (1.0 - shifted_h))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _smooth_max_overlap(h, beta=10.0):
    """
    Smooth approximation of max overlap using log-sum-exp.
    overlap_k = sum_i h[i] * (1 - h[(i+k) % n])
    smooth_max ≈ (1/beta) * log(sum_k exp(beta * overlap_k))
    """
    n = len(h)
    overlaps = np.zeros(n - 1)
    for k in range(1, n):
        shifted_h = np.roll(h, -k)
        overlaps[k-1] = np.sum(h * (1.0 - shifted_h))
    
    # Log-sum-exp for numerical stability
    max_ov = np.max(overlaps)
    if max_ov < -1000:
        return 0.0
    log_sum = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov))))
    return log_sum / beta

def _objective_and_grad(x, beta=10.0):
    """
    Objective: smooth max overlap + penalty for violating constraints.
    x: 2N vector, first N are h values, last N are slack or just use first N.
    Actually let's just use h directly and handle constraints via projection or penalty.
    """
    n = N
    h = x[:n]
    
    # Penalty for sum constraint: sum(h) should be n/2
    sum_penalty = 1000.0 * (np.sum(h) - n / 2.0) ** 2
    
    # Penalty for bounds: h in [0, 1]
    bound_penalty = 1000.0 * np.sum(np.maximum(0, -h) ** 2 + np.maximum(0, h - 1) ** 2)
    
    obj = _smooth_max_overlap(h, beta) + sum_penalty + bound_penalty
    
    return obj

def _fix_sum(h):
    """Adjust h so that sum(h) = N/2, keeping values in [0,1]."""
    n = len(h)
    target = n / 2.0
    current_sum = np.sum(h)
    diff = current_sum - target
    if abs(diff) < 1e-12:
        return h.copy()
    # Distribute correction evenly, then clip and redistribute
    h = h - diff / n
    h = np.clip(h, 0, 1)
    # Iterate to fix sum after clipping
    for _ in range(100):
        s = np.sum(h)
        if abs(s - target) < 1e-12:
            break
        diff = s - target
        # Adjust only non-saturated values
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if not np.any(mask):
            break
        h[mask] -= diff / np.sum(mask)
        h = np.clip(h, 0, 1)
    return h

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start from the seed solution
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _compute_overlap(h)
    
    # Use smooth objective with increasing beta (annealing)
    betas = [2.0, 5.0, 10.0, 20.0, 50.0]
    
    for beta in betas:
        if not _deadline_ok(margin=10.0):
            break
        
        # Try multiple restarts with different initializations
        for restart in range(3):
            if not _deadline_ok(margin=10.0):
                break
            
            if restart == 0:
                h_init = best_h.copy()
            else:
                # Add small random perturbation
                h_init = best_h + np.random.normal(0, 0.1, N)
                h_init = np.clip(h_init, 0, 1)
                h_init = _fix_sum(h_init)
            
            # Use L-BFGS-B with bounds
            bounds = [(0, 1)] * N
            result = minimize(
                _objective_and_grad,
                h_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_new = result.x
            h_new = _fix_sum(h_new)
            
            score = _compute_overlap(h_new)
            if score < best_score:
                best_score = score
                best_h = h_new.copy()
    
    # Final polish: exact local search on the best solution
    if _deadline_ok(margin=5.0):
        # Try pairwise swaps and small adjustments
        improved = True
        while improved and _deadline_ok(margin=5.0):
            improved = False
            h = best_h.copy()
            for i in range(N):
                if not _deadline_ok(margin=5.0):
                    break
                for j in range(i + 1, N):
                    if not _deadline_ok(margin=5.0):
                        break
                    # Try transferring mass from i to j
                    for delta in [0.01, 0.05, 0.1, -0.01, -0.05, -0.1]:
                        h_trial = h.copy()
                        h_trial[i] += delta
                        h_trial[j] -= delta
                        if 0 <= h_trial[i] <= 1 and 0 <= h_trial[j] <= 1:
                            score = _compute_overlap(h_trial)
                            if score < best_score - 1e-12:
                                best_score = score
                                best_h = h_trial.copy()
                                improved = True
    
    return best_h
# EVOLVE-BLOCK-END
