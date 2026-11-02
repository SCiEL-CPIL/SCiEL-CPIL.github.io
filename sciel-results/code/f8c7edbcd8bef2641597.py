"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _time_remaining(deadline):
    return deadline - time.time()


def _compute_overlaps(h):
    """Compute the overlap of h with shifted 1-h for all shifts."""
    n = len(h)
    overlaps = np.zeros(n)
    for k in range(n):
        # Shift by k
        h_shifted = np.roll(h, k)
        one_minus_h = 1.0 - h
        overlap = np.sum(np.minimum(h_shifted, one_minus_h))
        overlaps[k] = overlap
    return overlaps


def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    return np.max(_compute_overlaps(h))


def _max_overlap_subgrad(h):
    """Compute the maximum overlap and a subgradient."""
    n = len(h)
    overlaps = _compute_overlaps(h)
    max_idx = np.argmax(overlaps)
    max_val = overlaps[max_idx]
    
    # Compute subgradient for the max shift
    h_shifted = np.roll(h, max_idx)
    one_minus_h = 1.0 - h
    
    # Subgradient of min(a,b) is 1 if a<b, 0 if a>b, [0,1] if a==b
    subgrad = np.zeros(n)
    for i in range(n):
        if h_shifted[i] < one_minus_h[i]:
            subgrad[i] = 1.0
        elif h_shifted[i] > one_minus_h[i]:
            subgrad[i] = 0.0
        else:
            subgrad[i] = 0.5
    
    # The overlap is sum(min(h_shifted, 1-h))
    # d/dh_j of min(h_shifted[i], 1-h[i]) for each i
    # h_shifted[i] = h[(i - max_idx) % n]
    # So d/dh_j of min(h_shifted[i], 1-h[i]) = 1 if h_shifted[i] < 1-h[i] and j == (i - max_idx) % n
    # = -1 if h_shifted[i] > 1-h[i] and j == i
    # = [0,1] if equal
    
    grad = np.zeros(n)
    for i in range(n):
        if h_shifted[i] < one_minus_h[i]:
            grad[(i - max_idx) % n] += 1.0
        elif h_shifted[i] > one_minus_h[i]:
            grad[i] -= 1.0
        else:
            grad[(i - max_idx) % n] += 0.5
            grad[i] -= 0.5
    
    return max_val, grad


def _project_to_simplex(v, target_sum):
    """Project v onto {x: 0 <= x_i <= 1, sum(x) = target_sum}."""
    n = len(v)
    # Use water-filling / bisection
    def _sum_clipped(v, lam):
        return np.sum(np.clip(v - lam, 0, 1))
    
    lo, hi = -10, 10
    for _ in range(100):
        mid = (lo + hi) / 2
        s = _sum_clipped(v, mid)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2
    return np.clip(v - lam, 0, 1)


def construct():
    deadline = time.time() + 110  # Leave 10s margin
    
    # Start with a random initialization
    rng = np.random.default_rng(42)
    h = rng.uniform(0, 1, N)
    h = _project_to_simplex(h, N / 2)
    
    best_h = h.copy()
    best_score = _max_overlap(h)
    
    # Gradient descent with subgradient
    lr = 0.01
    for step in range(5000):
        if time.time() > deadline:
            break
        
        val, grad = _max_overlap_subgrad(h)
        
        # Update
        h_new = h - lr * grad
        h_new = _project_to_simplex(h_new, N / 2)
        
        # Check if improved
        new_score = _max_overlap(h_new)
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
        
        h = h_new
        lr *= 0.999
    
    # Final projection to ensure exact sum
    best_h = _project_to_simplex(best_h, N / 2)
    
    return best_h
# EVOLVE-BLOCK-END
