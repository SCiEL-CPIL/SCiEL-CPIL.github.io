"""
Simulated annealing with binary swaps to minimize maximum overlap
of h with shifted 1-h. Exploits combinatorial structure of binary-like
solutions to escape the uniform local optimum.
"""

import numpy as np
import time

N = 100
DEADLINE = 120.0
SAFETY_MARGIN = 5.0


def _max_overlap(h):
    """Compute the maximum overlap C_k = sum_i h_i * (1 - h_{i+k}) for all shifts k."""
    n = len(h)
    max_c = 0.0
    for k in range(1, n):
        c = np.dot(h[:n-k], 1.0 - h[k:])
        if c > max_c:
            max_c = c
    return max_c


def _swap_pair_indices(h, target_sum):
    """Find two indices i, j such that swapping h[i] and h[j] keeps sum = target_sum."""
    # For binary-like solutions, we want to swap a 0-ish with a 1-ish
    # More generally, find pairs where h[i] + h[j] = h[j] + h[i] (always true)
    # Actually we need: after swap, sum remains target_sum
    # sum - h[i] - h[j] + h[j] + h[i] = sum, so any swap preserves sum!
    # So we just pick two random distinct indices
    i = np.random.randint(0, len(h))
    j = np.random.randint(0, len(h))
    while j == i:
        j = np.random.randint(0, len(h))
    return i, j


def _try_swap(h, i, j):
    """Attempt swap of h[i] and h[j], return new value if valid."""
    new_i = h[j]
    new_j = h[i]
    if 0.0 <= new_i <= 1.0 and 0.0 <= new_j <= 1.0:
        return True
    return False


def construct():
    """
    Use simulated annealing with binary swaps to minimize max overlap.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    np.random.seed(42)
    
    # Start with uniform solution
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _max_overlap(h)
    
    # Initialize with some binary-like structure to help SA escape
    # Create a random binary partition with exactly N/2 ones
    h_binary = np.zeros(N)
    ones = np.random.choice(N, N // 2, replace=False)
    h_binary[ones] = 1.0
    h = h_binary.copy()
    best_h = h.copy()
    best_score = _max_overlap(h)
    
    # Simulated annealing
    T = 1.0
    cooling = 0.999
    min_T = 1e-6
    
    while time.time() < deadline and T > min_T:
        # Pick a random pair to swap
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        while j == i:
            j = np.random.randint(0, N)
        
        # Compute overlap before swap
        old_score = _max_overlap(h)
        
        # Perform swap
        h[i], h[j] = h[j], h[i]
        
        # Compute overlap after swap
        new_score = _max_overlap(h)
        
        delta = new_score - old_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Reject: revert
            h[i], h[j] = h[j], h[i]
        
        T *= cooling
    
    # Local search polish: try all single-element adjustments
    while time.time() < deadline:
        improved = False
        for i in range(N):
            for delta in [-0.01, 0.01, -0.05, 0.05, -0.1, 0.1]:
                old_val = h[i]
                new_val = old_val + delta
                if 0.0 <= new_val <= 1.0:
                    h[i] = new_val
                    # Adjust another element to maintain sum
                    # Find j != i and adjust h[j] by -delta
                    j = (i + 1) % N
                    while j == i:
                        j = (j + 1) % N
                    adj_val = h[j] - delta
                    if 0.0 <= adj_val <= 1.0:
                        h[j] = adj_val
                        new_score = _max_overlap(h)
                        if new_score < best_score:
                            best_score = new_score
                            best_h = h.copy()
                            improved = True
                        # Revert
                        h[i] = old_val
                        h[j] = h[j] + delta
        if not improved:
            break
    
    # Final verification and return
    # Ensure sum is exactly N/2
    current_sum = np.sum(best_h)
    diff = current_sum - N / 2.0
    if abs(diff) > 1e-10:
        # Distribute correction evenly
        correction = diff / N
        best_h = best_h - correction
        best_h = np.clip(best_h, 0.0, 1.0)
        # Renormalize if clipping changed sum
        new_sum = np.sum(best_h)
        if abs(new_sum - N / 2.0) > 1e-10:
            best_h = best_h * (N / 2.0) / new_sum
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return list(best_h)
