"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _precompute_aps(n):
    """
    Precompute all arithmetic progressions (APs) in [0, n-1].
    Returns a list of arrays, where each array contains the indices of one AP.
    """
    aps = []
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                break
            ap = np.arange(r, n, d)
            if len(ap) > 1: # Skip singletons, handled separately
                aps.append(ap)
    return aps

def _max_abs_ap_sum_fast(signs, ap_indices, hist):
    """
    Compute max absolute AP sum.
    signs: np.array of -1/1
    ap_indices: list of np.array indices
    hist: dict mapping (abs_sum) -> count? No, we just need the max.
    Actually, for the SA loop, we need a fast way to evaluate the *change* in max.
    But for initialization, we just need the value.
    """
    max_disc = 1
    for ap in ap_indices:
        s = np.sum(signs[ap])
        abs_s = abs(s)
        if abs_s > max_disc:
            max_disc = abs_s
    return max_disc

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s budget, 10s safety margin
    
    n = 256
    rng = random.Random(42)
    
    # Precompute APs
    ap_indices = _precompute_aps(n)
    num_aps = len(ap_indices)
    
    # Initialize with random signs
    signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int64)
    
    # Initial evaluation
    current_max = _max_abs_ap_sum_fast(signs, ap_indices, None)
    best_signs = signs.copy()
    best_max = current_max
    
    # To speed up SA, we can precompute for each index i, which APs it belongs to.
    # idx_to_aps: list of lists of ap_indices
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(ap_indices):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    # Precompute ap_sums array to update incrementally
    # ap_sums[i] = sum of signs in ap_indices[i]
    ap_sums = np.zeros(num_aps, dtype=np.int64)
    for i, ap in enumerate(ap_indices):
        ap_sums[i] = np.sum(signs[ap])
    
    # To quickly find the max abs sum, we can maintain a count of how many APs have a specific abs sum.
    # Max possible abs sum is n=256.
    # counts[s] = number of APs with abs(sum) == s
    max_possible = n
    counts = np.zeros(max_possible + 1, dtype=np.int64)
    
    def update_counts(ap_idx, old_sum, new_sum):
        old_abs = abs(old_sum)
        new_abs = abs(new_sum)
        counts[old_abs] -= 1
        counts[new_abs] += 1
    
    # Initialize counts
    for i in range(num_aps):
        counts[abs(ap_sums[i])] += 1
    
    def get_current_max():
        # Find largest index with count > 0
        # Since max_possible is small (256), linear scan from top is fast
        for s in range(max_possible, 0, -1):
            if counts[s] > 0:
                return s
        return 0
        
    current_max = get_current_max()
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline:
        # Pick random index to flip
        idx = rng.randint(0, n - 1)
        
        # Flip sign
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        # Update ap_sums and counts
        # For each AP containing idx, sum changes by -2 * old_sign
        delta_sum = -2 * old_sign
        
        affected_aps = idx_to_aps[idx]
        for ap_i in affected_aps:
            old_s = ap_sums[ap_i]
            new_s = old_s + delta_sum
            ap_sums[ap_i] = new_s
            update_counts(ap_i, old_s, new_s)
        
        new_max = get_current_max()
        
        # Acceptance
        if new_max < current_max:
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
        else:
            delta = new_max - current_max
            if delta > 0:
                # Probabilistic accept
                # exp(-delta / T)
                # If T is small, this is 0.
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_max = new_max
                else:
                    # Reject: revert
                    signs[idx] = old_sign
                    for ap_i in affected_aps:
                        new_s = ap_sums[ap_i]
                        old_s = new_s - delta_sum
                        ap_sums[ap_i] = old_s
                        update_counts(ap_i, new_s, old_s)
                    current_max = get_current_max() # Should be same as before
            else:
                # Equal or better (handled above), but if equal, we might accept to move
                if delta == 0:
                    current_max = new_max
                else:
                    # This shouldn't happen if new_max < current_max was caught
                    signs[idx] = old_sign
                    for ap_i in affected_aps:
                        new_s = ap_sums[ap_i]
                        old_s = new_s - delta_sum
                        ap_sums[ap_i] = old_s
                        update_counts(ap_i, new_s, old_s)
                    current_max = get_current_max()
        
        # Cool
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_signs.tolist()

# EVOLVE-BLOCK-END
