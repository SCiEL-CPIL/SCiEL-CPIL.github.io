"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _precompute_aps():
    """
    Precompute all arithmetic progressions (start, step, length) with length >= 2.
    Returns a list of tuples (start_idx, step, length) where start_idx is 0-based.
    """
    aps = []
    for step in range(1, N // 2 + 1):
        for start in range(0, N - step):  # 0-based start, must allow at least 2 terms
            # Max length for this start and step
            # Last term index: start + (L-1)*step < N => L <= (N-1-start)//step + 1
            max_len = (N - 1 - start) // step + 1
            if max_len < 2:
                continue
            for length in range(2, max_len + 1):
                aps.append((start, step, length))
    return aps

def _init_prefix_sums(signs):
    """
    Initialize prefix sums for each residue class modulo d for all d.
    prefix_sums[d][r][k] = sum of signs[i] for i in [0, N) such that i % d == r and i//d <= k.
    Actually, simpler: for each d, we have d residue classes.
    For a fixed d and residue r, the elements are at indices r, r+d, r+2d...
    Let's store P[d][r] as an array of cumulative sums for that specific subsequence.
    P[d][r][k] = sum_{j=0}^{k} signs[r + j*d]
    """
    prefix_sums = {}
    for d in range(1, N // 2 + 1):
        prefix_sums[d] = {}
        for r in range(d):
            # Get indices for this residue class
            idxs = np.arange(r, N, d)
            if len(idxs) == 0:
                prefix_sums[d][r] = np.array([0]) # Dummy
            else:
                vals = signs[idxs]
                # Cumulative sum: cumsum[k] = sum(vals[0..k])
                # We want sum(vals[a..b]) = cumsum[b] - (cumsum[a-1] if a>0 else 0)
                prefix_sums[d][r] = np.concatenate(([0], np.cumsum(vals)))
    return prefix_sums

def _update_prefix_sums(prefix_sums, pos, old_sign, new_sign, d_max):
    """
    Update prefix sums for all d where pos is part of the subsequence.
    pos belongs to residue r = pos % d for all d.
    The index in the subsequence is k = pos // d.
    We need to add (new_sign - old_sign) to all P[d][r][j] for j >= k.
    """
    delta = new_sign - old_sign
    if delta == 0:
        return
    for d in range(1, d_max + 1):
        r = pos % d
        k = pos // d
        P = prefix_sums[d][r]
        # P has length len(idxs) + 1. The value at P[k+1] corresponds to sum up to index k.
        # So we add delta to P[k+1 : end]
        P[k+1:] += delta

def _compute_max_abs_sum(signs, aps, prefix_sums):
    """
    Compute max absolute sum using prefix sums.
    """
    max_abs = 0
    # We can vectorize this to some extent, but Python loop over ~10k-20k APs is okay if fast.
    # Total APs count:
    # d=1: ~255 starts, avg len 128 -> ~16k APs?
    # Let's check count roughly.
    # Sum_{d=1}^{127} Sum_{s=0}^{N-d-1} ( (N-1-s)//d + 1 - 1 ) 
    # This is roughly O(N^2 log N). For N=256, it's manageable.
    
    for start, step, length in aps:
        r = start % step
        # Index in subsequence for start: k_start = start // step
        # Index in subsequence for end: k_end = (start + (length-1)*step) // step
        k_start = start // step
        k_end = k_start + length - 1
        
        P = prefix_sums[step][r]
        # Sum is P[k_end + 1] - P[k_start]
        s_val = P[k_end + 1] - P[k_start]
        abs_val = abs(s_val)
        if abs_val > max_abs:
            max_abs = abs_val
            
    return max_abs

def _construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Precompute APs
    aps = _precompute_aps()
    d_max = max(a[1] for a in aps) if aps else 1
    
    # Initial candidate: Random or Thue-Morse
    # Thue-Morse is often good for discrepancy
    signs = np.zeros(N, dtype=np.int8)
    for i in range(N):
        # Count set bits
        c = bin(i).count('1')
        signs[i] = 1 if c % 2 == 0 else -1
        
    # Initialize prefix sums
    prefix_sums = _init_prefix_sums(signs)
    
    current_max = _compute_max_abs_sum(signs, aps, prefix_sums)
    best_signs = signs.copy()
    best_max = current_max
    
    rng = random.Random(42)
    
    # Simulated Annealing
    T = 1.0  # Temperature
    T_min = 0.01
    cooling = 0.999
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2.0:
        if no_improve_count > max_no_improve:
            # Reset or break
            if T < T_min:
                break
            # Restart from best
            signs = best_signs.copy()
            prefix_sums = _init_prefix_sums(signs)
            current_max = best_max
            T = max(T_min, T * 0.5)
            no_improve_count = 0
            max_no_improve = 1000
            
        # Pick random position
        pos = rng.randint(0, N - 1)
        old_sign = signs[pos]
        new_sign = -old_sign
        
        # Update prefix sums
        _update_prefix_sums(prefix_sums, pos, old_sign, new_sign, d_max)
        signs[pos] = new_sign
        
        # Evaluate
        new_max = _compute_max_abs_sum(signs, aps, prefix_sums)
        
        delta = new_max - current_max
        
        if delta < 0:
            # Improvement
            current_max = new_max
            no_improve_count = 0
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Worse or same
            # Accept with probability exp(-delta / T)
            # Since delta is integer >= 0, if delta > 0, prob < 1
            if delta == 0:
                # Accept same to explore?
                accept = True
            else:
                # delta is at least 1
                # If T is small, this will be 0
                prob = np.exp(-delta / T)
                accept = (rng.random() < prob)
            
            if not accept:
                # Revert
                signs[pos] = old_sign
                _update_prefix_sums(prefix_sums, pos, new_sign, old_sign, d_max)
                no_improve_count += 1
            else:
                current_max = new_max
                no_improve_count += 1
                
        # Cool down
        T = max(T_min, T * cooling)

    # Final verification
    final_max = _compute_max_abs_sum(best_signs, aps, _init_prefix_sums(best_signs))
    
    return best_signs.tolist()

# EVOLVE-BLOCK-END
