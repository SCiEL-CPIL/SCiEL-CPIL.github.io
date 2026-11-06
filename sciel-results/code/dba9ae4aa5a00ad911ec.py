"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _build_ap_indices():
    """
    Build a list of index arrays (0-based) for all arithmetic progressions
    contained in [0, N-1] with positive step and length >= 1.
    To keep the number manageable, we include:
    - All singletons (trivial, sum is +/-1)
    - All APs of length >= 2
    But the number of APs can be large. Let's estimate:
    For start s in [0, N-1], step d in [1, N-1], length L such that s + (L-1)*d < N.
    Total is roughly sum_{d=1}^{N} sum_{s=0}^{N-1} (N - s)/d ~ N^2 log N / 2 ~ 256^2 * 5.5 / 2 ~ 180k.
    That's manageable for vectorized numpy operations.
    """
    aps = []
    for start in range(N):
        for step in range(1, N):
            # Generate the AP: start, start+step, start+2*step, ... < N
            seq = list(range(start, N, step))
            if len(seq) >= 2:
                aps.append(np.array(seq, dtype=np.int32))
            # Singletons are trivially handled; we don't need to store them
            # as their absolute sum is always 1.
    return aps

def _compute_max_ap_sum(signs, aps):
    """
    Compute the maximum absolute signed sum over all APs in aps.
    signs: array of shape (N,) with values -1 or 1.
    aps: list of index arrays.
    """
    max_sum = 0
    for ap in aps:
        s = np.sum(signs[ap])
        abs_s = abs(s)
        if abs_s > max_sum:
            max_sum = abs_s
    return max_sum

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute APs
    aps = _build_ap_indices()
    # Sort by length descending? Or just keep as is.
    # Longer APs have higher potential sums, so they are more likely to be the bottleneck.
    # Let's sort by length descending to check the most constrained ones first during evaluation.
    aps.sort(key=lambda x: -len(x))
    
    # Convert to a numpy array of arrays? No, list of arrays is fine.
    
    # Initial solution: random or all +1?
    # All +1 gives max sum = max length = N (for step 1, start 0).
    # Random gives ~sqrt(N) * sqrt(log N) roughly.
    
    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    
    # Greedy improvement:
    # Iterate through positions. For each position, try flipping the sign.
    # If it reduces the max AP sum, keep it.
    
    best_signs = signs.copy()
    best_max_sum = _compute_max_ap_sum(best_signs, aps)
    
    # We can also try a few random restarts if time permits.
    # But first, let's do a thorough local search on the initial random solution.
    
    # To speed up evaluation, we can cache the sums of APs?
    # When we flip sign at index i, only APs containing i change their sum.
    # Let's build a map: index -> list of AP indices containing that index.
    
    index_to_aps = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for idx in ap:
            index_to_aps[idx].append(ap_idx)
    
    # Precompute initial sums for all APs
    ap_sums = np.zeros(len(aps), dtype=np.int32)
    for ap_idx, ap in enumerate(aps):
        ap_sums[ap_idx] = np.sum(best_signs[ap])
    
    current_max_sum = np.max(np.abs(ap_sums))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try flipping each position
        for i in range(N):
            # Flip sign at i
            old_sign = best_signs[i]
            best_signs[i] = -old_sign
            
            # Update ap_sums for APs containing i
            for ap_idx in index_to_aps[i]:
                ap_sums[ap_idx] += 2 * (-old_sign)  # change is -2*old_sign -> +2*new_sign? 
                # Wait: sum_new = sum_old - old_sign + new_sign = sum_old - old_sign - old_sign = sum_old - 2*old_sign.
                # Since new_sign = -old_sign.
                # So change is -2 * old_sign.
            
            # Compute new max
            new_max_sum = np.max(np.abs(ap_sums))
            
            if new_max_sum < current_max_sum:
                current_max_sum = new_max_sum
                improved = True
            else:
                # Revert
                best_signs[i] = old_sign
                for ap_idx in index_to_aps[i]:
                    ap_sums[ap_idx] -= 2 * (-old_sign)  # revert the change: add 2*old_sign
            
            if time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break
    
    # If we have time, try a few random restarts
    while time.time() < deadline - 5:
        signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        ap_sums = np.zeros(len(aps), dtype=np.int32)
        for ap_idx, ap in enumerate(aps):
            ap_sums[ap_idx] = np.sum(signs[ap])
        
        current_max_sum = np.max(np.abs(ap_sums))
        
        if current_max_sum < best_max_sum:
            best_signs = signs.copy()
            best_max_sum = current_max_sum
        
        # Quick local search on this new start
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(N):
                old_sign = best_signs[i]
                best_signs[i] = -old_sign
                
                for ap_idx in index_to_aps[i]:
                    ap_sums[ap_idx] -= 2 * old_sign
                
                new_max_sum = np.max(np.abs(ap_sums))
                
                if new_max_sum < current_max_sum:
                    current_max_sum = new_max_sum
                    improved = True
                else:
                    best_signs[i] = old_sign
                    for ap_idx in index_to_aps[i]:
                        ap_sums[ap_idx] += 2 * old_sign
                
                if time.time() > deadline - 2:
                    break
            
            if time.time() > deadline - 2:
                break
        
        if current_max_sum < best_max_sum:
            best_max_sum = current_max_sum
    
    # Convert back to list of ints
    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
