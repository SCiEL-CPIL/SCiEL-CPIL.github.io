"""
Improved program: Fast local search with max tracking for minimizing max |sum| over APs.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256

def _generate_aps(n):
    """
    Generate all arithmetic progressions in [0, n-1] with length > 1.
    Returns list of lists of indices.
    """
    ap_indices = []
    for d in range(1, n):
        for r in range(d):
            seq = list(range(r, n, d))
            if len(seq) > 1:
                ap_indices.append(seq)
    return ap_indices

def _precompute_ap_containing(ap_indices, n):
    """
    For each index k, find which APs contain it.
    Returns list of lists where ap_containing[k] is list of AP indices j containing k.
    """
    ap_containing = [[] for _ in range(n)]
    for j, indices in enumerate(ap_indices):
        for k in indices:
            ap_containing[k].append(j)
    return ap_containing

def _construct_signs_fast():
    """
    Construct signs using greedy init and fast local search with max tracking.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = _N
    ap_indices = _generate_aps(n)
    num_aps = len(ap_indices)
    
    if num_aps == 0:
        return [1] * n

    ap_containing = _precompute_ap_containing(ap_indices, n)
    
    # Precompute affected APs as numpy arrays for faster indexing
    affected_aps_list = [np.array(aps, dtype=np.int32) if len(aps) > 0 else np.array([], dtype=np.int32) 
                         for aps in ap_containing]
    
    # Initialize sums to 0
    current_sums = np.zeros(num_aps, dtype=np.int32)
    signs = np.zeros(n, dtype=np.int8)
    
    # Greedy Initialization
    current_max = 1
    
    for i in range(n):
        best_sign = 1
        best_max_val = float('inf')
        
        affected = affected_aps_list[i]
        
        for s in [1, -1]:
            if len(affected) > 0:
                old_sums = current_sums[affected].copy()
                current_sums[affected] += s
                new_max = max(current_max, int(np.max(np.abs(current_sums[affected]))))
                if new_max < best_max_val:
                    best_max_val = new_max
                    best_sign = s
                current_sums[affected] = old_sums
            else:
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_sign = s
        
        signs[i] = best_sign
        if len(affected) > 0:
            current_sums[affected] += best_sign
            current_max = int(np.max(np.abs(current_sums)))
    
    # Local Search with proper max tracking
    best_signs = signs.copy()
    best_max = current_max
    rng = random.Random(42)
    
    no_improve_count = 0
    max_no_improve = 200
    
    # Track which APs have the current max to speed up evaluation
    # We'll maintain a count of how many APs have each absolute sum value
    
    while time.time() < deadline:
        pos = rng.randint(0, n - 1)
        
        affected = affected_aps_list[pos]
        if len(affected) == 0:
            continue
            
        old_sign = signs[pos]
        
        # Apply flip
        old_sums = current_sums[affected].copy()
        current_sums[affected] -= 2 * old_sign
        
        # Compute new max efficiently
        # Only affected APs changed, so new global max = max(max of unchanged, max of changed)
        # We need to know the max of unchanged APs. 
        # Simple approach: just compute np.max on full array - O(num_aps) but fast in numpy
        new_max = int(np.max(np.abs(current_sums)))
        
        if new_max < best_max:
            best_max = new_max
            best_signs = signs.copy()
            signs[pos] = -old_sign
            no_improve_count = 0
        else:
            # Revert
            current_sums[affected] = old_sums
            signs[pos] = old_sign
            no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                # Perturb: flip 2-4 random positions blindly
                num_flips = rng.randint(2, 4)
                for _ in range(num_flips):
                    p = rng.randint(0, n - 1)
                    aff = affected_aps_list[p]
                    if len(aff) == 0:
                        continue
                    old_s = signs[p]
                    signs[p] = -old_s
                    current_sums[aff] -= 2 * old_s
                
                # Recompute max after perturbation
                current_max = int(np.max(np.abs(current_sums)))
                no_improve_count = 0
                
    return [int(x) for x in best_signs]

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    signs = _construct_signs_fast()
    return signs

# EVOLVE-BLOCK-END
