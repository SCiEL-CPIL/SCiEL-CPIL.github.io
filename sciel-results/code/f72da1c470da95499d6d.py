import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    K = [L // d for d in denominators]
    
    # Split indices
    n1 = 23
    n2 = n - n1
    
    K1 = K[:n1]
    K2 = K[n1:]
    
    # Generate subset sums for group 1
    # Using numpy for speed
    def gen_sums_float(K_group, n_bits):
        # Start with sum=0, mask=0
        sums = np.array([0.0], dtype=np.float64)
        masks = np.array([0], dtype=np.uint32)
        
        for i, k in enumerate(K_group):
            # Create new sums
            new_sums = sums + k
            new_masks = masks | (1 << i)
            
            sums = np.concatenate([sums, new_sums])
            masks = np.concatenate([masks, new_masks])
            
            if time.time() > deadline:
                break
        
        # Sort by sum
        indices = np.argsort(sums)
        sums_sorted = sums[indices]
        masks_sorted = masks[indices]
        
        return sums_sorted, masks_sorted

    # Generate for group 1
    sums1, masks1 = gen_sums_float(K1, n1)
    
    if time.time() > deadline:
        return [0]

    # Generate for group 2
    sums2, masks2 = gen_sums_float(K2, n2)
    
    if time.time() > deadline:
        return [0]

    L_f = float(L)
    
    # We want to find pairs (s1, s2) such that s1 + s2 < L_f
    # and maximize s1 + s2.
    # Iterate over sums1
    best_total_f = -1.0
    best_mask1 = 0
    best_mask2 = 0
    best_candidates = [] # Store top candidates for exact verification
    
    # For efficiency, we can iterate sums1 and for each, find the best s2
    # Since sums2 is sorted, we can use searchsorted
    # For a given s1, we want s2 < L_f - s1
    # searchsorted gives the insertion point. We want the element before it.
    
    # We will collect a list of (total_float, mask1, mask2)
    # To avoid memory blowup, we only keep the top few.
    
    # Actually, we can just find the single best float candidate first,
    # but due to precision, we should check neighbors.
    # Let's find the top 10 float candidates.
    
    # Efficient way:
    # For each s1, find idx = searchsorted(sums2, L_f - s1, side='right') - 1
    # If idx >= 0, candidate = s1 + sums2[idx]
    # We can vectorize this?
    # targets = L_f - sums1
    # idxs = np.searchsorted(sums2, targets, side='right') - 1
    # valid = idxs >= 0
    # candidate_sums = sums1[valid] + sums2[idxs[valid]]
    # This gives an array of candidate sums.
    # We can then find the max, and also get the top K.
    
    targets = L_f - sums1
    idxs = np.searchsorted(sums2, targets, side='right') - 1
    valid = idxs >= 0
    
    if not np.any(valid):
        return [0]
    
    valid_idxs1 = np.where(valid)[0]
    valid_idxs2 = idxs[valid]
    
    candidate_sums = sums1[valid_idxs1] + sums2[valid_idxs2]
    
    # Get top 20 candidates
    if len(candidate_sums) > 20:
        top_k_indices = np.argpartition(candidate_sums, -20)[-20:]
    else:
        top_k_indices = np.arange(len(candidate_sums))
        
    # Get the corresponding masks
    top_sums = candidate_sums[top_k_indices]
    top_m1 = masks1[valid_idxs1[top_k_indices]]
    top_m2 = masks2[valid_idxs2[top_k_indices]]
    
    # Now verify with exact integers
    best_exact_sum = -1
    best_m1_exact = 0
    best_m2_exact = 0
    
    for m1, m2 in zip(top_m1, top_m2):
        # Calculate exact sum
        s = 0
        # Unpack m1
        mm = m1
        for j in range(n1):
            if mm & (1 << j):
                s += K1[j]
        # Unpack m2
        mm = m2
        for j in range(n2):
            if mm & (1 << j):
                s += K2[j]
                
        if s < L and s > best_exact_sum:
            best_exact_sum = s
            best_m1_exact = m1
            best_m2_exact = m2
            
        if time.time() > deadline:
            break
            
    if best_exact_sum == -1:
        # Fallback: just pick one small element
        return [0]
        
    # Reconstruct indices
    indices = []
    mm = best_m1_exact
    for j in range(n1):
        if mm & (1 << j):
            indices.append(j)
            
    mm = best_m2_exact
    for j in range(n2):
        if mm & (1 << j):
            indices.append(n1 + j)
            
    if not indices:
        indices = [0]
        
    return indices

if __name__ == "__main__":
    res = construct()
