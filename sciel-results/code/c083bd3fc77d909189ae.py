import numpy as np
from math import gcd

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    # Weights
    weights = np.array([L // d for d in denominators], dtype=np.int64)
    
    # Split
    mid = n // 2
    w_A = weights[:mid]
    w_B = weights[mid:]
    
    n_A = len(w_A)
    n_B = len(w_B)
    
    # --- Part B ---
    # Generate masks
    size_B = 1 << n_B
    masks_B = np.arange(size_B, dtype=np.uint64)
    
    # Generate bits (uint8 to save memory)
    bits_B = np.empty((size_B, n_B), dtype=np.uint8)
    for i in range(n_B):
        bits_B[:, i] = (masks_B >> i) & 1
        
    # Compute sums
    sums_B = bits_B @ w_B
    
    # Sort
    sort_idx_B = np.argsort(sums_B)
    sorted_sums_B = sums_B[sort_idx_B]
    sorted_masks_B = masks_B[sort_idx_B]
    
    # Free memory
    del bits_B, sums_B, masks_B, sort_idx_B
    
    # --- Part A ---
    size_A = 1 << n_A
    masks_A = np.arange(size_A, dtype=np.uint64)
    
    bits_A = np.empty((size_A, n_A), dtype=np.uint8)
    for i in range(n_A):
        bits_A[:, i] = (masks_A >> i) & 1
        
    sums_A = bits_A @ w_A
    
    del bits_A
    
    # --- Find Best ---
    # We want s_A + s_B < L => s_B < L - s_A
    limits = L - sums_A
    
    # Find index of largest s_B < limit
    # searchsorted returns insertion point
    # side='right' gives index of first element >= limit
    # So idx - 1 is index of largest element < limit
    idxs_B = np.searchsorted(sorted_sums_B, limits, side='right') - 1
    
    # Filter valid indices
    valid_mask = idxs_B >= 0
    if not np.any(valid_mask):
        # Fallback: return index for 1/2 (index 0)
        return [0]
        
    valid_idxs_B = idxs_B[valid_mask]
    valid_sums_A = sums_A[valid_mask]
    valid_masks_A = masks_A[valid_mask]
    
    candidate_sums = valid_sums_A + sorted_sums_B[valid_idxs_B]
    
    # Find max
    best_local_idx = np.argmax(candidate_sums)
    
    # Retrieve masks
    mask_A = valid_masks_A[best_local_idx]
    mask_B = sorted_masks_B[valid_idxs_B[best_local_idx]]
    
    # --- Construct Result ---
    indices = []
    for i in range(n_A):
        if (mask_A >> i) & 1:
            indices.append(i)
            
    for i in range(n_B):
        if (mask_B >> i) & 1:
            indices.append(mid + i)
            
    if not indices:
        indices = [0]
        
    return indices
