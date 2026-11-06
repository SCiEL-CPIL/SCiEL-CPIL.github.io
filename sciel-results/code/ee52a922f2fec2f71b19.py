import sys
from math import gcd
import numpy as np

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute L = lcm(2, ..., 48)
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    # Compute weights
    weights = np.array([L // d for d in denominators], dtype=np.int64)
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    n1 = len(w1)
    n2 = len(w2)
    
    # Generate subset sums for first half
    size1 = 1 << n1
    sums1 = np.zeros(size1, dtype=np.int64)
    for i in range(n1):
        w = w1[i]
        start = 1 << i
        end = start * 2
        sums1[start:end] = sums1[0:start] + w
        
    # Generate subset sums for second half
    size2 = 1 << n2
    sums2 = np.zeros(size2, dtype=np.int64)
    for i in range(n2):
        w = w2[i]
        start = 1 << i
        end = start * 2
        sums2[start:end] = sums2[0:start] + w
        
    # Filter and sort second half
    masks2 = np.arange(size2, dtype=np.int32)
    
    # Filter sums < L
    valid_mask = sums2 < L
    valid_sums2 = sums2[valid_mask]
    valid_masks2 = masks2[valid_mask]
    
    # Free memory
    del sums2
    del masks2
    
    # Sort by sum
    sorted_indices = np.argsort(valid_sums2)
    sorted_sums2 = valid_sums2[sorted_indices]
    sorted_masks2 = valid_masks2[sorted_indices]
    
    # Free memory
    del valid_sums2
    del valid_masks2
    del sorted_indices
    
    # Vectorized search
    # For each s1 in sums1, find max s2 such that s1 + s2 < L
    # s2 < L - 1 - s1
    targets = L - 1 - sums1
    
    # searchsorted returns index where target would be inserted to maintain order
    # We want the largest element <= target
    # So we look at index - 1
    pos = np.searchsorted(sorted_sums2, targets)
    
    # Valid candidates are where pos > 0
    valid_pos_mask = pos > 0
    
    # If no valid candidates at all, fallback
    if not np.any(valid_pos_mask):
        return [0]
        
    # Get the corresponding s2 and m2
    # pos[valid_pos_mask] - 1 gives the index in sorted arrays
    candidate_indices = pos[valid_pos_mask] - 1
    best_s2_values = sorted_sums2[candidate_indices]
    best_m2_values = sorted_masks2[candidate_indices]
    
    # The corresponding s1 values
    s1_values = sums1[valid_pos_mask]
    
    # Total sums
    total_sums = s1_values + best_s2_values
    
    # Find the maximum total sum
    max_idx = np.argmax(total_sums)
    
    best_total = total_sums[max_idx]
    best_s1 = s1_values[max_idx]
    best_m2 = best_m2_values[max_idx]
    
    # We need the mask for s1
    # sums1 was generated in order, so the index in sums1 is the mask.
    # We need to find the index in the original sums1 array for best_s1?
    # No, we selected s1_values from sums1[valid_pos_mask].
    # We need the index of this element in the original sums1 array to get its mask.
    
    # Let's retrieve the indices of valid_pos_mask
    valid_indices = np.where(valid_pos_mask)[0]
    best_mask1_idx = valid_indices[max_idx]
    best_mask1 = best_mask1_idx # Since sums1 is indexed by mask
    
    # Reconstruct indices
    indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            indices.append(i)
            
    for i in range(n2):
        if best_m2 & (1 << i):
            indices.append(mid + i)
            
    if not indices:
        indices = [0]
        
    return indices
