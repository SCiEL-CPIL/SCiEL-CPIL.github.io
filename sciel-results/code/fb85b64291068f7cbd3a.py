import numpy as np
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    mid = n // 2
    left_denoms = denominators[:mid]
    right_denoms = denominators[mid:]
    
    n_left = len(left_denoms)
    n_right = len(right_denoms)
    
    size_left = 1 << n_left
    size_right = 1 << n_right
    
    # 1. Generate left sums
    sums_left = np.zeros(size_left, dtype=np.float64)
    masks_left = np.arange(size_left, dtype=np.int64)
    
    for i in range(1, size_left):
        lsb = i & -i
        bit_idx = lsb.bit_length() - 1
        prev_mask = i ^ lsb
        sums_left[i] = sums_left[prev_mask] + 1.0 / left_denoms[bit_idx]
        
    if time.time() > deadline:
        return [0]

    sort_idx = np.argsort(sums_left)
    sums_left_sorted = sums_left[sort_idx]
    masks_left_sorted = masks_left[sort_idx]
    
    # 2. Generate right sums
    sums_right = np.zeros(size_right, dtype=np.float64)
    masks_right = np.arange(size_right, dtype=np.int64)
    
    for i in range(1, size_right):
        lsb = i & -i
        bit_idx = lsb.bit_length() - 1
        prev_mask = i ^ lsb
        sums_right[i] = sums_right[prev_mask] + 1.0 / right_denoms[bit_idx]
        
    if time.time() > deadline:
        return [0]

    # 3. Vectorized search
    targets = 1.0 - sums_right
    # We want s_left < target
    # searchsorted with side='right' gives insertion point after existing values
    # So idx - 1 is the largest value < target (or <= target if we used 'left' and adjusted?)
    # If we use side='right', it points to the first element > target.
    # So element at idx-1 is <= target.
    # We need strictly < 1.0 total.
    # s_left + s_right < 1.0
    # s_left < 1.0 - s_right
    # If s_left == 1.0 - s_right, sum is 1.0, which is invalid.
    # So we need the largest s_left strictly less than target.
    # np.searchsorted(a, v, side='right') returns i such that a[i-1] <= v < a[i]
    # So a[i-1] is the largest element <= v.
    # If a[i-1] == v, sum is 1.0. Invalid.
    # We need to handle this.
    
    idxs = np.searchsorted(sums_left_sorted, targets, side='right') - 1
    
    # Check if the found sum is strictly less than target
    # valid_mask = (idxs >= 0) & (sums_left_sorted[idxs] < targets)
    # Note: idxs might be -1.
    
    valid_mask = np.zeros(size_right, dtype=bool)
    valid_indices_mask = idxs >= 0
    if np.any(valid_indices_mask):
        # Only check those with valid index
        s_l_vals = sums_left_sorted[idxs[valid_indices_mask]]
        t_vals = targets[valid_indices_mask]
        # Strictly less
        strict_mask = s_l_vals < t_vals
        # Map back
        valid_mask[valid_indices_mask] = strict_mask
        
    # Calculate candidate sums for valid ones
    # We want the max sum
    # Set invalid ones to -inf
    candidate_sums = np.full(size_right, -np.inf)
    
    if np.any(valid_mask):
        # We need to be careful with indexing
        # idxs[valid_mask] gives the indices into sums_left_sorted
        # sums_right[valid_mask] gives the right sums
        candidate_sums[valid_mask] = sums_left_sorted[idxs[valid_mask]] + sums_right[valid_mask]
        
    if time.time() > deadline:
        return [0]

    # Find top 100 candidates
    # Get indices of the largest 100 values
    if np.all(candidate_sums == -np.inf):
        return [0]
        
    # Use argpartition for top K
    k = 100
    if size_right < k:
        k = size_right
        
    top_k_indices = np.argpartition(candidate_sums, -k)[-k:]
    # Sort these top k by value descending
    top_k_indices = top_k_indices[np.argsort(candidate_sums[top_k_indices])[::-1]]
    
    best_indices = None
    best_deficit = None
    
    for idx_r in top_k_indices:
        if time.time() > deadline:
            break
        if candidate_sums[idx_r] == -np.inf:
            continue
            
        idx_l = idxs[idx_r]
        m_l = masks_left_sorted[idx_l]
        m_r = masks_right[idx_r]
        
        # Reconstruct indices
        indices = []
        for j in range(n_left):
            if (m_l >> j) & 1:
                indices.append(j)
        for j in range(n_right):
            if (m_r >> j) & 1:
                indices.append(mid + j)
                
        total_frac = Fraction(0)
        for idx in indices:
            total_frac += Fraction(1, denominators[idx])
            
        if total_frac < 1:
            deficit = 1 - total_frac
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_indices = indices
                
    if best_indices is None:
        # Fallback
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
        
    return best_indices
