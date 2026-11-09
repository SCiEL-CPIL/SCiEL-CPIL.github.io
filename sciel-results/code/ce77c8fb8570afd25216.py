"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    denominators = list(range(2, 49))  # [2, 3, ..., 48], 47 items
    n = len(denominators)  # 47
    
    # Split into two halves
    n1 = n // 2  # 23
    n2 = n - n1  # 24
    
    half1 = denominators[:n1]
    half2 = denominators[n1:]
    
    # Compute all subset sums for each half
    def compute_subset_sums(values):
        m = len(values)
        size = 1 << m
        sums = np.zeros(size, dtype=np.float64)
        for i, v in enumerate(values):
            bit = 1 << i
            step = bit << 1
            for start in range(0, size, step):
                sums[start + bit:start + step] += v
        return sums
    
    sums1 = compute_subset_sums(half1)
    if time.time() > deadline:
        # Fallback: greedy
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    sums2 = compute_subset_sums(half2)
    if time.time() > deadline:
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    # Sort half2 sums with their indices
    order2 = np.argsort(sums2)
    sorted_sums2 = sums2[order2]
    sorted_indices2 = order2
    
    # For each subset in half1, find best matching subset in half2
    # We need s1 + s2 < 1, maximize s1 + s2
    best_total = -1.0
    best_mask1 = 0
    best_mask2 = 0
    
    # Filter half1 sums that are < 1
    valid1 = sums1 < 1.0
    s1_valid = sums1[valid1]
    idx1_valid = np.where(valid1)[0]
    
    # For each valid s1, find largest s2 < 1 - s1
    # Use binary search
    targets = 1.0 - s1_valid
    # bisect_left gives first position >= target, so we want position - 1
    pos = np.searchsorted(sorted_sums2, targets, side='left')
    pos = pos - 1  # largest index with sorted_sums2[idx] < target
    
    # Only consider valid positions
    valid_pos = pos >= 0
    if not np.any(valid_pos):
        # No valid combination found, fallback
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    pos_valid = pos[valid_pos]
    s1_valid_pos = s1_valid[valid_pos]
    idx1_valid_pos = idx1_valid[valid_pos]
    
    s2_matched = sorted_sums2[pos_valid]
    total_matched = s1_valid_pos + s2_matched
    
    # Find the maximum total that is < 1 (should be by construction, but verify)
    # Due to floating point, some might be >= 1, filter them
    final_valid = total_matched < 1.0
    if not np.any(final_valid):
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    total_final = total_matched[final_valid]
    pos_final = pos_valid[final_valid]
    idx1_final = idx1_valid_pos[final_valid]
    
    best_idx = np.argmax(total_final)
    best_total = total_final[best_idx]
    best_mask1 = idx1_final[best_idx]
    best_mask2 = sorted_indices2[pos_final[best_idx]]
    
    # Verify with exact Fraction arithmetic
    selected = []
    total = Fraction(0)
    for i in range(n1):
        if (best_mask1 >> i) & 1:
            d = half1[i]
            selected.append(i)
            total += Fraction(1, d)
    for i in range(n2):
        if (best_mask2 >> i) & 1:
            d = half2[i]
            selected.append(n1 + i)
            total += Fraction(1, d)
    
    if total >= 1 or not selected:
        # Fallback: greedy
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
    
    return selected
# EVOLVE-BLOCK-END
