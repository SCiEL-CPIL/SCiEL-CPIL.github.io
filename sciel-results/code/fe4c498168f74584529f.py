# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of denominators [2,3,...,48] (zero-based indices)
    whose reciprocal sum is STRICTLY BELOW 1 and as close to 1 as possible.
    """
    import numpy as np
    from math import gcd
    from functools import reduce
    from fractions import Fraction

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # LCM of 2..48
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = reduce(lcm, denominators)
    
    # Values 1/d as integer multiples of 1/L
    vals = [L // d for d in denominators]
    
    # Meet-in-the-middle: split into A (first 23) and B (last 24)
    A_vals = vals[:23]
    B_vals = vals[23:]
    
    # Generate all subset sums for A (2^23 ~ 8.4M)
    # We'll store sums in a list of ints, and corresponding mask (bitmask over A)
    nA = len(A_vals)
    sums_A = []
    masks_A = []
    for mask in range(1 << nA):
        s = 0
        for i in range(nA):
            if mask & (1 << i):
                s += A_vals[i]
        sums_A.append(s)
        masks_A.append(mask)
    
    # Generate all subset sums for B (2^24 ~ 16.7M)
    nB = len(B_vals)
    sums_B = np.zeros(1 << nB, dtype=np.int64)
    masks_B = np.zeros(1 << nB, dtype=np.int64)
    
    # Fill B sums using iterative approach
    sums_B[0] = 0
    masks_B[0] = 0
    for i in range(nB):
        v = B_vals[i]
        bit = 1 << i
        # For all existing entries, add a new entry with bit i set
        current_count = 1 << i
        for j in range(current_count):
            new_idx = j | bit
            sums_B[new_idx] = sums_B[j] + v
            masks_B[new_idx] = masks_B[j] | bit
    
    # Sort B sums and keep corresponding masks
    idx = np.argsort(sums_B)
    sums_B_sorted = sums_B[idx]
    masks_B_sorted = masks_B[idx]
    
    # Find best combination
    best_sum = 0
    best_mask_A = 0
    best_mask_B = 0
    
    # For each sum_A, find the largest sum_B such that sum_A + sum_B < L
    # Also ensure the combined set is nonempty
    for sA, mA in zip(sums_A, masks_A):
        # We need sA + sB < L, so sB < L - sA
        limit = L - sA  # sB must be < limit
        # Find rightmost position where sums_B_sorted < limit
        # searchsorted with side='right' gives first index where value >= limit
        pos = np.searchsorted(sums_B_sorted, limit, side='left') - 1
        if pos >= 0:
            sB = sums_B_sorted[pos]
            total = sA + sB
            if total > best_sum:
                # Check nonempty
                if mA != 0 or masks_B_sorted[pos] != 0:
                    best_sum = total
                    best_mask_A = mA
                    best_mask_B = masks_B_sorted[pos]
    
    # Construct the result indices
    result = []
    for i in range(nA):
        if best_mask_A & (1 << i):
            result.append(i)
    for i in range(nB):
        if best_mask_B & (1 << i):
            result.append(23 + i)
    
    if not result:
        # Fallback: should not happen, but just in case
        return [0]
    
    return result
# EVOLVE-BLOCK-END
