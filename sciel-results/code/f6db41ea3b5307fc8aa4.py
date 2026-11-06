"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from fractions import Fraction
    import math
    import bisect

    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # Split indices into two halves
    # A: first 23, B: last 24
    mid = 23
    indices_A = list(range(mid))
    indices_B = list(range(mid, n))
    
    # Precompute values as floats for search, Fractions for verification
    vals_A_float = [1.0 / denominators[i] for i in indices_A]
    vals_B_float = [1.0 / denominators[i] for i in indices_B]
    
    vals_A_frac = [Fraction(1, denominators[i]) for i in indices_A]
    vals_B_frac = [Fraction(1, denominators[i]) for i in indices_B]
    
    nA = len(indices_A)
    nB = len(indices_B)
    
    # Generate all subsets of B
    # Store (float_sum, mask)
    sums_B = []
    # Use iterative generation to avoid recursion overhead
    # mask is an integer from 0 to 2^nB - 1
    for mask in range(1 << nB):
        s = 0.0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += vals_B_float[i]
            m >>= 1
            i += 1
        if s < 1.0:
            sums_B.append((s, mask))
            
    # Sort by float sum descending
    sums_B.sort(key=lambda x: x[0], reverse=True)
    B_sums = [x[0] for x in sums_B]
    B_masks = [x[1] for x in sums_B]
    
    # Generate all subsets of A and find best match
    best_deficit_frac = Fraction(1)
    best_mask_A = 0
    best_mask_B = 0
    
    # We will collect a few top candidates to verify with Fraction
    candidates = []
    
    for mask_A in range(1 << nA):
        sA = 0.0
        m = mask_A
        i = 0
        while m:
            if m & 1:
                sA += vals_A_float[i]
            m >>= 1
            i += 1
            
        if sA >= 1.0:
            continue
            
        target = 1.0 - sA
        
        # Binary search for largest sB < target
        # B_sums is sorted descending.
        # We want the first element in B_sums that is < target?
        # No, B_sums is descending.
        # We want the largest sB such that sB < target.
        # This is equivalent to finding the position where sB < target.
        # Since it's descending, all elements before the "cut" are >= target.
        # We want the first element that is < target? No, that would be the smallest one that is still < target?
        # Wait.
        # Example: B_sums = [0.9, 0.8, 0.5, 0.1]
        # target = 0.6
        # We want 0.5.
        # bisect works on ascending lists.
        # Let's use ascending for binary search.
        
    # Re-sort B_sums ascending for bisect
    sums_B_asc = sorted([(s, m) for s, m in sums_B], key=lambda x: x[0])
    B_sums_asc = [x[0] for x in sums_B_asc]
    B_masks_asc = [x[1] for x in sums_B_asc]
    
    # Now for each A, find largest sB < target in ascending list
    # This is the element just before the insertion point of target.
    
    for mask_A in range(1 << nA):
        if time.time() > deadline:
            break
            
        sA = 0.0
        m = mask_A
        i = 0
        while m:
            if m & 1:
                sA += vals_A_float[i]
            m >>= 1
            i += 1
            
        if sA >= 1.0:
            continue
            
        target = 1.0 - sA
        
        # Find index of first element >= target
        # bisect_left returns the index where target would be inserted to keep order
        # i.e., all elements before idx are < target
        idx = bisect.bisect_left(B_sums_asc, target)
        
        # The largest element < target is at idx - 1
        if idx > 0:
            sB = B_sums_asc[idx - 1]
            mask_B = B_masks_asc[idx - 1]
            total_float = sA + sB
            deficit_float = 1.0 - total_float
            
            # Add to candidates
            candidates.append((deficit_float, mask_A, mask_B, total_float))
            
            # Also check idx (if it exists and is < target due to float errors?)
            # bisect_left ensures B[idx] >= target.
            # But due to float precision, B[idx] might be slightly less than true target?
            # Or B[idx-1] might be slightly more?
            # We will verify top candidates later.
            
    # Sort candidates by deficit (ascending)
    candidates.sort(key=lambda x: x[0])
    
    # Verify top candidates with Fraction
    # We check the top 100 candidates to be safe
    verified_best = None
    verified_best_deficit = Fraction(1)
    
    for deficit_f, mA, mB, total_f in candidates[:100]:
        # Calculate exact sum
        sum_A = Fraction(0)
        m = mA
        i = 0
        while m:
            if m & 1:
                sum_A += vals_A_frac[i]
            m >>= 1
            i += 1
            
        sum_B = Fraction(0)
        m = mB
        i = 0
        while m:
            if m & 1:
                sum_B += vals_B_frac[i]
            m >>= 1
            i += 1
            
        total = sum_A + sum_B
        if total < 1:
            deficit = 1 - total
            if deficit < verified_best_deficit:
                verified_best_deficit = deficit
                verified_best = (mA, mB)
    
    if verified_best is None:
        # Fallback: simple greedy
        # This should not happen if candidates is not empty
        # But if all float sums were >= 1, we need a valid solution
        # 1/2 is always valid.
        return [0]
        
    mA, mB = verified_best
    
    # Reconstruct indices
    result_indices = []
    
    # Map mask A to original indices
    m = mA
    i = 0
    while m:
        if m & 1:
            result_indices.append(indices_A[i])
        m >>= 1
        i += 1
        
    m = mB
    i = 0
    while m:
        if m & 1:
            result_indices.append(indices_B[i])
        m >>= 1
        i += 1
        
    return sorted(result_indices)
# EVOLVE-BLOCK-END
