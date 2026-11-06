import math
import bisect
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)
        
    # Calculate weights
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    weights1 = weights[:mid]
    weights2 = weights[mid:]
    
    n1 = len(weights1)
    n2 = len(weights2)
    
    # Generate all subset sums for first half
    # Pack sum and mask into one integer: (sum << n1) | mask
    # n1 is at most 24. Sum is up to L (~10^14).
    # (10^14 << 24) is approx 10^22, fits in Python int easily.
    
    sums1 = [0] # (sum << n1) | mask, initially (0 << n1) | 0
    for i in range(n1):
        w = weights1[i]
        # For each existing sum, create a new one with bit i set
        # We iterate over the current size of sums1 to avoid modifying while iterating
        current_len = len(sums1)
        for j in range(current_len):
            val = sums1[j]
            # Extract sum and mask? No, we can just add w << n1 to the sum part.
            # The mask part is in the lower n1 bits.
            # val = (sum << n1) | mask
            # new_val = ((sum + w) << n1) | (mask | (1 << i))
            # new_val = val + (w << n1) + (1 << i)
            # Wait, is this correct?
            # val + (w << n1) adds w to the sum part.
            # (1 << i) sets the i-th bit in the mask part.
            # Since the sum part is shifted by n1, and mask is < 2^n1, there is no carry-over interference.
            sums1.append(val + (w << n1) + (1 << i))
            
    # Sort sums1
    sums1.sort()
    
    # Pre-extract sums for binary search?
    # We can just use the packed values.
    # We want to find max s1 such that s1 + s2 < L.
    # s1 < L - s2.
    # Let limit = L - s2.
    # We want the largest packed value P in sums1 such that (P >> n1) < limit.
    # This is equivalent to finding the insertion point of (limit << n1) in sums1.
    # Let pos = bisect_left(sums1, limit << n1).
    # The candidate is at pos - 1.
    
    best_sum = 0
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate through all subset sums of second half
    # We can generate them on the fly or pre-generate.
    # Pre-generating is symmetric.
    sums2 = [0]
    for i in range(n2):
        w = weights2[i]
        current_len = len(sums2)
        for j in range(current_len):
            val = sums2[j]
            sums2.append(val + (w << n2) + (1 << i))
            
    # Now iterate through sums2
    # To speed up, we can iterate through sums2.
    # 2^23 is ~8 million.
    # Binary search in 2^24 is ~24 steps.
    # Total ~200 million operations. Might be too slow in pure Python?
    # Let's optimize the inner loop.
    
    # Actually, we can iterate through sums2 and for each, do bisect.
    # 8 million * 24 comparisons = 200 million comparisons.
    # In Python, this might take 10-20 seconds. It's risky.
    
    # Alternative: Sort sums2 as well?
    # If we sort both, we can use a two-pointer technique?
    # We want max s1 + s2 < L.
    # Sort sums1 (ascending). Sort sums2 (ascending).
    # Let i = len(sums1) - 1 (largest s1)
    # Let j = 0 (smallest s2)
    # This doesn't work directly because we want to maximize the sum.
    # Standard approach for "max sum < K" with two sorted lists:
    # For each s2 in sums2:
    #   Find max s1 such that s1 < K - s2.
    #   This is still O(N log M).
    
    # Can we do better?
    # If we sort sums2, we can iterate s2 in descending order?
    # No, the constraint is s1 + s2 < L.
    
    # Let's stick to O(N log M) but optimize constants.
    # We can skip sums2 that are already too large?
    # If s2 >= L, skip. (Sum of weights is > L, so many subsets will exceed L).
    # Actually, the total sum of all weights is L * sum(1/d) > L.
    # So many subsets will be > L.
    
    # Let's try the direct approach first.
    
    mask1_full = (1 << n1) - 1
    mask2_full = (1 << n2) - 1
    
    for val2 in sums2:
        s2 = val2 >> n2
        if s2 >= L:
            continue
        m2 = val2 & mask2_full
        
        limit = L - s2
        # We want max s1 < limit
        # Search key: limit << n1
        key = limit << n1
        pos = bisect.bisect_left(sums1, key)
        if pos > 0:
            val1 = sums1[pos - 1]
            s1 = val1 >> n1
            m1 = val1 & mask1_full
            
            total_sum = s1 + s2
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask1 = m1
                best_mask2 = m2
                
    # Reconstruct indices
    indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            indices.append(i)
            
    for i in range(n2):
        if best_mask2 & (1 << i):
            indices.append(mid + i)
            
    # Ensure non-empty
    if not indices:
        # Fallback to single item with largest value (smallest denominator)
        # which is index 0 (denominator 2)
        return [0]
        
    return sorted(indices)
