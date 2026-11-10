import time
import math
import bisect

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Values scaled by L
    values = [L // d for d in denominators]
    
    # Split into two sets
    # Set 1: first 24 items (denominators 2..25)
    # Set 2: last 23 items (denominators 26..48)
    split = 24
    vals_1 = values[:split]
    vals_2 = values[split:]
    n1 = len(vals_1)
    n2 = len(vals_2)
    
    # Generate all subset sums for Set 1
    size_1 = 1 << n1
    sums_1 = [0] * size_1
    for mask in range(1, size_1):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        sums_1[mask] = sums_1[prev_mask] + vals_1[idx]
        
    # Filter and sort Set 1 sums
    # Store (sum, mask) pairs where sum < L
    pairs_1 = []
    for mask in range(size_1):
        s = sums_1[mask]
        if s < L:
            pairs_1.append((s, mask))
            
    pairs_1.sort()
    
    # Separate sums and masks for faster binary search
    sums_1_sorted = [p[0] for p in pairs_1]
    masks_1_sorted = [p[1] for p in pairs_1]
    
    best_deficit = L
    best_mask_1 = 0
    best_mask_2 = 0
    best_sum_val = 0
    
    # Generate subset sums for Set 2 and search
    size_2 = 1 << n2
    sums_2 = [0] * size_2
    for mask in range(1, size_2):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        sums_2[mask] = sums_2[prev_mask] + vals_2[idx]
        
    # Iterate through Set 2 subsets
    # We check time periodically
    check_interval = 10000
    last_check = time.time()
    
    for i, mask_2 in enumerate(range(size_2)):
        # Periodic time check
        if i % check_interval == 0:
            if time.time() > deadline - 5:
                break
                
        s2 = sums_2[mask_2]
        if s2 >= L:
            continue
            
        # We want s1 such that s1 + s2 < L  => s1 < L - s2
        target = L - s2
        
        # Find largest s1 in sums_1_sorted such that s1 < target
        # bisect_left finds first index where element >= target
        pos = bisect.bisect_left(sums_1_sorted, target)
        
        # Candidate is at pos - 1
        if pos > 0:
            s1 = sums_1_sorted[pos - 1]
            mask_1 = masks_1_sorted[pos - 1]
            total_sum = s1 + s2
            if total_sum < L:
                deficit = L - total_sum
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_mask_1 = mask_1
                    best_mask_2 = mask_2
                    best_sum_val = total_sum
                    
    # Reconstruct indices
    indices = []
    
    # Indices from Set 1
    for i in range(n1):
        if best_mask_1 & (1 << i):
            indices.append(i)
            
    # Indices from Set 2
    for i in range(n2):
        if best_mask_2 & (1 << i):
            indices.append(split + i)
            
    # Ensure non-empty
    if not indices:
        # Fallback
        indices = [0, 1, 5, 41]
        
    return indices
