import time
import math
import bisect

def construct():
    deadline = time.time() + 55  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    # Python 3.9+ has math.lcm, but to be safe with older versions or if not imported:
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
    # sums_1[mask] = sum of vals_1 for bits set in mask
    size_1 = 1 << n1
    sums_1 = [0] * size_1
    for mask in range(1, size_1):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        sums_1[mask] = sums_1[prev_mask] + vals_1[idx]
        
    # Filter and sort Set 1 sums
    # We need (sum, mask) pairs where sum < L
    pairs_1 = []
    for mask in range(size_1):
        s = sums_1[mask]
        if s < L:
            pairs_1.append((s, mask))
            
    pairs_1.sort()
    
    # Extract sums for binary search
    # bisect works on list of tuples, comparing first element primarily
    # We will use bisect_left on pairs_1 directly
    
    best_deficit = L # Max possible deficit is L (sum=0)
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
        
    for mask_2 in range(size_2):
        if time.time() > deadline:
            break
            
        s2 = sums_2[mask_2]
        if s2 >= L:
            continue
            
        # We want s1 such that s1 + s2 < L  => s1 < L - s2
        target = L - s2
        
        # Find largest s1 in pairs_1 such that s1 < target
        # bisect_left finds first index where element >= target
        pos = bisect.bisect_left(pairs_1, (target, -1))
        
        # Candidate is at pos - 1
        if pos > 0:
            s1, mask_1 = pairs_1[pos - 1]
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
        # Fallback to a known valid small set, e.g., just [1/2, 1/3, 1/7, 1/43, 1/1806] is 1.
        # We need strictly below 1.
        # 1/2 + 1/3 + 1/7 + 1/43 + 1/1806 = 1.
        # Remove one term? 1 - 1/1806 = 1805/1806.
        # Indices for 2,3,7,43.
        # 2 is idx 0, 3 is idx 1, 7 is idx 5, 43 is idx 41.
        indices = [0, 1, 5, 41]
        
    return indices
