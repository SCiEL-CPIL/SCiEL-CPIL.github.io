import time
import bisect
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 1..48
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 49))
    
    # Weights
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    # Generate subset sums for w1
    # Returns list of (sum, mask)
    def gen_sums(weights):
        sums = [(0, 0)]
        for i, w in enumerate(weights):
            bit = 1 << i
            # Add new subsets
            # Using list comprehension for speed
            # We iterate over the current size to avoid adding to the list while iterating
            current_len = len(sums)
            for j in range(current_len):
                s, m = sums[j]
                sums.append((s + w, m | bit))
        return sums

    # Generate sums for both halves
    sums1 = gen_sums(w1)
    sums2 = gen_sums(w2)
    
    # Filter out sums >= L
    sums1 = [(s, m) for s, m in sums1 if s < L]
    sums2 = [(s, m) for s, m in sums2 if s < L]
    
    # Sort sums2 by sum value
    sums2.sort(key=lambda x: x[0])
    
    # Extract values for binary search
    sums2_vals = [s for s, m in sums2]
    
    best_deficit = L # Max deficit is L (empty set)
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate through sums1
    for s1, m1 in sums1:
        # We want s1 + s2 < L  =>  s2 < L - s1
        # Strictly below 1, so s1 + s2 <= L - 1
        target = L - 1 - s1
        if target < 0:
            continue
            
        # Find the largest s2 <= target
        # bisect_right returns the insertion point to the right of existing entries
        idx = bisect.bisect_right(sums2_vals, target) - 1
        
        if idx >= 0:
            s2, m2 = sums2[idx]
            current_sum = s1 + s2
            current_deficit = L - current_sum
            
            if current_deficit < best_deficit:
                best_deficit = current_deficit
                best_mask1 = m1
                best_mask2 = m2
                
                # Early exit if deficit is 0? No, deficit must be positive.
                # If deficit is very small, we can stop?
                # Min positive deficit is 1/L? No, sums are integers.
                # Min deficit is 1/L? No, 1 - sum = 1/L * (L - sum).
                # So min deficit is 1/L.
                if best_deficit == 1:
                    break

    # Reconstruct indices
    indices = []
    
    # Mask 1 corresponds to first 'mid' denominators
    for i in range(mid):
        if best_mask1 & (1 << i):
            indices.append(i)
            
    # Mask 2 corresponds to remaining denominators
    for i in range(len(w2)):
        if best_mask2 & (1 << i):
            indices.append(mid + i)
            
    # Ensure non-empty
    if not indices:
        # This shouldn't happen as single elements are valid and better than empty
        indices = [0]
        
    return indices
