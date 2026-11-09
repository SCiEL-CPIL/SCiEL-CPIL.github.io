"""Optimized solver for closest strict underpacking using Meet-in-the-Middle."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)  # 47 items

    # Compute LCM of all denominators
    lcm_val = reduce(lambda a, b: a * b // gcd(a, b), denominators)
    
    # Weights: LCM // denominator
    weights = [lcm_val // d for d in denominators]
    target = lcm_val  # sum of weights must be strictly less than this

    # Split into two halves
    # 23 items in left, 24 items in right
    split = 23
    left_weights = weights[:split]
    right_weights = weights[split:]
    
    # Generate all subset sums and masks for left half
    # Mask bits 0..22 correspond to left_weights indices 0..22
    left_sums = [0]
    left_masks = [0]
    for i, w in enumerate(left_weights):
        # Add w to all existing sums
        new_sums = [s + w for s in left_sums]
        new_masks = [m | (1 << i) for m in left_masks]
        left_sums.extend(new_sums)
        left_masks.extend(new_masks)
        
    # Create list of (sum, mask) pairs for left half
    left_pairs = list(zip(left_sums, left_masks))
    
    # Sort by sum ascending
    left_pairs.sort(key=lambda x: x[0])
    
    # Extract sums for bisect? No, bisect on tuples works.
    # We need to find largest sum < limit.
    # bisect_left finds first >= limit.
    # So index - 1 is the largest < limit.
    
    # Generate all subset sums and masks for right half
    # Mask bits 23..46 correspond to right_weights indices 0..23
    # We will shift these masks by 23 when combining
    right_sums = [0]
    right_masks = [0]
    for i, w in enumerate(right_weights):
        new_sums = [s + w for s in right_sums]
        new_masks = [m | (1 << i) for m in right_masks]
        right_sums.extend(new_sums)
        right_masks.extend(new_masks)
        
    # Iterate through right half subsets
    best_total = -1
    best_mask = 0
    
    # To speed up, we can pre-compute right_pairs list
    right_pairs = list(zip(right_sums, right_masks))
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Fallback to greedy if MITM setup is too slow (unlikely)
        return [0] 

    for s2, m2 in right_pairs:
        if time.time() - start_time > DEADLINE:
            break
            
        limit = target - s2
        # Find largest sum in left_pairs such that sum < limit
        # bisect_left returns index of first element >= limit
        pos = bisect.bisect_left(left_pairs, (limit, -1)) - 1
        
        if pos >= 0:
            s1, m1 = left_pairs[pos]
            total = s1 + s2
            if total > best_total:
                best_total = total
                best_mask = m1 | (m2 << split)
                
    # Convert mask to indices
    indices = []
    for i in range(n):
        if best_mask & (1 << i):
            indices.append(i)
            
    return indices
# EVOLVE-BLOCK-END
