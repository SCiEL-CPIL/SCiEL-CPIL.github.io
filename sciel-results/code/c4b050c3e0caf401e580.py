import time
from math import lcm
from functools import reduce
from bisect import bisect_right

def construct():
    start_time = time.time()
    deadline = 50.0  # 50 seconds safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    L = reduce(lcm, denominators)
    
    # Convert to integer values
    vals = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2 # 23
    n1 = mid
    n2 = n - mid # 24
    
    v1 = vals[:mid]
    v2 = vals[mid:]
    
    # Generate all subset sums for second half (larger one)
    size2 = 1 << n2
    sums2 = [0] * size2
    # DP generation
    for i in range(1, size2):
        lsb = i & -i
        idx = lsb.bit_length() - 1
        sums2[i] = sums2[i ^ lsb] + v2[idx]
        
    # Sort sums2 to allow binary search
    # We need to keep track of which mask produced which sum?
    # Actually, for finding the best sum, we just need the sum value.
    # But we need to reconstruct the indices.
    # So we must store (sum, mask).
    # Storing 16M tuples is memory heavy (~128MB+).
    # Alternative: Store sums in a list, and after finding the best sum, 
    # we can re-derive the mask? No, multiple masks can have same sum.
    # We need the specific mask that gives the sum closest to target.
    
    # Let's store (sum, mask) as a list of tuples.
    # 16M tuples of 2 ints. Memory: 16M * 16 bytes (roughly) = 256MB.
    # This might be high but usually acceptable.
    
    pairs2 = [(s, i) for i, s in enumerate(sums2)]
    pairs2.sort(key=lambda x: x[0])
    
    # Extract sorted sums for bisect
    sorted_sums2 = [p[0] for p in pairs2]
    
    best_deficit = L
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate over all subset sums for first half
    size1 = 1 << n1
    sums1 = [0] * size1
    for i in range(1, size1):
        lsb = i & -i
        idx = lsb.bit_length() - 1
        sums1[i] = sums1[i ^ lsb] + v1[idx]
        
    for i in range(size1):
        s1 = sums1[i]
        # We need s2 such that s1 + s2 < L
        # s2 < L - s1
        # Max s2 is L - s1 - 1
        target = L - s1 - 1
        if target < 0:
            continue
            
        # Find largest s2 <= target
        # bisect_right returns index to insert to keep sorted
        # So idx - 1 is the largest element <= target
        pos = bisect_right(sorted_sums2, target) - 1
        
        if pos >= 0:
            s2 = sorted_sums2[pos]
            total_sum = s1 + s2
            deficit = L - total_sum
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask1 = i
                best_mask2 = pairs2[pos][1]
                
                # Early termination? If deficit is 1, it's minimal (integers).
                # But deficit is integer. Min positive deficit is 1.
                if deficit == 1:
                    break
                    
        # Time check
        if time.time() > deadline:
            break
            
    if best_deficit == L:
        # Fallback: return empty? No, must be non-empty.
        # This shouldn't happen as we can always pick 1/2.
        return []
        
    # Reconstruct indices
    indices = []
    for j in range(n1):
        if best_mask1 & (1 << j):
            indices.append(j)
            
    for j in range(n2):
        if best_mask2 & (1 << j):
            indices.append(mid + j)
            
    return indices
