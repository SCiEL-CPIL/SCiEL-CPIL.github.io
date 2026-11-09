import time
import bisect
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    weights = [L // d for d in denominators]
    
    # Split into two halves
    half = n // 2
    left_size = half
    right_size = n - half
    
    left_weights = weights[:left_size]
    right_weights = weights[left_size:]
    
    # Function to generate all subset sums and masks
    def generate_sums(weights_list):
        size = len(weights_list)
        total_masks = 1 << size
        sums = [0] * total_masks
        masks = list(range(total_masks)) # Not needed, mask is index
        # We store (sum, mask)
        pairs = []
        for mask in range(total_masks):
            s = 0
            # Efficient sum calculation using DP or bit counting?
            # For 24 bits, simple loop is fine.
            # Or use the DP approach from the seed which was fast.
            pass
        
        # The seed's generate_sums was:
        # sums[bit | mask] = sums[mask] + w
        # This is O(N * 2^N) which is 24 * 16M = 384M ops. Might be slow in Python.
        # Let's use a standard iterative list expansion.
        
        current_sums = [(0, 0)]
        for i, w in enumerate(weights_list):
            # Add w to all existing sums
            # This doubles the list size each step
            # new_sums = [(s + w, m | (1 << i)) for s, m in current_sums]
            # current_sums.extend(new_sums)
            # This creates many intermediate lists.
            
            # Alternative: pre-allocate array of size 2^size
            pass
            
        # Let's stick to the DP array approach but optimized.
        # sums[i] for i in 0..2^size-1
        # sums[0] = 0
        # sums[i] = sums[i without lowest set bit] + weight[lowest set bit index]
        
        size = len(weights_list)
        total = 1 << size
        sums = [0] * total
        for i in range(1, total):
            # find lowest set bit
            # lsb = i & -i
            # idx = lsb.bit_length() - 1
            # sums[i] = sums[i ^ lsb] + weights_list[idx]
            
            # This is O(N * 2^N) with small constant.
            lsb = i & -i
            idx = lsb.bit_length() - 1
            sums[i] = sums[i ^ lsb] + weights_list[idx]
            
        # Filter and return list of (sum, mask) where sum < L
        return [(sums[mask], mask) for mask in range(total) if sums[mask] < L]

    left_pairs = generate_sums(left_weights)
    
    if time.time() > deadline:
        return [0]
    
    right_pairs = generate_sums(right_weights)
    
    if time.time() > deadline:
        return [0]
    
    # Sort right by sum ascending
    right_pairs.sort(key=lambda x: x[0])
    
    # Extract sums for bisect
    right_sums = [p[0] for p in right_pairs]
    
    if time.time() > deadline:
        return [0]
    
    best_deficit = L
    best_left_mask = -1
    best_right_mask = -1
    
    # Iterate through left pairs
    # No need to sort left, just iterate
    for ls, lm in left_pairs:
        if time.time() > deadline:
            break
            
        target = L - ls
        # Find largest right_sum < target
        # bisect_left returns index of first element >= target
        # We want the element before that
        idx = bisect.bisect_left(right_sums, target)
        if idx > 0:
            rs = right_sums[idx - 1]
            rm = right_pairs[idx - 1][1]
            total_sum = ls + rs
            deficit = L - total_sum
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = lm
                best_right_mask = rm
                if deficit == 1:
                    break

    # Fallback
    if best_left_mask == -1:
        # Should not happen as empty set sum 0 < L is valid
        best_left_mask = 0
        best_right_mask = 0
        best_deficit = L

    # Reconstruct indices
    result_indices = []
    for k in range(left_size):
        if best_left_mask & (1 << k):
            result_indices.append(k)
            
    for k in range(right_size):
        if best_right_mask & (1 << k):
            result_indices.append(left_size + k)
            
    # Ensure non-empty
    if not result_indices:
        result_indices = [0]
        
    return result_indices
