import time
from math import lcm
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2 to 48
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are L / d
    weights = [L // d for d in denominators]
    
    # Split into two halves
    half = n // 2
    left_size = half
    right_size = n - half
    
    left_weights = weights[:left_size]
    right_weights = weights[left_size:]
    
    # Generate right sums and masks, sorted
    right_count = 1 << right_size
    right_sums = [0] * right_count
    for i in range(right_size):
        bit = 1 << i
        w = right_weights[i]
        for mask in range(1 << i):
            right_sums[bit | mask] = right_sums[mask] + w
            
    # Create list of (sum, mask) for right and sort by sum
    right_pairs = [(right_sums[m], m) for m in range(right_count)]
    right_pairs.sort(key=lambda x: x[0])
    
    # Extract sorted sums and masks separately for binary search
    right_sorted_sums = [p[0] for p in right_pairs]
    right_sorted_masks = [p[1] for p in right_pairs]
    
    # Generate left sums using dense bitset approach
    # We use a Python int as a bitset where bit s is set if sum s is achievable
    # Start with only sum 0 achievable
    left_bitset = 1  # bit 0 is set
    
    # Track which masks achieve each sum (we need this to reconstruct indices)
    # For each sum, store one mask that achieves it
    left_sum_to_mask = {0: 0}
    
    # Build the bitset by adding weights one by one
    for i in range(left_size):
        w = left_weights[i]
        # New sums are old sums + w
        # We need to shift the bitset by w positions
        new_bits = (left_bitset << w)
        # Combine with existing
        left_bitset |= new_bits
        
        # Update sum-to-mask mapping for new sums
        # We need to find which sums were newly added
        # This is tricky with just a bitset...
        
    # Alternative: generate all left sums and masks directly
    # But use a more efficient method
    left_count = 1 << left_size
    left_sums = [0] * left_count
    for i in range(left_size):
        bit = 1 << i
        w = left_weights[i]
        for mask in range(1 << i):
            left_sums[bit | mask] = left_sums[mask] + w
            
    # Create list of (sum, mask) for left
    left_pairs = [(left_sums[m], m) for m in range(left_count)]
    left_pairs.sort(key=lambda x: x[0])
    
    # For each left sum, binary search in right_sorted_sums for max right sum < L - left_sum
    best_sum = 0
    best_left_mask = 0
    best_right_mask = 0
    
    # Process left sums in order, but we can do this more efficiently
    # For each unique left sum, find the best right sum
    # Use binary search on right_sorted_sums
    
    # To avoid duplicate work, process each unique left sum once
    # But left_pairs may have duplicates
    
    # Two-pointer approach is actually O(n_left + n_right) which is better than O(n_left * log n_right)
    # Let's stick with two-pointer but optimize the inner loop
    
    i = 0
    j = right_count - 1
    
    while i < left_count and j >= 0:
        if time.time() > deadline:
            break
            
        l_sum, l_mask = left_pairs[i]
        r_sum = right_sorted_sums[j]
        
        total = l_sum + r_sum
        
        if total < L:
            if total > best_sum:
                best_sum = total
                best_left_mask = l_mask
                best_right_mask = right_sorted_masks[j]
            i += 1
        else:
            j -= 1
            
    # Reconstruct indices
    result_indices = []
    for i in range(left_size):
        if best_left_mask & (1 << i):
            result_indices.append(i)
    for i in range(right_size):
        if best_right_mask & (1 << i):
            result_indices.append(left_size + i)
            
    if not result_indices:
        result_indices = [0]
        
    return result_indices
