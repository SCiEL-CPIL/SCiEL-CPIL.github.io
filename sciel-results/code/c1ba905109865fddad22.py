import time
from math import lcm

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
    # 23 and 24 is a good split for 47
    half = n // 2
    left_size = half
    right_size = n - half
    
    left_weights = weights[:left_size]
    right_weights = weights[left_size:]
    
    # Generate left sums and masks
    # We need masks to reconstruct the indices later
    # Storing (sum, mask) tuples
    left_count = 1 << left_size
    left_pairs = [(0, 0)] * left_count
    
    # Efficient generation using bit manipulation
    # sums[mask] = sum of weights where bits are set in mask
    # We can generate sums first, then zip with masks?
    # Or generate pairs directly?
    # Generating sums is faster.
    left_sums = [0] * left_count
    for i in range(left_size):
        bit = 1 << i
        w = left_weights[i]
        for mask in range(1 << i):
            left_sums[bit | mask] = left_sums[mask] + w
            
    # Create list of (sum, mask) for left
    # This takes memory but is fast enough
    left_pairs = [(left_sums[m], m) for m in range(left_count)]
    left_pairs.sort(key=lambda x: x[0])
    
    # Generate right sums and masks
    right_count = 1 << right_size
    right_sums = [0] * right_count
    for i in range(right_size):
        bit = 1 << i
        w = right_weights[i]
        for mask in range(1 << i):
            right_sums[bit | mask] = right_sums[mask] + w
            
    right_pairs = [(right_sums[m], m) for m in range(right_count)]
    right_pairs.sort(key=lambda x: x[0])
    
    # Two-pointer search
    # We want max(l_sum + r_sum) < L
    # left_pairs is sorted ascending by sum
    # right_pairs is sorted ascending by sum
    
    best_sum = 0
    best_left_mask = 0
    best_right_mask = 0
    
    # i starts from smallest left sum
    # j starts from largest right sum
    i = 0
    j = right_count - 1
    
    # Optimization: Check time periodically
    # The loop runs at most left_count + right_count times, which is ~25M
    # Python loop of 25M might take 5-10 seconds.
    # We should be safe.
    
    while i < left_count and j >= 0:
        if time.time() > deadline:
            break
            
        l_sum, l_mask = left_pairs[i]
        r_sum, r_mask = right_pairs[j]
        
        total = l_sum + r_sum
        
        if total < L:
            if total > best_sum:
                best_sum = total
                best_left_mask = l_mask
                best_right_mask = r_mask
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
