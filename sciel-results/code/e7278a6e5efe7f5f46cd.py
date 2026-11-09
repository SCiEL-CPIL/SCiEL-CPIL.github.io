import time
import bisect
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
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
        sums = [0] * (1 << size)
        # Standard DP to generate subset sums
        # sums[mask] = sums[mask without lowest set bit] + weight[lowest set bit]
        for i in range(size):
            bit = 1 << i
            w = weights_list[i]
            for mask in range(bit):
                sums[bit | mask] = sums[mask] + w
        
        # Filter out sums >= L and keep (sum, mask) pairs
        valid_pairs = []
        for mask in range(1 << size):
            s = sums[mask]
            if s < L:
                valid_pairs.append((s, mask))
        
        return valid_pairs

    left_pairs = generate_sums(left_weights)
    right_pairs = generate_sums(right_weights)
    
    # Sort by sum
    left_pairs.sort(key=lambda x: x[0])
    right_pairs.sort(key=lambda x: x[0])
    
    # Extract sorted right sums for bisect
    right_sums = [p[0] for p in right_pairs]
    
    # For each left sum, find the largest right sum such that total < L
    # Use bisect to find the right boundary
    best_deficit = L
    best_left_mask = -1
    best_right_mask = -1
    
    for ls, lm in left_pairs:
        # Find largest rs such that ls + rs < L, i.e., rs < L - ls
        max_rs = L - ls - 1  # We need rs <= L - ls - 1 (integer arithmetic)
        
        # bisect_right gives insertion point; we want the last index where right_sums[idx] <= max_rs
        # bisect_right(right_sums, max_rs) gives index after last element <= max_rs
        idx = bisect.bisect_right(right_sums, max_rs)
        if idx > 0:
            # The best right pair is at index idx - 1
            rs, rm = right_pairs[idx - 1]
            deficit = L - (ls + rs)
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = lm
                best_right_mask = rm
        
        # Deadline check
        if time.time() > deadline:
            break
    
    # Fallback if no valid pair found
    if best_left_mask == -1:
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
