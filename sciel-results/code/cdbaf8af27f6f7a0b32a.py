"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import random

def construct():
    # Fallback: Static set of 31 consecutive integers
    n_fallback = 31
    start_val = 226
    fallback_result = [start_val + i for i in range(n_fallback)]
    
    # Try to find a set of size 32 using randomized search
    # with the sufficient condition: disjoint sum ranges for consecutive cardinalities
    target_n = 32
    max_val = 256
    
    for _ in range(1000):
        # Sample 32 distinct integers
        candidate = random.sample(range(1, max_val + 1), target_n)
        candidate.sort()
        
        # Check if max sum of k elements < min sum of k+1 elements
        valid = True
        # Precompute prefix sums for min sums
        # min_sum[k] = sum of first k elements
        # max_sum[k] = sum of last k elements
        prefix = [0] * (target_n + 1)
        for i in range(target_n):
            prefix[i+1] = prefix[i] + candidate[i]
        
        for k in range(1, target_n):
            # Min sum of k+1 elements is sum of first k+1
            min_sum_k_plus_1 = prefix[k+1]
            # Max sum of k elements is sum of last k
            # Sum of last k = total_sum - sum of first target_n - k
            total_sum = prefix[target_n]
            max_sum_k = total_sum - prefix[target_n - k]
            
            if max_sum_k >= min_sum_k_plus_1:
                valid = False
                break
        
        if valid:
            return [int(x) for x in candidate]
            
    return [int(x) for x in fallback_result]

# EVOLVE-BLOCK-END
