"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to check n in [1000, 1000000]
    # For each n, we look at n+1 to n+24
    # So we need factors for numbers in [1001, 1000024]
    
    min_n = 1000
    max_n = 1000000
    
    start_num = min_n + 1
    end_num = max_n + 24
    
    # Generate array of numbers
    nums = np.arange(start_num, end_num + 1, dtype=np.int64)
    
    # Initialize factors array
    fac = np.ones(len(nums), dtype=np.int64)
    
    # Compute small-prime factors for all numbers
    for p in primes:
        mask = (nums % p == 0)
        fac[mask] *= p
        
    # Now fac[i] corresponds to the factor of nums[i]
    # nums[0] = 1001, which is n+1 for n=1000
    # So for a candidate n, let k = n - 1000
    # The factors are fac[k : k+24]
    
    best_n = min_n
    best_score = 100  # Initialize with a high value
    
    # k ranges from 0 to (max_n - min_n)
    max_k = max_n - min_n
    
    # To optimize the distinct count in the loop, we can use a set.
    # Since the window size is small (24), this is efficient.
    
    for k in range(0, max_k + 1):
        if time.time() > deadline:
            break
            
        # Extract window
        # Using a set to count distinct values
        # fac[k:k+24] is a numpy array slice, set() converts it to a python set
        # This is fast enough for 1e6 iterations
        window = fac[k:k+24]
        distinct_count = len(set(window))
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = k + min_n
            
            # If we find 1, it's the absolute minimum.
            # If we find 2, it's likely optimal, but we continue to be safe
            # unless we have a proof 1 is impossible.
            if best_score == 1:
                break
                
    return best_n
# EVOLVE-BLOCK-END
