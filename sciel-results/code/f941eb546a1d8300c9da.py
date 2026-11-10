import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to evaluate n in [1000, 1000000]
    # For each candidate n, look at n+1 .. n+24
    # Extract the factor composed of primes strictly below 24 from each
    # Count distinct extracted factors
    
    # Precompute small-prime factor for numbers up to 1000024
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # small_factor[x] = product of prime factors < 24 (with multiplicity)
    # Use a sieve-like approach for speed: for each small prime p,
    # multiply p into small_factor[x] for all x divisible by p^k
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # For each power p^k, multiply p into all multiples of p^k
        power = p
        while power <= max_val:
            # Indices: power, 2*power, 3*power, ...
            indices = np.arange(power, max_val + 1, power)
            small_factor[indices] *= p
            power *= p

    # Convert to list for faster access in the sliding window
    sf_list = small_factor.tolist()
    
    best_score = 25  # worst case: all distinct
    best_n = 1000
    
    # Initialize window for n=1000: indices 1001 to 1024
    # Note: n=1000 -> numbers 1001, ..., 1024
    # sf_list indices are 1-based in the problem description? 
    # No, sf_list is 0-indexed. 
    # sf_list[i] corresponds to the factor of integer i.
    # So for n=1000, we look at integers 1001 to 1024.
    # These correspond to sf_list[1001] to sf_list[1024].
    
    window_counts = defaultdict(int)
    for i in range(1001, 1025):
        window_counts[sf_list[i]] += 1
        
    current_distinct = len(window_counts)
    if current_distinct < best_score:
        best_score = current_distinct
        best_n = 1000
        if best_score == 1:
            return best_n

    # Slide window for n from 1001 to 1000000
    # For n, window is sf_list[n+1] to sf_list[n+24]
    # Previous n-1 window was sf_list[n] to sf_list[n+23]
    # So we remove sf_list[n] and add sf_list[n+24]
    
    for n in range(1001, max_n + 1):
        if n % 10000 == 0:
            if time.time() > deadline:
                break
        
        # Remove outgoing element: sf_list[n]
        val_out = sf_list[n]
        window_counts[val_out] -= 1
        if window_counts[val_out] == 0:
            del window_counts[val_out]
            
        # Add incoming element: sf_list[n+24]
        val_in = sf_list[n+24]
        window_counts[val_in] += 1
        
        current_distinct = len(window_counts)
        
        if current_distinct < best_score:
            best_score = current_distinct
            best_n = n
            if best_score == 1:
                break

    return best_n
