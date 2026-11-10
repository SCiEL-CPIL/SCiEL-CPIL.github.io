"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

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
    
    # Now slide window of 24 over n in [1000, 1000000]
    # For candidate n, the numbers are n+1, n+2, ..., n+24
    
    best_score = 25  # worst case: all distinct
    best_n = 1000
    
    # Use numpy for faster window processing
    # We can compute distinct counts for multiple windows at once
    # But let's first try a vectorized approach for the sliding window
    
    # For each n, we need the set of small_factor[n+1:n+25]
    # We can use a sliding window with a frequency array
    
    # Convert to list for faster set operations if needed
    sf_list = small_factor.tolist()
    
    # Iterate over valid n with periodic time checks
    for n in range(1000, max_n + 1):
        # Check time periodically to avoid timeout
        if n % 5000 == 0:
            if time.time() > deadline:
                break
        
        # Extract factors for n+1 .. n+24
        window = sf_list[n + 1 : n + 25]
        
        # Count distinct values
        count = len(set(window))
        
        if count < best_score:
            best_score = count
            best_n = n
            if best_score == 1:
                break  # Can't do better than 1
    
    return best_n
# EVOLVE-BLOCK-END
