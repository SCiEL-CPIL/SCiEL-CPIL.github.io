import time
import numpy as np

def construct():
    start_time = time.time()
    # 90s limit, use 80s deadline for safety
    deadline = start_time + 80
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute smooth parts
    # smooth[i] will hold the largest divisor of i composed of primes < 24
    smooth = np.arange(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        for j in range(p, max_val + 1, p):
            # Divide out all factors of p from smooth[j]
            # To avoid Python while loop overhead, we can do this in C-like steps
            # But for clarity and speed, a simple division loop is fine if optimized
            # Actually, we can just do:
            # val = smooth[j]
            # while val % p == 0: val //= p
            # smooth[j] //= val  <-- This is wrong. We want to REMOVE p from the factor.
            # We want smooth[j] to be the part made of small primes.
            # So we should DIVIDE smooth[j] by the large prime factors? No.
            # smooth[j] starts as j.
            # If j has a factor p, we keep p.
            # If j has a factor q >= 24, we remove q.
            # So the algorithm is:
            # Initialize smooth[i] = i.
            # For each prime p >= 24, for each multiple j of p, divide smooth[j] by p until not divisible.
            # This is faster because there are fewer large primes?
            # Primes >= 24: 29, 31, ... up to 1000024.
            # Number of such primes is ~78,000.
            # Iterating over multiples for each large prime is very expensive (N/p sum).
            
            # The original approach:
            # For each small prime p, for each multiple j, multiply smooth_part?
            # No, the original code did:
            # smooth_part = np.ones(...)
            # for p in small_primes:
            #   for multiple in range(p, ...):
            #     temp = multiple
            #     while temp % p == 0:
            #       smooth_part[multiple] *= p
            #       temp //= p
            # This works because smooth_part starts at 1. It builds the smooth part up.
            # Complexity: Sum_{p<24} (N/p * log_p(N)).
            # For p=2, N/2 * 20 ~ 10M ops.
            # Total is manageable.
            
            pass # We will implement the "build up" method or "divis out large primes" method.
    
    # Let's use the "build up" method with NumPy vectorization where possible?
    # Hard to vectorize the "while divisible" part.
    # Let's stick to the efficient Python loop for precomputation, it's ~0.5s.
    
    smooth = np.ones(max_val + 1, dtype=np.int64)
    for p in small_primes:
        for j in range(p, max_val + 1, p):
            temp = j
            factor = 1
            while temp % p == 0:
                factor *= p
                temp //= p
            smooth[j] *= factor
            
    # Now find best n
    best_n = 1000
    min_distinct = float('inf')
    
    # Sliding window
    freq = {}
    # Init window for n=1000 (values 1001 to 1024)
    for i in range(1, 25):
        val = smooth[1000 + i]
        freq[val] = freq.get(val, 0) + 1
        
    cur_distinct = len(freq)
    if cur_distinct < min_distinct:
        min_distinct = cur_distinct
        best_n = 1000
        if min_distinct == 1:
            return best_n
            
    # Slide
    # 1,000,000 iterations
    # To optimize, we can check time every 10000 steps
    for n in range(1001, max_n + 1):
        if n % 10000 == 0 and time.time() > deadline:
            break
        
        # Remove smooth[n]
        v_out = smooth[n]
        freq[v_out] -= 1
        if freq[v_out] == 0:
            del freq[v_out]
            
        # Add smooth[n+24]
        v_in = smooth[n + 24]
        freq[v_in] = freq.get(v_in, 0) + 1
        
        cur_distinct = len(freq)
        if cur_distinct < min_distinct:
            min_distinct = cur_distinct
            best_n = n
            if min_distinct == 1:
                break
                
    return best_n
