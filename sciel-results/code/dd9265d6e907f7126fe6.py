import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute sf[m] = product of p^k for p < 24, p^k || m
    # sf[m] is the "small prime part"
    sf = np.ones(max_val + 1, dtype=np.int64)
    
    # Sieve-like multiplication for each prime power
    for p in primes:
        pk = p
        while pk <= max_val:
            # For all multiples of pk, multiply sf by p
            # This adds one factor of p for each power of p dividing the number
            # Since we do this for p^1, p^2, ..., the total multiplication for m
            # will be p^(v_p(m)), which is exactly the part of m composed of p.
            indices = np.arange(pk, max_val + 1, pk)
            sf[indices] *= p
            pk *= p
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
            
    # Now find n in [1000, 1000000] minimizing distinct count in sf[n+1:n+25]
    
    best_n = min_n
    best_count = 25  # Max possible is 24 distinct values
    
    # Sliding window frequency counter
    # sf values can be large, so we use a dictionary
    freq = defaultdict(int)
    distinct_count = 0
    
    # Initialize window for n = min_n
    # Window covers indices min_n+1 to min_n+24
    for i in range(min_n + 1, min_n + 25):
        val = sf[i]
        if freq[val] == 0:
            distinct_count += 1
        freq[val] += 1
        
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = min_n
        
    # Slide window from n = min_n + 1 to max_n
    for n in range(min_n + 1, max_n + 1):
        if time.time() > deadline - 3:
            break
            
        # Remove element leaving the window: sf[n]
        # The previous window was for n-1, covering (n-1)+1 to (n-1)+24 => n to n+23
        # The new window is for n, covering n+1 to n+24
        # So we remove sf[n] and add sf[n+24]
        
        val_out = sf[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct_count -= 1
            
        val_in = sf[n + 24]
        if freq[val_in] == 0:
            distinct_count += 1
        freq[val_in] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                return best_n
                
    return best_n
