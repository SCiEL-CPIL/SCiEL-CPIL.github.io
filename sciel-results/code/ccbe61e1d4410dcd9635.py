import time
import sys
import numpy as np

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N = 1000000
    MAX_VAL = N + 24
    
    # Precompute the small part for all numbers up to MAX_VAL using NumPy
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Vectorized sieve: for each prime power, multiply all multiples
    for p in primes:
        power = p
        while power <= MAX_VAL:
            sp[power::power] *= p
            power *= p
    
    # Convert to list for fast Python slicing in the search loop
    sp_list = sp.tolist()
    
    best_n = 1000
    min_distinct = 100
    start_time = time.time()
    deadline = 85
    
    for n in range(1000, N + 1):
        if (n & 2047) == 0:
            if time.time() - start_time > deadline:
                break
        
        factors = set(sp_list[n+1 : n+25])
        distinct_count = len(factors)
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
            if min_distinct == 1:
                break
    
    return best_n

construct()
