import time
import sys

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N = 1000000
    MAX_VAL = N + 24
    
    # Precompute the small part for all numbers up to MAX_VAL
    # small_part[x] will be the factor of x composed of primes in 'primes'
    small_part = [1] * (MAX_VAL + 1)
    
    # Efficient way: for each prime, multiply the small part of multiples
    # But careful: if x is divisible by p, we need to extract ALL p's.
    # The previous method (loop x, then loop primes) is O(N * 9 * log). 
    # N=1e6. 1e6 * 9 * ~20 ops = 1.8e8 ops. This might be too slow in pure Python (10-20s).
    
    # Let's optimize precomputation.
    # We can iterate x from 1 to MAX_VAL.
    # For each x, calculate its small part.
    
    # Alternative: Sieve-like update.
    # Initialize small_part[x] = 1 for all x.
    # This is tricky because we need the full power.
    
    # Let's stick to the direct calculation but optimize it.
    # Actually, 1.8e8 simple integer operations in Python might take ~10-20 seconds.
    # We have 90s. So it's feasible.
    
    # Let's refine the direct calculation loop.
    for i in range(1, MAX_VAL + 1):
        x = i
        f = 1
        for p in primes:
            if x % p == 0:
                while x % p == 0:
                    x //= p
                    f *= p
                if x == 1:
                    break
        small_part[i] = f

    best_n = 1000
    min_distinct = 100  # Start high
    start_time = time.time()
    deadline = 88  # Leave 2s margin for overhead/return
    
    sp = small_part
    # Optimize search loop
    # Using slicing and set() is faster than generator expressions for small lists
    for n in range(1000, N + 1):
        # Check time every 1024 iterations to reduce overhead
        if (n & 1023) == 0 and time.time() - start_time > deadline:
            break
            
        # Slicing creates a list in C, set() consumes it in C
        factors = set(sp[n+1 : n+25])
        distinct_count = len(factors)
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
            if min_distinct == 1:
                break

    return best_n

construct()
