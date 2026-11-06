"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Max value needed for f(x) is for x up to 1000000 + 24
    max_x = 1000024
    f = [1] * (max_x + 1)
    
    # Sieve-like precomputation of small-prime parts
    for p in primes:
        for j in range(p, max_x + 1, p):
            f[j] *= p

    # Sliding window logic
    # Window covers n+1 to n+24
    # Start with n=1000
    counts = {}
    distinct = 0
    
    # Initialize window for n=1000
    for i in range(1, 25):
        v = f[1000 + i]
        if v in counts:
            counts[v] += 1
        else:
            counts[v] = 1
            distinct += 1
            
    best_n = 1000
    best_distinct = distinct

    # Slide n from 1001 to 1000000
    for n in range(1001, 1000001):
        # Remove element at n (which was n-1+1)
        v_out = f[n]
        counts[v_out] -= 1
        if counts[v_out] == 0:
            distinct -= 1
            
        # Add element at n+24
        v_in = f[n + 24]
        if v_in in counts:
            counts[v_in] += 1
        else:
            counts[v_in] = 1
            distinct += 1
            
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            
        # Check time
        if time.time() > deadline:
            break

    return best_n
# EVOLVE-BLOCK-END
