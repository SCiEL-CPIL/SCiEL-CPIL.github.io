"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import log

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin
    
    N = 123457
    
    # Precompute largest prime factor for all numbers up to N
    # Using a sieve approach: for each prime p, mark all multiples with p as a factor
    lpf = [1] * (N + 1)
    
    # Sieve for primes and compute largest prime factor
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i  # i is a prime factor of j, and since we go in increasing order, last one is largest
    
    # Now lpf[n] contains the largest prime factor of n for all n in [2, N]
    # lpf[1] = 1 by definition
    
    # Iterate through all splits a + b = N, minimize max(lpf[a], lpf[b])
    best_max_pf = float('inf')
    best_a = 1
    
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        m = max(lpf[a], lpf[b])
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
    
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
