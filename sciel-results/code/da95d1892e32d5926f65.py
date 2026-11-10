"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    LIM = MAX_N + 1  # we need pp up to MAX_N+1
    
    # Build smallest prime factor sieve up to LIM
    spf = list(range(LIM + 1))
    for i in range(2, int(math.isqrt(LIM)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number using a modified sieve approach
    # powerful_part(n) = product of p^e where e >= 2 in factorization
    # We'll use an array-based approach: for each prime p, for each power p^k with k>=2,
    # multiply into all multiples of p^k
    
    pp = [1] * (LIM + 1)
    spf_local = spf
    deadline_local = deadline
    
    # For each prime p, find all powers p^k with k>=2 that are <= LIM
    # Then for each such power, multiply into all multiples
    # This is more efficient than factoring each number individually
    
    for p in range(2, LIM + 1):
        if spf_local[p] != p:
            continue  # not prime
        
        # Find all powers p^k with k >= 2
        power = p * p
        while power <= LIM:
            # Multiply power into all multiples of 'power'
            for m in range(power, LIM + 1, power):
                pp[m] *= power
            power *= p
        
        if time.time() > deadline_local:
            break
    
    # Find best n using NumPy for vectorized max product
    # pp[MIN_N : MAX_N+1] contains powerful parts for n in [MIN_N, MAX_N]
    # We need pp[n] and pp[n+1] for n in [MIN_N, MAX_N]
    # So we need pp[MIN_N : MAX_N+2]
    pp_arr = np.array(pp[MIN_N : MAX_N + 2], dtype=np.int64)
    
    if len(pp_arr) < 2:
        return MIN_N

    # Calculate products of adjacent powerful parts
    # products[i] = pp_arr[i] * pp_arr[i+1] corresponds to n = MIN_N + i
    products = pp_arr[:-1] * pp_arr[1:]
    
    # Find the index of the maximum product
    best_idx = np.argmax(products)
    best_n = MIN_N + best_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
