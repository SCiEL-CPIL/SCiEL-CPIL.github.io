"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find integer n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Compute the largest prime factor (LPF) for every integer up to 1,000,001 using a sieve-like approach.
    2. For each n in [10000, 1000000], compute LPF(n * (n+1)) = max(LPF(n), LPF(n+1)).
    3. Find the n that minimizes this value.
    
    We use an array `lpf` where lpf[i] is the largest prime factor of i.
    We can compute this efficiently by iterating through primes and updating multiples.
    """
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # We need LPF up to 1000001
    
    # Initialize LPF array
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    lpf[0] = 0
    lpf[1] = 1
    
    # Sieve for largest prime factor
    # For each number i from 2 to MAX_VAL, if it hasn't been assigned a prime factor yet (i.e., it's prime),
    # then for all multiples of i, set their largest prime factor to i (since we go in increasing order,
    # the last prime that divides a number is its largest prime factor).
    
    # More efficient: iterate through all numbers, and for each prime p, mark all multiples.
    # But we need the LARGEST prime factor. So we can do:
    # lpf[i] starts as 0.
    # For p from 2 to MAX_VAL:
    #   if lpf[p] == 0: p is prime
    #   for multiple m of p: lpf[m] = p
    # Since we iterate p in increasing order, the last assignment to lpf[m] will be from the largest prime factor of m.
    
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:  # p is prime
            for multiple in range(p, MAX_VAL + 1, p):
                lpf[multiple] = p
    
    # Now find the n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = float('inf')
    
    for n in range(10000, MAX_N + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return int(best_n)
# EVOLVE-BLOCK-END
