import time
import sys
from math import prod

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N = 1000000
    MAX_VAL = N + 24
    
    # Precompute the small part for all numbers up to MAX_VAL
    # small_part[x] will be the factor of x composed of primes in 'primes'
    small_part = [1] * (MAX_VAL + 1)
    
    # Efficient precomputation using a modified sieve approach:
    # For each prime p, multiply small_part[multiples of p] by p
    # But we need the FULL power of p dividing x.
    # So: for each prime p, for each power p^k, multiply small_part[multiples of p^k] by p
    # This is equivalent to: for each p, for each multiple m of p, small_part[m] *= p while m % p == 0
    
    # Actually, a cleaner approach:
    # small_part[x] = product of p^v_p(x) for p in primes
    # We can compute this by: for each p in primes, for each k >= 1, for each multiple of p^k, multiply by p
    
    # Even simpler: just do the direct calculation but optimized.
    # Let's use the sieve-like approach:
    
    # Initialize small_part to 1
    # For each prime p:
    #   for multiple in range(p, MAX_VAL+1, p):
    #     while small_part[multiple] % p == 0:  # No, this doesn't work directly
    #       ...
    
    # Actually, let's just do it directly but with NumPy for speed if possible,
    # or just accept the O(N * 9 * log) cost.
    
    # Let's try a different approach: for each prime p, compute the p-part of each number.
    # p_part[x] = p^v_p(x)
    # small_part[x] = product of p_part[x] for all p
    
    # We can compute p_part for each p separately and multiply into small_part.
    
    sp = small_part  # alias
    
    for p in primes:
        # For this prime p, multiply sp[m] by p for each p dividing m
        # i.e., sp[m] *= p^v_p(m)
        # We can do: for m in range(p, MAX_VAL+1, p): sp[m] *= p; then for m in range(p*p, MAX_VAL+1, p*p): sp[m] *= p; etc.
        power = p
        while power <= MAX_VAL:
            for m in range(power, MAX_VAL + 1, power):
                sp[m] *= p
            power *= p
    
    # Now sp[i] contains the complete factor of i composed of primes strictly below 24.
    
    best_n = 1000
    min_distinct = 100  # Start high
    start_time = time.time()
    deadline = 85  # Leave 5s margin for overhead/return
    
    # Search for n minimizing the number of distinct values in {sp[n+1], ..., sp[n+24]}
    # Use slicing and set() for speed
    
    for n in range(1000, N + 1):
        if (n & 2047) == 0:
            if time.time() - start_time > deadline:
                break
        
        factors = set(sp[n+1 : n+25])
        distinct_count = len(factors)
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
            if min_distinct == 1:
                break
    
    return best_n

construct()
