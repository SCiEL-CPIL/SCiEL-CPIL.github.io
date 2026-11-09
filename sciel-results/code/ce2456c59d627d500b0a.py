import time
import sys

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N = 1000000
    MAX_VAL = N + 24
    
    # Precompute the small part for all numbers up to MAX_VAL
    # small_part[x] will be the factor of x composed of primes in 'primes'
    sp = [1] * (MAX_VAL + 1)
    
    # Efficient precomputation using a sieve-like approach:
    # For each prime p, for each power p^k <= MAX_VAL,
    # multiply sp[m] by p for all m that are multiples of p^k.
    # This correctly accumulates p^v_p(x) for each x.
    # Complexity: sum over primes p of (MAX_VAL/p + MAX_VAL/p^2 + ...) ≈ MAX_VAL * sum(1/(p-1))
    # which is much faster than the O(N * P * log) direct factorization.
    
    for p in primes:
        power = p
        while power <= MAX_VAL:
            for m in range(power, MAX_VAL + 1, power):
                sp[m] *= p
            power *= p
    
    best_n = 1000
    min_distinct = 100  # Start high
    start_time = time.time()
    deadline = 85  # Leave 5s margin for overhead/return
    
    # Search for n minimizing the number of distinct values in {sp[n+1], ..., sp[n+24]}
    # Use slicing and set() for speed
    
    for n in range(1000, N + 1):
        # Check time every 2048 iterations to reduce overhead
        if (n & 2047) == 0:
            if time.time() - start_time > deadline:
                break
        
        # Slicing creates a list in C, set() consumes it in C
        factors = set(sp[n+1 : n+25])
        distinct_count = len(factors)
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
            # If we found the theoretical minimum (1), we can stop
            if min_distinct == 1:
                break
    
    return best_n

construct()
