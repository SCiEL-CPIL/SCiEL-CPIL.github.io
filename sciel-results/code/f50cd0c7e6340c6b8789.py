"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Strategy:
    1. Precompute primes up to 1000000 using a sieve.
    2. Generate candidate numbers n in increasing order of their "complexity" (number of prime factors).
       Actually, a simpler and more robust approach is to just iterate n from 1 to 1000000,
       but computing phi(n) for each is expensive.
       
    Better Strategy:
    Since we want the MINIMUM n, we can iterate n from 1 upwards.
    To compute phi(n) quickly, we can use the property:
    phi(n) = n * product_{p|n} (1 - 1/p)
    
    We can precompute the smallest prime factor (SPF) for all numbers up to 1000000.
    Then we can compute phi(n) for all n up to 1000000 in O(N log log N) or similar.
    
    However, iterating all n from 1 to 1000000 and checking phi(n) % 120 == 0 is fast enough
    if phi is precomputed efficiently.
    
    Let's precompute phi for all n up to 1000000 using a sieve-like method.
    phi[1] = 1
    For i from 2 to N:
        if phi[i] == 0: # i is prime
            for j from i to N step i:
                phi[j] -= phi[j] // i
    This is O(N log log N).
    
    Then just find the first n where phi[n] % 120 == 0.
    """
    deadline = time.time() + 85  # 5 seconds safety margin
    
    LIMIT = 1000000
    
    # Precompute phi using a sieve
    phi = list(range(LIMIT + 1))
    
    for i in range(2, LIMIT + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, LIMIT + 1, i):
                phi[j] -= phi[j] // i
                if time.time() > deadline:
                    # If we run out of time, we can try to estimate or return a known candidate
                    # But this loop should be fast enough.
                    # Let's just break and hope we've found the answer or return a fallback.
                    # Actually, if we break here, phi is incomplete.
                    # Better to not have a deadline inside the sieve if it's fast enough.
                    # 1000000 * log log 1000000 is roughly 1000000 * 3 = 3,000,000 operations.
                    # Python can do ~10^6 to 10^7 simple ops per second. So it should take < 1s.
                    pass
    
    # Find the smallest n with phi(n) % 120 == 0
    for n in range(1, LIMIT + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback (should never be reached)
    return 14400
# EVOLVE-BLOCK-END
