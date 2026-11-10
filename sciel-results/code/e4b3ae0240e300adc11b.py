"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 24  # we need values up to 1000024
    T = 24  # primes strictly below 24
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Compute the 23-smooth part (largest divisor with all prime factors < 24)
    # for every number from 1 to MAX_VAL
    smooth_part = list(range(MAX_VAL + 1))  # smooth_part[x] = x initially
    
    for p in primes:
        for multiple in range(p, MAX_VAL + 1, p):
            val = multiple
            # Divide out all powers of p
            while val % p == 0:
                val //= p
            smooth_part[multiple] = smooth_part[multiple] // (multiple // val)
    
    # Actually, let me redo this more carefully.
    # smooth_part[x] should be the largest divisor of x that is 23-smooth.
    # Approach: for each x, smooth_part[x] = x with all prime factors >= 24 removed.
    # Better: start with result = x, then for each prime p >= 24, divide out all powers.
    # But that's expensive. Instead, use the sieve approach:
    # smooth_part[x] = x, then for each prime p in our set, we don't remove p.
    # We want to REMOVE factors that are NOT in our prime set.
    
    # Let me use a different approach:
    # For each x, smooth_part[x] = product of p^a where p is in {2,3,5,7,11,13,17,19,23} and p^a || x
    # Equivalent: x divided by the largest divisor of x that is coprime to 2*3*5*7*11*13*17*19*23
    
    # Let me recompute properly using a sieve for the "rough part" (coprime to primorial)
    primorial = 1
    for p in primes:
        primorial *= p
    # primorial = 2*3*5*7*11*13*17*19*23 = 223092870
    
    # rough_part[x] = x / smooth_part[x] = largest divisor of x coprime to primorial
    # smooth_part[x] = x / rough_part[x]
    
    # Compute rough_part using sieve
    rough_part = list(range(MAX_VAL + 1))
    for p in primes:
        for multiple in range(p, MAX_VAL + 1, p):
            val = multiple
            while val % p == 0:
                val //= p
            # multiple // val = p^a where p^a || multiple
            rough_part[multiple] = rough_part[multiple] // (multiple // val)
    
    # Now smooth_part[x] = x // rough_part[x]
    smooth_part = [0] * (MAX_VAL + 1)
    for x in range(1, MAX_VAL + 1):
        smooth_part[x] = x // rough_part[x]
    
    # Now search for n in [1000, MAX_N] minimizing distinct smooth_part[n+i] for i=1..24
    best_n = 1000
    best_count = float('inf')
    
    # Quick scan
    for n in range(1000, MAX_N + 1):
        if time.time() > deadline:
            break
        # Count distinct smooth parts in window
        seen = set()
        for i in range(1, 25):
            seen.add(smooth_part[n + i])
        cnt = len(seen)
        if cnt < best_count:
            best_count = cnt
            best_n = n
            if best_count <= 1:
                break
    
    return best_n
# EVOLVE-BLOCK-END
