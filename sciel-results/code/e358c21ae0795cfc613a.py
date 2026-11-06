"""
Optimized smooth two-summand decomposition.
Uses a segmented sieve to compute the largest prime factor for all integers up to 123457,
then performs a linear scan to directly identify the global minimum of max(P(a), P(b)).
"""
import time
import math

def construct():
    deadline = time.time() + 50  # Safety margin
    TARGET = 123457
    
    # Step 1: Sieve of Eratosthenes to get all primes up to TARGET
    # For N = 123457, this is fast
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(TARGET**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, TARGET + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, TARGET + 1) if is_prime[i]]
    
    # Step 2: Compute the largest prime factor P(n) for each n in [1, TARGET]
    # P(1) = 1
    largest_pf = [1] * (TARGET + 1)
    
    # For each prime p, mark all multiples of p: the largest prime factor of p*k is at least p
    # We process primes in increasing order, so the last prime that divides n gives P(n)
    # Actually, we want the LARGEST prime factor, so we process primes in decreasing order
    # and set largest_pf[multiple] = p if not already set by a larger prime.
    # Alternatively, process in increasing order and overwrite - the last one will be the largest.
    # Let's process in increasing order: for each prime p, for each multiple m of p, set largest_pf[m] = p.
    # Since we go in increasing order, the final value is the largest prime factor.
    
    for p in primes:
        for m in range(p, TARGET + 1, p):
            largest_pf[m] = p
    
    # Step 3: Linear scan over all pairs (a, b) where a + b = TARGET
    # We want to minimize max(P(a), P(b))
    # Since a + b = TARGET, b = TARGET - a
    # We scan a from 1 to TARGET-1
    
    best_max = TARGET  # Upper bound: P(TARGET-1) <= TARGET-1
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET):
        b = TARGET - a
        pa = largest_pf[a]
        pb = largest_pf[b]
        current_max = max(pa, pb)
        if current_max < best_max:
            best_max = current_max
            best_a = a
            best_b = b
            # Early termination: if best_max == 2, we can't do better than 2
            # (since both a and b must be > 1, and the smallest prime is 2)
            # Actually, if a=1, P(1)=1, so best_max could be as low as 1 if b is a power of 2... 
            # but P(b) for b>1 is at least 2. So minimum possible is 2.
            # Wait: if a=1, P(a)=1, P(b)=P(TARGET-1). If TARGET-1 is a power of 2, then P(b)=2.
            # But we can't have both P(a)=1 and P(b)=1 unless both are 1, which sums to 2.
            # So the minimum possible max is 2 (if one is 1 and the other is a power of 2) 
            # or higher. Let's not early terminate on 2 unless we're sure.
            # Actually, let's just complete the scan since N is small.
    
    return [best_a, best_b]
