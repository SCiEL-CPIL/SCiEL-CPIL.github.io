"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    N_MIN = 10000
    N_MAX = 1000000

    # We need the largest prime factor of n and of n+1 for n in [N_MIN, N_MAX].
    # The largest prime factor of n(n+1) is max(lpf(n), lpf(n+1)).
    # So we need lpf for all integers up to N_MAX + 1.
    limit = N_MAX + 1

    # Sieve: lpf[i] = largest prime factor of i
    lpf = list(range(limit + 1))
    lpf[0] = 0
    if limit >= 1:
        lpf[1] = 1

    # Mark composites with their largest prime factor
    # For each prime p, for multiples of p, set lpf[multiple] = max(lpf[multiple], p)
    # More efficiently: iterate primes and update multiples
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False

    # We'll do a modified sieve to compute largest prime factor
    # Initialize lpf[i] = i (assume prime)
    # For each prime p, for each multiple m of p, lpf[m] = max(lpf[m], p)
    # But this is O(n log log n) if done naively per prime.
    # Better: for each i from 2 to sqrt(limit), if i is prime, for multiples j = i, 2i, 3i...
    #   lpf[j] = max(lpf[j], i)  -- but we want the LARGEST prime factor, so we process
    #   primes in increasing order and just set lpf[j] = i for all multiples? No, that gives
    #   the smallest. We need the largest.
    #
    # Alternative: factor each number by trial division up to sqrt, but that's too slow for 1M.
    #
    # Better sieve approach for LARGEST prime factor:
    # Process primes in DECREASING order. For each prime p (from limit down to 2), if p is prime,
    # then for all multiples m of p that haven't been assigned yet, lpf[m] = p.
    # But checking "haven't been assigned" is tricky.
    #
    # Simpler: use the standard smallest-prime-factor sieve, then for each number, 
    # repeatedly divide to find all prime factors and take the max. But that's per-number work.
    #
    # Actually, the cleanest efficient approach:
    # 1. Sieve to get smallest prime factor (spf) for all numbers up to limit.
    # 2. For each number, factorize using spf and track the largest prime factor.
    # But step 2 for 1M numbers each with ~7 factors max is fine.
    #
    # Even better: we can compute lpf directly.
    # Initialize lpf[i] = 0 for all i.
    # For p from 2 to limit:
    #   if p is prime (lpf[p] == 0 means not yet marked, so p is prime):
    #     for m in range(p, limit+1, p):
    #       lpf[m] = p   # since we go in increasing order, the LAST prime that divides m is the largest
    # Wait, that's not right either. If we go p=2, then p=3, then p=5... for m=30:
    #   p=2: lpf[30]=2
    #   p=3: lpf[30]=3  
    #   p=5: lpf[30]=5
    #   p=... 30 is not divisible by larger primes
    # So lpf[30]=5, which is correct! The largest prime factor.
    # This works because we overwrite with larger primes as we go.
    # Time: sum over primes p of (limit/p) ≈ limit * log(log(limit)) ≈ 1M * ~3 = 3M operations. Fast.

    lpf = [0] * (limit + 1)
    for p in range(2, limit + 1):
        if lpf[p] == 0:  # p is prime
            for m in range(p, limit + 1, p):
                lpf[m] = p

    # Now find n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = float('inf')
    
    # Check elapsed time periodically
    check_interval = 10000
    for n in range(N_MIN, N_MAX + 1):
        val = lpf[n]
        if lpf[n + 1] > val:
            val = lpf[n + 1]
        if val < best_val:
            best_val = val
            best_n = n
        
        if (n - N_MIN) % check_interval == 0:
            if time.time() > deadline:
                break

    return best_n
# EVOLVE-BLOCK-END
