"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _build_spf(limit):
    """Build smallest prime factor sieve up to limit."""
    spf = list(range(limit + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _powerful_part(n, spf):
    """Compute the powerful part of n: product of p^e for each prime power p^e || n with e >= 2."""
    if n <= 1:
        return 1
    pp = 1
    while n > 1:
        p = spf[n]
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        if e >= 2:
            pp *= p ** e
    return pp

def _precompute_powerful_parts(limit, spf):
    """Precompute powerful parts for all n in [1, limit]."""
    pp = [1] * (limit + 1)
    for n in range(2, limit + 1):
        x = n
        pprod = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pprod *= p ** e
        pp[n] = pprod
    return pp

def construct():
    start_time = time.time()
    deadline = 90.0  # safety margin: evaluator kills at 90s, use ~85s
    
    # We need to search n in [10000, 1000000]
    # We need powerful_part(n) and powerful_part(n+1)
    # So we need to compute up to 1000001
    limit = 1000001
    
    # Build SPF
    spf = _build_spf(limit)
    
    # Precompute powerful parts
    pp = _precompute_powerful_parts(limit, spf)
    
    # Check time
    if time.time() - start_time > deadline - 2:
        return 10000
    
    # Search for the pair (n, n+1) that maximizes pp[n] * pp[n+1]
    best_n = 10000
    best_val = pp[10000] * pp[10001]
    
    for n in range(10000, 1000000):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        
        # Time check every 10000 iterations
        if (n - 10000) % 10000 == 0 and time.time() - start_time > deadline - 5:
            break
    
    return best_n
# EVOLVE-BLOCK-END
