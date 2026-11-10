"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _powerful_part(n):
    """Compute the powerful part of n: product of p^e for prime factors with e>=2."""
    if n <= 1:
        return 1
    result = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            if exp >= 2:
                result *= d ** exp
        d += 1
    # If temp is a remaining prime factor, its exponent is 1, so no contribution
    return result

def _score_pair(n):
    """Compute powerful_part(n) * powerful_part(n+1)."""
    return _powerful_part(n) * _powerful_part(n + 1)

def construct():
    """Return an integer n in [10000, 1000000] maximizing powerful_part(n)*powerful_part(n+1)."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    best_n = 10000
    best_score = _score_pair(10000)
    
    # First pass: scan the entire range [10000, 1000000] to find a good initial candidate
    # This is O(N * sqrt(N)) which might be too slow, so we do a smarter approach.
    
    # Strategy: 
    # 1. Precompute powerful parts for all numbers in range using a sieve-like approach.
    # 2. For each prime p, find all multiples of p^2 in range, and for each, determine the full p^e.
    # 3. Accumulate the powerful part for each number.
    
    N = 1000000
    low = 10000
    
    # powerful_part[i] for i in range [1, N+1]
    # We'll use an array where powerful_part[i] starts at 1 and we multiply in p^e for each prime p
    # where p^e divides i with e >= 2.
    
    pp = [1] * (N + 2)  # 1-indexed, up to N+1
    
    # For each prime p, for each power p^k (k>=2), mark multiples of p^k
    # But we need the FULL p^e where e is the total exponent in i.
    # Better: for each prime p, for each multiple m of p, determine the highest power p^e || m,
    # and if e >= 2, multiply pp[m] by p^e.
    
    # Sieve-based approach:
    # For each p from 2 to sqrt(N):
    #   Check if p is prime (pp[p] hasn't been touched... but pp is for powerful parts, not primality)
    #   Actually, let's use a separate is_prime or just trial.
    
    # Let's do: for p in range(2, int(sqrt(N)) + 1):
    #   if p is prime:
    #     p2 = p * p
    #     for m in range(p2, N + 2, p2):
    #       # m is divisible by p^2
    #       # Find the full exponent of p in m
    #       temp = m
    #       e = 0
    #       while temp % p == 0:
    #         temp //= p
    #         e += 1
    #       if e >= 2:
    #         pp[m] *= p ** e
    
    # This is O(sum_{p prime} N/p^2 * log_p(N)) which is manageable.
    
    # But we need to check primality. Let's do a simple sieve for primes up to sqrt(N).
    limit = int(math.isqrt(N + 1)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    for p in primes:
        p2 = p * p
        if p2 > N + 1:
            break
        # For each multiple of p2 in [1, N+1], determine full p-adic valuation
        # and multiply pp[m] by p^e if e >= 2
        for m in range(p2, N + 2, p2):
            # m is divisible by p^2, so e >= 2
            # Compute full exponent
            temp = m
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            pp[m] *= p ** e
    
    # Now pp[i] contains the powerful part of i for i in [1, N+1]
    # Find best n in [10000, N] such that pp[n] * pp[n+1] is maximized
    
    best_n = low
    best_score = pp[low] * pp[low + 1]
    
    for n in range(low, N):
        sc = pp[n] * pp[n + 1]
        if sc > best_score:
            best_score = sc
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
