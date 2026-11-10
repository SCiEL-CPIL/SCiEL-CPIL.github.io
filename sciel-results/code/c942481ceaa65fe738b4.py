# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def _sigma_sieve(limit):
    """Compute sigma(k) for all k up to limit using a sieve."""
    sigma = [0] * (limit + 1)
    for i in range(1, limit + 1):
        for j in range(i, limit + 1, i):
            sigma[j] += i
    return sigma

def _factorize(n, primes=None):
    """Factorize n. If primes list given, use it; else trial division."""
    factors = {}
    if primes is None:
        # Generate primes up to sqrt(n)
        limit = int(math.isqrt(n)) + 1
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        primes = [i for i in range(2, limit+1) if is_prime[i]]
    
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            factors[p] = cnt
    if temp > 1:
        factors[temp] = 1
    return factors

def _sigma_from_factors(factors):
    """Compute sigma from prime factorization dict {p: a}."""
    result = 1
    for p, a in factors.items():
        result *= (p**(a+1) - 1) // (p - 1)
    return result

def _sigma_by_trial(n):
    """Compute sigma(n) via trial division."""
    if n <= 1:
        return 1 if n == 1 else 0
    temp = n
    result = 1
    f = 2
    while f * f <= temp:
        if temp % f == 0:
            p_power = f
            cnt = 0
            while temp % f == 0:
                temp //= f
                cnt += 1
                p_power *= f
            result *= (p_power * f - 1) // (f - 1)
        f += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _find_best_m(sigma, limit, deadline):
    """Group k by M = k*sigma(k), find M with most k values."""
    m_to_ks = defaultdict(list)
    for k in range(1, limit + 1):
        if k % 1000000 == 0 and time.time() > deadline:
            break
        M = k * sigma[k]
        if M <= 10**18:
            m_to_ks[M].append(k)
    
    best_m = None
    best_ks = []
    for M, ks in m_to_ks.items():
        if len(ks) > len(best_ks):
            best_m = M
            best_ks = ks
    return best_m, best_ks

def _extend_solutions(M, best_ks, deadline):
    """Try to find additional k values for M beyond what sieve found."""
    existing = set(best_ks)
    sqrt_m = math.isqrt(M)
    additional = []
    
    # Generate candidate k values: divisors of M that are <= sqrt_m
    # k must satisfy: k * sigma(k) = M, so sigma(k) = M/k, meaning k | M
    # Also k <= sqrt(M)
    
    # Get all divisors of M up to sqrt(M)
    divisors = []
    i = 1
    while i * i <= M:
        if M % i == 0:
            if i <= sqrt_m:
                divisors.append(i)
            other = M // i
            if other != i and other <= sqrt_m:
                divisors.append(other)
        i += 1
        if time.time() > deadline:
            break
    
    for k in divisors:
        if k in existing:
            continue
        if k < 1 or k > sqrt_m:
            continue
        # Check if k * sigma(k) == M
        s = _sigma_by_trial(k)
        if k * s == M:
            additional.append(k)
    
    return additional

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    # Known good fallback
    best_m = 333312
    best_ks = [336, 372, 434]
    
    # Try increasing sieve limits
    limits_to_try = [100000, 300000, 600000, 1200000, 2000000, 3000000, 5000000]
    
    for limit in limits_to_try:
        if time.time() > deadline - 30:
            break
        
        sigma = _sigma_sieve(limit)
        m, ks = _find_best_m(sigma, limit, deadline - 10)
        
        if m is not None and len(ks) > len(best_ks):
            best_m = m
            best_ks = ks
        
        if time.time() > deadline - 15:
            break
    
    # Extension phase: use best M to find more k values
    if time.time() < deadline - 5:
        extra = _extend_solutions(best_m, best_ks, deadline - 2)
        if extra:
            best_ks = best_ks + extra
            best_ks = list(set(best_ks))
    
    # Ensure types and constraints
    best_m = int(best_m)
    sqrt_m = math.isqrt(best_m)
    verified_ks = []
    for k in best_ks:
        k = int(k)
        if 1 <= k <= sqrt_m:
            verified_ks.append(k)
    verified_ks = verified_ks[:1000]
    
    if not verified_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_m, 'k': verified_ks}
# EVOLVE-BLOCK-END
