"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def _sigma_sieve(N):
    """Compute sigma(n) for all n up to N using a sieve."""
    sigma = list(range(N + 1))
    for i in range(2, N + 1):
        if sigma[i] == i:  # i is prime
            p_power = i
            while p_power <= N:
                for j in range(p_power, N + 1, p_power):
                    sigma[j] = sigma[j] // p_power * (p_power + i)
                p_power *= i
    return sigma

def _factorize(n):
    """Factorize n using trial division with small primes first."""
    factors = {}
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            factors[d] = exp
        d += 1
    if n > 1:
        factors[n] = 1
    return factors

def _sigma_by_factorization(n):
    """Compute sigma(n) from its prime factorization."""
    if n == 1:
        return 1
    factors = _factorize(n)
    result = 1
    for p, exp in factors.items():
        result *= (p ** (exp + 1) - 1) // (p - 1)
    return result

def _isqrt(n):
    """Integer square root."""
    return math.isqrt(n)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Start with known solution
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Try larger N values to find M with more solutions
    for N in [200000, 500000, 1000000, 2000000, 5000000]:
        if time.time() > deadline - 10:
            break
            
        # Compute sigma up to N
        sigma = _sigma_sieve(N)
        
        # Map M -> list of k
        m_to_k = defaultdict(list)
        for k in range(1, N + 1):
            m = k * sigma[k]
            if m <= 10**18:
                m_to_k[m].append(k)
        
        # Find M with most k values
        for m, k_list in m_to_k.items():
            if len(k_list) > len(best_k):
                best_M = m
                best_k = k_list
        
        if time.time() > deadline - 5:
            break
    
    # Verify the best solution
    if time.time() < deadline - 2:
        verified_k = []
        for k in best_k:
            if k * _sigma_by_factorization(k) == best_M and k <= _isqrt(best_M):
                verified_k.append(k)
        best_k = verified_k
    
    # Ensure we have at least the original solution
    if len(best_k) < 3:
        best_M = 333312
        best_k = [336, 372, 434]
    
    # Convert to proper types
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}
# EVOLVE-BLOCK-END
