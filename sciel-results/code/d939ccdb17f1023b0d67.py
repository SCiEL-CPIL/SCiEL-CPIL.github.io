# EVOLVE-BLOCK-START
import math
import time
from math import isqrt
import numpy as np
from collections import defaultdict

# Precompute primes up to a limit to speed up sigma calculation
_PRIME_LIMIT = 200000
_primes = []
_is_prime = [True] * (_PRIME_LIMIT + 1)
_is_prime[0] = False
_is_prime[1] = False
for i in range(2, int(_PRIME_LIMIT**0.5) + 1):
    if _is_prime[i]:
        step = i
        start = i * i
        _is_prime[start::_PRIME_LIMIT+1] = [False] * len(_is_prime[start::_PRIME_LIMIT+1])
for i in range(2, _PRIME_LIMIT + 1):
    if _is_prime[i]:
        _primes.append(i)

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n using precomputed primes."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    for p in _primes:
        if p * p > temp:
            break
        if temp % p == 0:
            p_pow = 1
            while temp % p == 0:
                temp //= p
                p_pow *= p
            result *= (p * p_pow - 1) // (p - 1)
    if temp > 1:
        result *= (temp + 1)
    return result

def _factorize(n):
    """Return list of (prime, exponent) for n."""
    if n <= 1:
        return []
    factors = []
    temp = n
    if temp % 2 == 0:
        cnt = 0
        while temp % 2 == 0:
            temp //= 2
            cnt += 1
        factors.append((2, cnt))
    f = 3
    while f * f <= temp:
        if temp % f == 0:
            cnt = 0
            while temp % f == 0:
                temp //= f
                cnt += 1
            factors.append((f, cnt))
        f += 2
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _all_divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in factors:
        new_divs = []
        p_pow = 1
        for i in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_pow)
            p_pow *= p
        divisors = new_divs
    return divisors

def _find_k_for_M(M):
    """
    Find all k <= sqrt(M) such that k * sigma(k) = M.
    """
    sqrtM = isqrt(M)
    results = []
    factors = _factorize(M)
    divisors = _all_divisors_from_factors(factors)
    
    for k in divisors:
        if k > sqrtM:
            continue
        if k < 1:
            continue
        s = _sigma(k)
        if k * s == M:
            results.append(k)
    return results

def _compute_sigma_sieve(N):
    """
    Compute sigma(n) for n=1..N using a sieve.
    Returns an array of dtype np.int64 (or object if values overflow).
    """
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    return sigma

def _compute_sigma_linear_sieve(N):
    """
    Linear sieve for sigma(n) for n=1..N.
    O(N) time complexity.
    Returns sigma array.
    
    Uses the standard linear sieve approach:
    - spf[n] = smallest prime factor of n
    - ppow[n] = highest power of spf[n] dividing n
    - sigma[n] = sum of divisors of n
    
    Recurrence:
    If p | n and spf[n] == p:
      If ppow[n] == n (i.e., n is a power of p):
        ppow[n*p] = n*p
        sigma[n*p] = sigma[n] + n*p  # sigma(p^(a+1)) = sigma(p^a) + p^(a+1)
      Else:
        ppow[n*p] = ppow[n]
        sigma[n*p] = sigma[n] * (p+1) - sigma[n // ppow[n]] * ppow[n]
    If p !| n:
      ppow[n*p] = p
      sigma[n*p] = sigma[n] * (p+1)
    """
    sigma = np.zeros(N + 1, dtype=np.int64)
    spf = np.zeros(N + 1, dtype=np.int64)
    ppow = np.zeros(N + 1, dtype=np.int64)
    primes = []
    
    sigma[1] = 1
    ppow[1] = 1
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            sigma[i] = i + 1
            ppow[i] = i
        
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            
            if i % p == 0:
                # p divides i
                if ppow[i] == i:
                    # i is a power of p
                    ppow[i * p] = i * p
                    sigma[i * p] = sigma[i] + i * p
                else:
                    ppow[i * p] = ppow[i]
                    # sigma(i*p) = sigma(i) * (p+1) - sigma(i/ppow[i]) * ppow[i]
                    sigma[i * p] = sigma[i] * (p + 1) - sigma[i // ppow[i]] * ppow[i]
                break
            else:
                # p does not divide i
                ppow[i * p] = p
                sigma[i * p] = sigma[i] * (p + 1)
    
    return sigma

def _find_best_collisions(M_vals, k_vals, best_count, best_M, best_ks, deadline):
    """Find best collisions using dictionary-based grouping."""
    # Use dictionary to group by M value
    m_to_ks = defaultdict(list)
    
    for m, k in zip(M_vals, k_vals):
        m_to_ks[int(m)].append(int(k))
        if time.time() > deadline:
            break
    
    for M_val, ks in m_to_ks.items():
        if len(ks) <= best_count:
            continue
        sqrt_M = isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M_val
            best_ks = valid_ks
    
    return best_count, best_M, best_ks

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 second deadline to be safe (150s total limit)
    
    # Fallback known solution
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Strategy:
    # 1. Use the fast linear sieve to compute sigma(k) for k up to L.
    # 2. Compute M = k * sigma(k) for all k.
    # 3. Find collisions in M using dictionary grouping.
    # 4. For each M with many collisions, verify k <= sqrt(M).
    # 5. If time permits, extend the search range.
    
    # Start with a larger L thanks to O(N) sieve
    L = 20_000_000
    
    try:
        sigma = _compute_sigma_linear_sieve(L)
    except Exception:
        try:
            sigma = _compute_sigma_sieve(10_000_000)
            L = 10_000_000
        except Exception:
            L = 5_000_000
            try:
                sigma = _compute_sigma_sieve(L)
            except Exception:
                return {'M': best_M, 'k': best_ks}
            
    if time.time() > deadline:
        return {'M': best_M, 'k': best_ks}
        
    k_vals = np.arange(1, L + 1, dtype=np.int64)
    M_vals = k_vals * sigma[1:L+1]
    
    # Filter M_vals <= 10^18
    mask = M_vals <= 10**18
    M_vals = M_vals[mask]
    k_vals = k_vals[mask]
    
    if len(M_vals) == 0:
        return {'M': best_M, 'k': best_ks}

    # Use dictionary-based collision finding
    best_count, best_M, best_ks = _find_best_collisions(
        M_vals, k_vals, best_count, best_M, best_ks, deadline
    )

    # If we have time, try an even larger L
    remaining_time = deadline - time.time()
    if remaining_time > 25:
        L2 = 40_000_000
        try:
            sigma2 = _compute_sigma_linear_sieve(L2)
        except Exception:
            sigma2 = None
        
        if sigma2 is not None and time.time() < deadline - 10:
            k_vals2 = np.arange(1, L2 + 1, dtype=np.int64)
            M_vals2 = k_vals2 * sigma2[1:L2+1]
            
            mask2 = M_vals2 <= 10**18
            M_vals2 = M_vals2[mask2]
            k_vals2 = k_vals2[mask2]
            
            if len(M_vals2) > 0:
                best_count, best_M, best_ks = _find_best_collisions(
                    M_vals2, k_vals2, best_count, best_M, best_ks, deadline
                )

    if best_count == 0:
        return {'M': best_M, 'k': best_ks}
        
    best_ks = [int(k) for k in best_ks]
    
    # Final verification
    sqrtM = isqrt(best_M)
    verified_ks = []
    for k in best_ks:
        if k <= sqrtM and k * _sigma(k) == best_M:
            verified_ks.append(k)
    
    verified_ks = list(set(verified_ks))
    
    if len(verified_ks) == 0:
        return {'M': best_M, 'k': best_ks}
    
    return {'M': int(best_M), 'k': verified_ks}
# EVOLVE-BLOCK-END
