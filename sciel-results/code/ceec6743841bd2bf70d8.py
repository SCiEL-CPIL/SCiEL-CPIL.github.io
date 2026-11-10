"""Combined program: fast linear sieve + multi-candidate extension for max k*sigma(k) collisions."""
import time
import math
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _sigma_linear_sieve(N):
    """Compute sigma(k) for k=1..N using a linear (Euler) sieve. O(N) time."""
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    primes = []
    is_composite = np.zeros(N + 1, dtype=np.bool_)
    # For linear sieve of sigma:
    # sigma is multiplicative. For prime p: sigma(p) = 1+p.
    # For p^a: sigma(p^a) = (p^(a+1)-1)/(p-1)
    # Track: p_pow[n] = highest power of spf[n] dividing n
    #       sigma_p[n] = sigma(p_pow[n])
    #       rest_sig[n] = sigma(n / p_pow[n])
    # Then sigma[n] = sigma_p[n] * rest_sig[n]
    
    p_pow = np.ones(N + 1, dtype=np.int64)
    sigma_p = np.ones(N + 1, dtype=np.int64)
    rest_sig = np.ones(N + 1, dtype=np.int64)
    
    for i in range(2, N + 1):
        if not is_composite[i]:
            primes.append(i)
            sigma_p[i] = 1 + i
            rest_sig[i] = 1
            p_pow[i] = i
            sigma[i] = 1 + i
        for p in primes:
            prod = i * p
            if prod > N:
                break
            is_composite[prod] = True
            if i % p == 0:
                # p divides i, so p_pow[prod] = p_pow[i] * p
                p_pow[prod] = p_pow[i] * p
                # sigma(p^(a+1)) = sigma(p^a) + p^(a+1)
                sigma_p[prod] = sigma_p[i] + p_pow[prod]
                rest_sig[prod] = rest_sig[i]
                sigma[prod] = sigma_p[prod] * rest_sig[prod]
                break
            else:
                # p doesn't divide i, so p_pow[prod] = p, sigma_p = 1+p
                p_pow[prod] = p
                sigma_p[prod] = 1 + p
                rest_sig[prod] = sigma[i]
                sigma[prod] = sigma_p[prod] * rest_sig[prod]
    return sigma

def _sigma_by_factorization(n):
    """Compute sigma(n) by trial division factorization."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            power = 0
            while temp % d == 0:
                temp //= d
                power += 1
            factor = (d ** (power + 1) - 1) // (d - 1)
            result *= factor
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _isqrt(x):
    return math.isqrt(x)

def construct():
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Phase 1: Linear sieve up to N
    N = 5000000  # 5 million
    
    sigma = _sigma_linear_sieve(N)
    
    elapsed1 = time.time() - (deadline - 140)
    # print(f"Sieve done in {elapsed1:.1f}s")
    
    # Phase 2: Group k by M_k = k * sigma(k)
    m_to_k = defaultdict(list)
    
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val <= 10**18:
            m_to_k[M_val].append(k)
    
    elapsed2 = time.time() - (deadline - 140)
    # print(f"Grouping done in {elapsed2:.1f}s")
    
    # Phase 3: Find top candidates
    candidates = []
    for M_val, ks in m_to_k.items():
        if len(ks) >= 2:
            candidates.append((len(ks), M_val, ks))
    
    candidates.sort(key=lambda x: -x[0])
    
    # Phase 4: Extend top candidates by searching divisors > N
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    top_candidates = candidates[:20]
    
    for count, M_val, ks in top_candidates:
        if time.time() > deadline - 25:
            break
        
        sqrt_M = _isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        
        if len(valid_ks) <= best_count:
            continue
        
        # Get all divisors of M_val
        divisors = set()
        i = 1
        while i * i <= M_val:
            if M_val % i == 0:
                divisors.add(i)
                divisors.add(M_val // i)
            i += 1
        
        extended_ks = list(valid_ks)
        existing_ks = set(valid_ks)
        
        for k in sorted(divisors):
            if k > N and k <= sqrt_M and k not in existing_ks:
                expected_sigma = M_val // k
                if _sigma_by_factorization(k) == expected_sigma:
                    extended_ks.append(k)
                    existing_ks.add(k)
            
            if time.time() > deadline - 20:
                break
        
        if len(extended_ks) > best_count:
            best_M = M_val
            best_ks = sorted(extended_ks)
            best_count = len(extended_ks)
    
    # Phase 5: Also try a broader search for M values that might have more solutions
    # by checking if any M from the sieve has solutions we missed
    # (this is a safety net - the sieve should catch all k <= N)
    
    # Final: ensure valid types
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    return {'M': best_M, 'k': best_ks}

# EVOLVE-BLOCK-END
