import math
import time
from collections import defaultdict

def _sigma(n):
    """Compute sum of divisors of n using trial division."""
    if n <= 1:
        return n
    result = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            # d is a prime factor
            term = 1
            while n % d == 0:
                n //= d
                term *= d
            result *= (term * d - 1) // (d - 1)
        d += 1
    if n > 1:
        result *= (n + 1)
    return result

def _factorize(n):
    """Factorize n using trial division, returns dict of prime->exponent."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        pe = 1
        for i in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _sigma_from_factors(factors):
    """Compute sigma from prime factorization."""
    result = 1
    for p, e in factors.items():
        term = 1
        pe = 1
        for i in range(e):
            pe *= p
            term += pe
        result *= term
    return result

def construct():
    start_time = time.time()
    deadline = 130  # 130s safety margin from 150s limit
    
    # Phase 1: Sieve-based enumeration for small k to find candidate M values
    # Use a reasonable sieve limit
    SIEVE_LIMIT = 5_000_000
    
    # Compute sigma for all numbers up to SIEVE_LIMIT using sieve
    sigma_arr = [0] * (SIEVE_LIMIT + 1)
    for i in range(1, SIEVE_LIMIT + 1):
        for j in range(i, SIEVE_LIMIT + 1, i):
            sigma_arr[j] += i
    
    # Build M -> list of k mapping
    m_to_k = defaultdict(list)
    for k in range(2, SIEVE_LIMIT + 1):
        M = k * sigma_arr[k]
        if M <= 10**18:
            m_to_k[M].append(k)
    
    # Find best M from sieve
    best_M = 1
    best_count = 0
    candidate_Ms = []
    for M, ks in m_to_k.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
        # Keep candidates with multiple solutions for extension
        if len(ks) >= 2:
            candidate_Ms.append((M, len(ks)))
    
    # Sort candidates by count descending, then by M ascending
    candidate_Ms.sort(key=lambda x: (-x[1], x[0]))
    
    # Phase 2: Extend top candidates by checking if larger k values also work
    # For each candidate M, we can check k values up to sqrt(M)
    # But we only need to check k > SIEVE_LIMIT
    # We can factor M and enumerate divisors k of M where k <= sqrt(M)
    # and check if sigma(k) = M // k
    
    extended_best = (best_M, best_count)
    
    # Process top candidates
    for M, initial_count in candidate_Ms[:50]:  # Top 50 candidates
        if time.time() - start_time > deadline:
            break
        
        sqrt_M = math.isqrt(M)
        if sqrt_M <= SIEVE_LIMIT:
            continue
        
        # Factorize M to get all divisors
        try:
            factors = _factorize(M)
        except:
            continue
        
        # Get all divisors
        divs = _divisors_from_factors(factors)
        
        # Find all k divisors where k <= sqrt(M) and sigma(k) == M // k
        valid_ks = []
        for k in divs:
            if k <= sqrt_M and k >= 2:
                # Check if sigma(k) == M // k
                if M % k == 0:
                    required_sigma = M // k
                    try:
                        actual_sigma = _sigma(k)
                    except:
                        continue
                    if actual_sigma == required_sigma:
                        valid_ks.append(k)
        
        if len(valid_ks) > extended_best[1]:
            extended_best = (M, len(valid_ks))
    
    # Use the extended best
    final_M, final_count = extended_best
    
    # Collect all valid k for final_M
    final_ks = []
    if final_M > 1:
        sqrt_final = math.isqrt(final_M)
        
        # Check from sieve
        for k in m_to_k.get(final_M, []):
            if k <= sqrt_final:
                final_ks.append(k)
        
        # Check additional k by factoring
        if sqrt_final > SIEVE_LIMIT:
            try:
                factors = _factorize(final_M)
                divs = _divisors_from_factors(factors)
                existing = set(final_ks)
                for k in divs:
                    if k <= sqrt_final and k >= 2 and k not in existing:
                        if final_M % k == 0:
                            required_sigma = final_M // k
                            try:
                                actual_sigma = _sigma(k)
                            except:
                                continue
                            if actual_sigma == required_sigma:
                                final_ks.append(k)
                                existing.add(k)
            except:
                pass
        
        # Remove duplicates and sort
        final_ks = list(set(final_ks))
        final_ks.sort()
    
    return {'M': final_M, 'k': final_ks}
