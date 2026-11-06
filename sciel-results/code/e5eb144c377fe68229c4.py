"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _get_divisors_from_factors(factors):
    divisors = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    return divisors

def _factor_small(n):
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

def _score_from_factors(factors):
    if not factors:
        return 0
    divisors = _get_divisors_from_factors(factors)
    divisors.sort()
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    # Precompute primorials
    primorials = [1]
    for p in primes:
        primorials.append(primorials[-1] * p)
        
    best_n = 1
    best_score = 0
    
    # We search for n = P_k * m
    # k is the number of primes in P_k
    # We iterate k from high to low
    # Limit m to keep runtime reasonable
    
    limits = {
        15: 10**18 // primorials[15], # ~1
        14: 10**18 // primorials[14], # ~77
        13: 10**18 // primorials[13], # ~3333
        12: 10**18 // primorials[12], # ~33333
        11: 10**18 // primorials[11], # ~333333
        10: 10**18 // primorials[10]  # ~3333333
    }
    
    # Cap the search ranges to avoid timeout
    max_m_search = {
        15: limits[15],
        14: limits[14],
        13: limits[13],
        12: min(limits[12], 5000),
        11: min(limits[11], 500),
        10: min(limits[10], 100)
    }
    
    for k in range(15, 9, -1):
        P_k = primorials[k]
        # Factors of P_k
        P_k_factors = {primes[i]: 1 for i in range(k)}
        
        max_m = max_m_search[k]
        # If max_m is very large, we might want to sample, but for now we just cap
        # Actually, for k=10, 100 is small.
        
        for m in range(1, max_m + 1):
            n = P_k * m
            if n > 10**18:
                break
            
            # Factor m
            m_factors = _factor_small(m)
            
            # Merge factors
            total_factors = P_k_factors.copy()
            for p, e in m_factors.items():
                total_factors[p] = total_factors.get(p, 0) + e
            
            s = _score_from_factors(total_factors)
            
            if s > best_score:
                best_score = s
                best_n = n
                
    return best_n
# EVOLVE-BLOCK-END
