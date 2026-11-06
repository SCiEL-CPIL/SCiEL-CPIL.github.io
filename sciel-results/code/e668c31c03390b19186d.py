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

def _score(n):
    if n < 2:
        return 0
    # Factorize n
    temp_n = n
    factors = {}
    d = 2
    while d * d <= temp_n:
        while temp_n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp_n //= d
        d += 1
    if temp_n > 1:
        factors[temp_n] = factors.get(temp_n, 0) + 1
    
    divisors = _get_divisors_from_factors(factors)
    divisors.sort()
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def construct():
    # Base primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Calculate Primorial P15
    p15 = 1
    for p in primes:
        p15 *= p
        
    best_n = p15
    best_score = _score(p15)
    
    # Calculate P14
    p14 = p15 // 47
    
    # Search P14 * k for k up to floor(10^18 / P14)
    limit_k = 10**18 // p14
    for k in range(1, limit_k + 1):
        n = p14 * k
        if n <= 10**18:
            s = _score(n)
            if s > best_score:
                best_score = s
                best_n = n
    
    # Search P13 * k for a smaller range
    p13 = p14 // 43
    limit_k13 = 10**18 // p13
    # We can't iterate all k up to 3333, so we check a subset or small range
    # Let's check k up to 100
    for k in range(1, min(100, limit_k13) + 1):
        n = p13 * k
        if n <= 10**18:
            s = _score(n)
            if s > best_score:
                best_score = s
                best_n = n
                
    return best_n
# EVOLVE-BLOCK-END
