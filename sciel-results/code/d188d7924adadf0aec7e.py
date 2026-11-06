import time
import math

def _factorize(num):
    """Return prime factorization as dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors[d] = factors.get(d, 0) + 1
            num //= d
        d += 1 if d == 2 else 2
    if num > 1:
        factors[num] = factors.get(num, 0) + 1
    return factors

def _divisors_of(factors):
    """Generate all divisors from prime factorization dict."""
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    divisors = [1]
    for p, e in zip(primes, exponents):
        current = []
        power = 1
        for i in range(e + 1):
            for d in divisors:
                current.append(d * power)
            power *= p
        divisors = current
    return divisors

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    MAX_DENOM = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # Precompute divisors of n^2
        fact_n = _factorize(n)
        fact_n2 = {p: 2*e for p, e in fact_n.items()}
        divs_n2 = _divisors_of(fact_n2)
        n2 = n * n
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            # Need u | n^2, v = n^2/u, (u+n) % a == 0, (v+n) % a == 0
            # y = (u+n)/a > x, z = (v+n)/a > y, y,z <= MAX_DENOM
            
            for u in divs_n2:
                if (u + n) % a != 0:
                    continue
                v = n2 // u
                if (v + n) % a != 0:
                    continue
                y = (u + n) // a
                z = (v + n) // a
                if y <= x:
                    continue
                if z <= y:
                    continue
                if y > MAX_DENOM or z > MAX_DENOM:
                    continue
                found = [x, y, z]
                break
            if found:
                break
        
        results[idx] = found
    
    return results
