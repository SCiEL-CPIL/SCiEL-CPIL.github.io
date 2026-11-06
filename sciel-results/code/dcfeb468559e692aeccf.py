"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

# Precompute smallest prime factor (SPF) up to 50000 for fast factorization
_SPF_LIMIT = 50000
_spf = list(range(_SPF_LIMIT + 1))
if _SPF_LIMIT >= 2:
    _spf[0] = 0
    _spf[1] = 1
for i in range(2, int(_SPF_LIMIT ** 0.5) + 1):
    if _spf[i] == i:  # i is prime
        for j in range(i * i, _SPF_LIMIT + 1, i):
            if _spf[j] == j:
                _spf[j] = i

def _factorize_fast(n):
    """Return list of (prime, exponent) for n using precomputed SPF."""
    factors = []
    if n <= 1:
        return factors
    while n > 1:
        p = _spf[n]
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        factors.append((p, e))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from a list of (prime, exponent) pairs."""
    divs = [1]
    for p, e in factors:
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _find_solution_for_n(n, deadline):
    """Find distinct positive integers x,y,z with 1/x+1/y+1/z=4/n. Return [x,y,z] or None."""
    # x must satisfy n/4 < x < 3n/4 (from 1/x < 4/n and 4/n < 3/x)
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    
    for x in range(x_min, x_max + 1):
        if time.time() > deadline:
            return None
        a = 4 * x - n
        b = n * x
        if a <= 0:
            continue
        
        # We need (a*y - b)(a*z - b) = b^2
        # d = a*y - b, so d | b^2, d < b, (d + b) % a == 0, y = (d+b)//a > x
        # z = (b^2//d + b) // a, need z > y, z != x
        
        # Factor b, then double exponents for b^2
        b_factors = _factorize_fast(b)
        b_sq_factors = [(p, 2 * e) for p, e in b_factors]
        divs = _divisors_from_factors(b_sq_factors)
        
        # Filter: d < b, (d + b) % a == 0
        found = None
        for d in divs:
            if d >= b:
                continue
            if (d + b) % a != 0:
                continue
            y = (d + b) // a
            if y <= x:
                continue
            # Check z
            rem = (b * b) // d
            if (rem + b) % a != 0:
                continue
            z = (rem + b) // a
            if z <= y:
                continue
            if z == x or y == x:
                continue
            # Verify equation by cross multiplication
            if n * (y * z + x * z + x * y) == 4 * x * y * z:
                if x <= 1000000000000 and y <= 1000000000000 and z <= 1000000000000:
                    found = [x, y, z]
                    break
        if found:
            return found
    
    return None

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        results[idx] = _find_solution_for_n(n, deadline)
    
    return results
# EVOLVE-BLOCK-END
