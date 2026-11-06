"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def _factorize(n):
    """Return dict of prime factors -> exponents."""
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
    if not factors:
        return [1]
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    divs = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divs:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    return divs

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # x range: n/4 < x < 3n/4
        x_lo = n // 4 + 1
        x_hi = (3 * n + 3) // 4
        
        for x in range(x_lo, x_hi + 1):
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # (a*y - b)(a*z - b) = b^2
            # Let d1 = a*y - b, d2 = a*z - b, d1*d2 = b^2
            # y = (d1 + b)/a, z = (d2 + b)/a
            # Need d1 | b^2, d2 = b^2/d1, both (d1+b) and (d2+b) divisible by a
            # And y, z > 0, distinct from x and each other, <= 10^12
            
            b_sq = b * b
            factors = _factorize(b_sq)
            divs = _divisors_from_factors(factors)
            
            # For each divisor d1 of b_sq, check if it gives valid y, z
            for d1 in divs:
                d2 = b_sq // d1
                # y = (d1 + b) / a, z = (d2 + b) / a
                num_y = d1 + b
                num_z = d2 + b
                if num_y % a != 0 or num_z % a != 0:
                    continue
                y = num_y // a
                z = num_z // a
                # Ensure y <= z (we can swap, but let's just check both orderings or enforce)
                # WLOG assume y <= z by taking d1 <= d2
                if d1 > d2:
                    continue
                if y <= 0 or z <= 0:
                    continue
                if y > 1000000000000 or z > 1000000000000:
                    continue
                # Distinctness: x, y, z all distinct
                if x == y or x == z or y == z:
                    continue
                found = [x, y, z]
                break
            
            if found:
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
