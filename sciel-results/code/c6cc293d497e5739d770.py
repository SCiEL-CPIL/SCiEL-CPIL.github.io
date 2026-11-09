"""Solve 1/x+1/y+1/z=4/n for n=3..256 with distinct denominators."""
import math
import time
from collections import defaultdict

def _factorize(n):
    """Return prime factorization of n as list of (prime, exponent)."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            factors.append((d, exp))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
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

def construct():
    deadline = time.time() + 85
    answers = [None] * 254
    LIMIT = 10**12
    
    # Precompute factorizations for values we'll need
    max_x = 5000
    factor_cache = {}
    for v in range(1, max_x + 1):
        factor_cache[v] = _factorize(v)
    
    for idx in range(254):
        if time.time() > deadline:
            break
        n = idx + 3
        found = None
        
        x_min = n // 4 + 1
        x_max = min(4 * n + 10, max_x)
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Combine factorizations of n^2 * x^2
            combined = defaultdict(int)
            for p, e in factor_cache[n]:
                combined[p] += 2 * e
            for p, e in factor_cache[x]:
                combined[p] += 2 * e
            
            bf = list(combined.items())
            divs = _divisors_from_factors(bf)
            
            for d in divs:
                if time.time() > deadline:
                    break
                e_val = b_sq // d
                num_y = d + b
                num_z = e_val + b
                if num_y % a != 0 or num_z % a != 0:
                    continue
                y = num_y // a
                z = num_z // a
                if y <= 0 or z <= 0:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                if x == y or x == z or y == z:
                    continue
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        
        answers[idx] = found
    
    return answers
