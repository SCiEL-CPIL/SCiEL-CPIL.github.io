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

def _try_closed_form(n):
    """Try known closed-form patterns for 4/n. Returns [x,y,z] or None."""
    BIG = 10**12
    # Pattern 1: n = 3m, m odd (n ≡ 3 mod 6): x=m, y=3m+1, z=3m(3m+1)
    if n % 6 == 3:
        m = n // 3
        x, y, z = m, 3*m + 1, 3*m*(3*m + 1)
        if x != y and y != z and x != z and 0 < x <= BIG and 0 < y <= BIG and 0 < z <= BIG:
            if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                return [x, y, z]
    
    # Pattern 2: n = 4m, m odd (n ≡ 4 mod 8): x=m+2, y=(m+1)(m+2), z=m(m+1)
    if n % 8 == 4:
        m = n // 4
        x, y, z = m + 2, (m + 1)*(m + 2), m*(m + 1)
        if x != y and y != z and x != z and 0 < x <= BIG and 0 < y <= BIG and 0 < z <= BIG:
            if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                return [x, y, z]
    
    # Pattern 3: n = 5m, m odd (n ≡ 5 mod 10): x=2m, y=5m, z=10m
    if n % 10 == 5:
        m = n // 5
        x, y, z = 2*m, 5*m, 10*m
        if x != y and y != z and x != z and 0 < x <= BIG and 0 < y <= BIG and 0 < z <= BIG:
            if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                return [x, y, z]
    
    # Pattern 4: n divisible by 8: x=n/4, y=n/2, z=n
    if n % 8 == 0:
        x, y, z = n // 4, n // 2, n
        if x != y and y != z and x != z and x >= 1:
            if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                return [x, y, z]
    
    return None

def construct():
    deadline = time.time() + 85
    answers = [None] * 254
    LIMIT = 10**12
    
    # Precompute factorizations for values we'll need
    max_x = 6000
    factor_cache = {}
    for v in range(1, max_x + 1):
        factor_cache[v] = _factorize(v)
    
    for idx in range(254):
        if time.time() > deadline:
            break
        n = idx + 3
        found = None
        
        # First try closed-form patterns
        cf = _try_closed_form(n)
        if cf is not None:
            answers[idx] = cf
            continue
        
        # Fallback: search with divisor enumeration
        # Strategy: try x = n+1 first (high success rate), then expand
        x_candidates = [n + 1]
        # Add nearby values
        x_min = n // 4 + 1
        x_max = min(4 * n + 10, max_x)
        
        # Build ordered list: n+1 first, then sequential from x_min
        seen = set()
        x_list = []
        for x in [n + 1]:
            if x_min <= x <= x_max and x not in seen:
                x_list.append(x)
                seen.add(x)
        for x in range(x_min, x_max + 1):
            if x not in seen:
                x_list.append(x)
        
        for x in x_list:
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
