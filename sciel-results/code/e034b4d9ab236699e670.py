"""Optimized distinct three-term unit-fraction representations of 4/n."""
import time
from math import gcd
from collections import Counter

def _factorize(n):
    """Trial division factorization, returns dict {prime: exponent}."""
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

def _gen_divisors_b2_filtered(b, a, x):
    """
    Generate divisors d of b^2 such that (d+b) % a == 0 and (b^2/d + b) % a == 0,
    yielding valid (y, z) pairs with y > x, z > y, all <= 10^12.
    Returns list of (x, y, z) triples.
    """
    b_fac = _factorize(b)
    # b^2 has factorization {p: 2*exp for p, exp in b_fac}
    b_sq = b * b
    
    # Enumerate divisors of b^2 via recursion on prime factors
    primes = list(b_fac.keys())
    exponents = [2 * b_fac[p] for p in primes]
    
    solutions = []
    
    def _enumerate(idx, current_d):
        if idx == len(primes):
            d = current_d
            if d > b_sq:
                return
            # Check (d + b) % a == 0
            if (d + b) % a != 0:
                return
            y_val = (d + b) // a
            if y_val <= x or y_val > 10**12:
                return
            rem = b_sq // d
            if (rem + b) % a != 0:
                return
            z_val = (rem + b) // a
            if z_val <= y_val or z_val > 10**12:
                return
            if x != y_val and y_val != z_val and x != z_val:
                solutions.append((x, y_val, z_val))
            return
        
        p = primes[idx]
        exp = exponents[idx]
        p_pow = 1
        for e in range(exp + 1):
            _enumerate(idx + 1, current_d * p_pow)
            p_pow *= p
    
    _enumerate(0, 1)
    return solutions

def _find_solution(n, deadline, _factor_cache=None):
    """Find a distinct triple (x,y,z) with 1/x+1/y+1/z=4/n, all <= 10^12."""
    if _factor_cache is None:
        _factor_cache = {}
    
    x_min = n // 4 + 1
    # x must satisfy 4/n - 1/x > 0, so x > n/4
    # Also need 4/n - 1/x >= 2/z_max, but let's just bound x < 4n/3 for safety
    x_max = (4 * n) // 3
    
    # Allow scanning more x values
    scan_limit = min(x_max + 1, x_min + 200000)
    
    for x in range(x_min, scan_limit):
        if time.time() > deadline:
            return None
        
        a = 4 * x - n
        if a <= 0:
            continue
        b = n * x
        
        # 1/y + 1/z = a/b
        # y > x, and 1/y < a/b so y > b/a
        # Also 1/z < a/b so z > b/a, and y < 2b/a (since 1/y > a/b - 1/z > 0)
        y_lo = max(x + 1, b // a + 1)
        y_hi = 2 * b // a
        
        if y_hi <= y_lo:
            continue
        
        range_size = y_hi - y_lo + 1
        
        if range_size > 5000:
            # Use factorization method
            sols = _gen_divisors_b2_filtered(b, a, x)
            if sols:
                return list(sols[0])
        else:
            # Direct iteration over y
            for y in range(y_lo, y_hi + 1):
                if time.time() > deadline:
                    return None
                num = a * y - b
                if num <= 0:
                    continue
                den = b * y
                if den % num != 0:
                    continue
                z = den // num
                if z <= y or z > 10**12:
                    continue
                if x != y and y != z and x != z:
                    return [x, y, z]
    
    return None

def _precompute_all():
    """Precompute solutions for all n in [3, 256]."""
    start = time.time()
    deadline = start + 75.0  # Leave 15s margin for safety
    table = {}
    _factor_cache = {}
    
    for n in range(3, 257):
        if time.time() > deadline:
            table[n] = None
            continue
        table[n] = _find_solution(n, deadline, _factor_cache)
    
    return table

_SOLUTIONS = _precompute_all()

def construct():
    """Return 254 rows for n=3..256 from precomputed table."""
    return [_SOLUTIONS[n] for n in range(3, 257)]
