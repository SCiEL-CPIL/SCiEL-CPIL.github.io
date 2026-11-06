"""Optimized hybrid divisor + y-scan for 4/n unit fractions."""
# EVOLVE-BLOCK-START
import time
from math import gcd

# Precompute small primes up to 100000 for fast factorization
def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

_PRIMES = _sieve(100000)

def _factorize(n):
    """Return dict of prime -> exponent for n using precomputed primes."""
    factors = {}
    for p in _PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _try_y_scan(x, n, deadline, max_iters=200000):
    """Try to find y, z by scanning y values. Returns [y, z] or None."""
    A = 4 * x - n
    if A <= 0:
        return None
    B = n * x
    y_min = max(x + 1, B // A + 1)
    y_max = 2 * B // A
    if y_max < y_min:
        return None
    count = 0
    for y in range(y_min, y_max + 1):
        count += 1
        if count > max_iters:
            return None
        if (count & 4095) == 0 and time.time() > deadline:
            return None
        denom = A * y - B
        if denom <= 0:
            continue
        if (B * y) % denom == 0:
            z = (B * y) // denom
            if z > y and z <= 10**12:
                return [y, z]
    return None

def _try_divisor_method(x, n, deadline):
    """Try to find y, z using divisor enumeration of B^2. Returns [y, z] or None."""
    A = 4 * x - n
    if A <= 0:
        return None
    B = n * x
    y_min = max(x + 1, B // A + 1)
    y_max = 2 * B // A
    if y_max < y_min:
        return None
    fb = _factorize(B)
    fb2 = {p: 2 * e for p, e in fb.items()}
    num_divs = 1
    for e in fb2.values():
        num_divs *= (e + 1)
        if num_divs > 100000:
            return None
    B2 = B * B
    divs = _divisors_from_factors(fb2)
    for d in divs:
        if (divs.index(d) & 4095) == 0 and time.time() > deadline:
            return None
        if d >= B:
            continue
        num_y = d + B
        if num_y % A != 0:
            continue
        y = num_y // A
        if y <= x or y < y_min or y > y_max:
            continue
        num_z = B2 // d + B
        if num_z % A != 0:
            continue
        z = num_z // A
        if z <= y or z > 10**12:
            continue
        return [y, z]
    return None

def _find_solution_for_n(n, deadline):
    """Find distinct positive integer denominators x,y,z with 1/x+1/y+1/z=4/n."""
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    # Scan x from largest to smallest: smaller y-range, faster
    for x in range(x_max, x_min - 1, -1):
        if time.time() > deadline:
            return None
        A = 4 * x - n
        if A <= 0:
            continue
        result = _try_y_scan(x, n, deadline)
        if result is not None:
            return [x] + result
        result = _try_divisor_method(x, n, deadline)
        if result is not None:
            return [x] + result
    return None

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        results[idx] = _find_solution_for_n(n, deadline)
    return results
# EVOLVE-BLOCK-END
