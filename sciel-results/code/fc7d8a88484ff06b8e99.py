"""Direct lookup for Egyptian fraction of 5/31 with minimal largest denominator."""

from fractions import Fraction
import time

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256


def _verify(denoms):
    """Verify a set of denominators sums to TARGET and meets constraints."""
    if not denoms:
        return False
    if len(denoms) > MAX_TERMS:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return False
    s = sum(Fraction(1, d) for d in denoms)
    return s == TARGET


def _search_3term(max_d):
    """Find 3 distinct denominators a<b<c<=max_d with 1/a+1/b+1/c=5/31."""
    for a in range(7, min(max_d, 50) + 1):
        for b in range(a + 1, min(max_d, 200) + 1):
            denom = 5 * a * b - 31 * a - 31 * b
            if denom <= 0:
                continue
            num = 31 * a * b
            if num % denom == 0:
                c = num // denom
                if c > b and c <= max_d:
                    return [a, b, c]
    return None


def _search_4term(max_d):
    """Find 4 distinct denominators with 1/a+1/b+1/c+1/d=5/31, all <= max_d."""
    for a in range(7, min(max_d, 25) + 1):
        for b in range(a + 1, min(max_d, 80) + 1):
            for c in range(b + 1, min(max_d, 150) + 1):
                num = 31 * a * b * c
                denom = 5 * a * b * c - 31 * b * c - 31 * a * c - 31 * a * b
                if denom <= 0:
                    continue
                if num % denom == 0:
                    d = num // denom
                    if d > c and d <= max_d:
                        return [a, b, c, d]
    return None


def construct():
    deadline = time.time() + 85

    # Known solution
    known = [8, 31, 248]
    if _verify(known):
        best = known
        best_max = 248
    else:
        best = None
        best_max = MAX_DENOM + 1

    # Search for 3-term solutions with smaller max
    for max_d in range(27, 248):
        if time.time() > deadline:
            break
        sol = _search_3term(max_d)
        if sol and _verify(sol):
            best = sol
            best_max = max_d
            break

    if best_max < 248:
        # Search for 4-term solutions with even smaller max
        for max_d in range(27, best_max):
            if time.time() > deadline:
                break
            sol = _search_4term(max_d)
            if sol and _verify(sol):
                best = sol
                best_max = max_d
                break

    if best is not None:
        return best

    # Fallback: greedy Egyptian fraction decomposition
    r = TARGET
    result = []
    last_d = 0
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d <= last_d:
            d = last_d + 1
        result.append(d)
        r -= Fraction(1, d)
        last_d = d
    return result
