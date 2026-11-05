import time
from math import factorial
from itertools import product

_TARGET = factorial(12)  # 479001600
_MIN = 13
_MAX_FACTORS = 12

# Prime factorization: 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
_PRIMES = [2, 3, 5, 7, 11]
_EXP_TARGET = (10, 5, 2, 1, 1)

def _factor(n):
    """Return exponent vector of n over _PRIMES."""
    exps = [0, 0, 0, 0, 0]
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            exps[i] += 1
            n //= p
    return tuple(exps)

# Precompute all divisors of 12! that are > 12, sorted
def _all_divisors():
    divs = []
    # Generate from prime exponent ranges
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        val = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if val > _MIN:
                            divs.append((val, (e2, e3, e5, e7, e11)))
    divs.sort()
    return divs

_DIVS = _all_divisors()  # list of (value, exp_vector)
_DIVS_VAL = [d[0] for d in _DIVS]
_DIVS_EXP = [d[1] for d in _DIVS]
_N_DIVS = len(_DIVS)

def _search(max_factor, deadline):
    """DFS: find distinct factors in [_MIN, max_factor] whose product is _TARGET.
    Returns list of factors or None."""
    # Filter divisors up to max_factor
    idx = 0
    while idx < _N_DIVS and _DIVS_VAL[idx] <= max_factor:
        idx += 1
    n_divs = idx
    if n_divs == 0:
        return None

    chosen = []
    remaining = list(_EXP_TARGET)

    def dfs(start, used_count):
        if time.time() > deadline:
            return None
        if all(r == 0 for r in remaining):
            return list(chosen)
        if used_count >= _MAX_FACTORS:
            return None
        for i in range(start, n_divs):
            val = _DIVS_VAL[i]
            exp = _DIVS_EXP[i]
            # Check if we can subtract exp from remaining
            ok = True
            for j in range(5):
                if remaining[j] < exp[j]:
                    ok = False
                    break
            if not ok:
                continue
            # Check if this divisor can be formed from remaining (already checked)
            # Also: val must divide remaining product (checked by exp comparison)
            chosen.append(val)
            for j in range(5):
                remaining[j] -= exp[j]
            result = dfs(i + 1, used_count + 1)
            if result is not None:
                return result
            for j in range(5):
                remaining[j] += exp[j]
            chosen.pop()
        return None

    return dfs(0, 0)

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin under 90s

    # Binary search on M: smallest M such that a valid decomposition exists
    # Lower bound: at least _MIN. Upper bound: _TARGET (single factor).
    # But we can tighten: with k factors all >= 13, product >= 13^k.
    # 13^12 is huge, so max factors is limited.
    # Actually the constraint is product = 12! = 479001600.
    # Minimum product for k factors > 12 is 13*14*...*12+k.
    # For k=1: min product = 13, but we need exactly 479001600.
    # For k=1: only factor is 479001600 itself.
    # For k=2: need a*b = 479001600, a,b > 12, distinct.
    # ...
    # Let's just binary search.

    lo = _MIN
    hi = _TARGET

    # First check if hi is feasible (single factor _TARGET)
    # _TARGET > 12, so yes.

    best_M = hi
    # Binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        result = _search(mid, deadline)
        if result is not None:
            best_M = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if time.time() >= deadline:
        # Fallback: return single factor
        return [_TARGET]

    # Now find the actual decomposition with max factor <= best_M
    decomp = _search(best_M, deadline)
    if decomp is None:
        return [_TARGET]

    # Verify
    prod = 1
    for x in decomp:
        prod *= x
    if prod != _TARGET:
        return [_TARGET]
    if len(decomp) != len(set(decomp)):
        return [_TARGET]
    if not all(x > 12 for x in decomp):
        return [_TARGET]
    if not (1 <= len(decomp) <= 12):
        return [_TARGET]

    return decomp
