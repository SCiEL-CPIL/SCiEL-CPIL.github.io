"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def _compute_lcm(lst):
    r = 1
    for x in lst:
        r = _lcm(r, x)
    return r

_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = _compute_lcm(_DIVS)  # 30030
_N = len(_DIVS)
_MAX_VAL = 1000000

def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # 10s safety margin before 90s kill

    # Precompute: for each divisor d, the residue classes mod LCM that are multiples of d.
    # Actually simpler: for a window [s, s+W], d has a multiple in it iff
    # there exists integer k such that s <= k*d <= s+W, i.e.
    # ceil(s/d) <= floor((s+W)/d)
    # The actual value is k*d for k in [ceil(s/d), floor((s+W)/d)].

    # Strategy: iterate W from 1 upward. For each W, slide start s over one LCM period.
    # For efficiency, note that as s changes by 1, the set of valid k for each d changes
    # only when s crosses a multiple boundary. But for small W, just checking each s is fine.

    # Optimization: for a given W and start s, the multiple of d in [s, s+W] (if it exists)
    # is uniquely determined when W < d (which it will be for our small W).
    # The multiple is: m_d = d * ceil(s / d), and it's valid iff m_d <= s + W.

    best_result = None
    best_W = None

    for W in range(1, 200):
        if time.time() > deadline:
            break

        found = False
        # Slide over one full LCM period
        for s in range(_LCM):
            if time.time() > deadline:
                break

            vals = []
            valid = True
            for d in _DIVS:
                # Find the smallest multiple of d >= s
                if s % d == 0:
                    m = s
                else:
                    m = s + (d - s % d)
                if m > s + W:
                    valid = False
                    break
                vals.append(m)

            if not valid:
                continue

            # Check distinctness
            if len(set(vals)) != _N:
                continue

            # Check all <= MAX_VAL
            # Since s < LCM and W < 200, max val < LCM + 200 < 1000000, so this is fine.
            # But let's be safe:
            if max(vals) > _MAX_VAL:
                continue

            # Valid solution found
            found = True
            best_W = W
            best_result = vals
            break

        if found:
            break

    if best_result is not None:
        # Ensure we return Python ints in a list
        return [int(x) for x in best_result]

    # Fallback: return the divisors themselves (valid but not optimal)
    return [int(d) for d in _DIVS]
# EVOLVE-BLOCK-END
