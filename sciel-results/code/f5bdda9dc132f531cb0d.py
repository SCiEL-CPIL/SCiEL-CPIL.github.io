"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
LCM = 1
for d in DIVISORS:
    LCM = LCM * d // __import__('math').gcd(LCM, d)

def _can_fit(W):
    """Check if there exists L mod LCM such that every divisor has a multiple in [L, L+W]."""
    for r in range(LCM):
        ok = True
        for d in DIVISORS:
            # distance from L to next multiple of d above L
            dist = (d - r % d) % d
            if dist > W:
                ok = False
                break
        if ok:
            return True
    return False

def construct():
    lo, hi = 0, 1000000
    # Binary search for minimal W
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_fit(mid):
            hi = mid
        else:
            lo = mid + 1
    W = lo
    # Find actual L
    for r in range(LCM):
        ok = True
        for d in DIVISORS:
            dist = (d - r % d) % d
            if dist > W:
                ok = False
                break
        if ok:
            break
    # r is L mod LCM. Pick L = r (smallest positive)
    L = r
    # For each divisor, pick the smallest multiple in [L, L+W] that is distinct
    reps = []
    used = set()
    for d in DIVISORS:
        # smallest multiple of d >= L
        k = (L + d - 1) // d
        m = k * d
        while m in used:
            k += 1
            m = k * d
        reps.append(m)
        used.add(m)
    return reps
# EVOLVE-BLOCK-END
