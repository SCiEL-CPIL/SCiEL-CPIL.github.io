"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import math

# EVOLVE-BLOCK-START

def _search(k, target, min_den, max_den, deadline):
    if time.time() > deadline:
        return None
    if k == 0:
        return [] if target == 0 else None
    if k == 1:
        if target.denominator == 1:
            d = target.numerator
            if min_den <= d <= max_den:
                return [d]
        return None
    if k == 2:
        p, q = target.numerator, target.denominator
        if p == 0:
            return None
        q2 = q * q
        i = 1
        while i * i <= q2:
            if q2 % i == 0:
                f1, f2 = i, q2 // i
                for fa, fb in [(f1, f2), (f2, f1)]:
                    na, nb = fa + q, fb + q
                    if na % p == 0 and nb % p == 0:
                        a, b = na // p, nb // p
                        if min_den <= a < b <= max_den:
                            return [a, b]
                        if min_den <= b < a <= max_den:
                            return [b, a]
            i += 1
        return None
    if target <= 0:
        return None

    # Tighter bounds: smallest possible sum with k terms starting from min_den
    # Largest possible sum with k terms starting from min_den
    min_possible_sum = sum(Fraction(1, min_den + i) for i in range(k))
    if target > min_possible_sum:
        return None
    
    # a must be at least ceil(1/target) but also at least min_den
    # and at most such that 1/a + 1/(a+1) + ... + 1/(a+k-1) >= target
    a_start = max(min_den, -(-1 // target.numerator) if target.denominator == 1 else int(1 / float(target)) + 1)
    
    # Upper bound: 1/a + (k-1)/(a+1) >= target => a <= (k-1)/target + 1 roughly
    a_end = min(max_den, int(k / float(target)) + 2)
    if a_end < a_start:
        return None

    for a in range(a_start, a_end + 1):
        if time.time() > deadline:
            return None
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        # Check if remaining k-1 terms can sum to rem
        # Max sum of k-1 terms starting from a+1
        max_rem_sum = sum(Fraction(1, a + 1 + i) for i in range(k - 1))
        if rem > max_rem_sum:
            continue
        # Min sum of k-1 terms starting from a+1 (using largest denominators)
        # This is hard to bound tightly, skip for now
        result = _search(k - 1, rem, a + 1, max_den, deadline)
        if result is not None:
            return [a] + result
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 80  # Leave 10s margin before 90s kill

    # The minimum possible number of terms is 8 (since 7 terms max sum < 1)
    # Try k=8 first (most constrained, fastest to search or prove impossible)
    # Then k=9, then fall back to known 10-term solution
    
    for k in [8, 9]:
        if time.time() > deadline - 2:
            break
        sol = _search(k, Fraction(1, 1), 5, 10000000, deadline)
        if sol is not None:
            return sol

    # If no shorter solution found, return the known 10-term solution
    return [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
# EVOLVE-BLOCK-END
