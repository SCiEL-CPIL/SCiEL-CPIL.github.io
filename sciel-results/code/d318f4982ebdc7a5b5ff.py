import time
from fractions import Fraction
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin within 90s

    # Strategy 1: Fast DP baseline to get a valid solution quickly
    def dp_baseline():
        target = 2520
        reached = {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total in list(reached.keys()):
                new_total = total + w
                if new_total <= target and new_total not in reached:
                    reached[new_total] = reached[total] + [d]
            if target in reached:
                sol = reached[target]
                if all(5 <= d <= 10000000 for d in sol) and len(set(sol)) == len(sol):
                    return sol
        return None

    # Strategy 2: Targeted search for small n using integer arithmetic
    def search_n4():
        """Find 4 distinct denominators >= 5 summing to 1."""
        # 1/a + 1/b + 1/c + 1/d = 1
        # => 1/c + 1/d = 1 - 1/a - 1/b = (a*b - a - b)/(a*b)
        # Let r = a*b - a - b, s = a*b
        # 1/c + 1/d = r/s => s*(c+d) = r*c*d => r*c*d - s*c - s*d = 0
        # (r*c - s)(r*d - s) = s^2
        # So we need to factor s^2 and find factors f1, f2 such that
        # f1 * f2 = s^2, c = (f1 + s)/r, d = (f2 + s)/r are integers > b
        for a in range(5, 20):
            if time.time() > deadline:
                return None
            for b in range(a + 1, 50):
                if time.time() > deadline:
                    return None
                r = a * b - a - b
                if r <= 0:
                    continue
                s = a * b
                # We need 1/c + 1/d = r/s where c > b
                # s*(c+d) = r*c*d
                # r*c*d - s*c - s*d = 0
                # (r*c - s)*(r*d - s) = s*s
                s2 = s * s
                # Factor s^2: find all divisors f of s^2
                # c = (f + s) / r, d = (s^2/f + s) / r
                # c must be integer, c > b, c >= 5
                # d must be integer, d > c, d >= 5
                # To avoid full factorization, iterate f over divisors of s^2
                # s^2 can be large, so let's be smarter:
                # c = (f + s) / r where f divides s^2
                # So (f + s) must be divisible by r
                # f ≡ -s (mod r) and f | s^2
                # Let's iterate over possible c values directly
                # 1/c < r/s => c > s/r = ab/(ab-a-b)
                # Also 1/c > (r/s)/2 (since 1/d < 1/c) => c < 2s/r
                c_min = max(b + 1, int(s // r) + 1)
                c_max = min(10000000, int(2 * s // r) + 2)
                if c_min > c_max:
                    continue
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        return None
                    # 1/d = r/s - 1/c = (r*c - s) / (s*c)
                    num = r * c - s
                    den = s * c
                    if num <= 0:
                        continue
                    if den % num != 0:
                        continue
                    d = den // num
                    if d <= c or d < 5:
                        continue
                    if d > 10000000:
                        continue
                    # Verify
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == 1:
                        return [a, b, c, d]
        return None

    def search_n5():
        """Find 5 distinct denominators >= 5 summing to 1."""
        # 1/a + 1/b + 1/c + 1/d + 1/e = 1
        # For given a, b, c: 1/d + 1/e = 1 - 1/a - 1/b - 1/c
        for a in range(5, 15):
            if time.time() > deadline:
                return None
            for b in range(a + 1, 25):
                if time.time() > deadline:
                    return None
                for c in range(b + 1, 40):
                    if time.time() > deadline:
                        return None
                    rem = 1 - Fraction(1, a) - Fraction(1, b) - Fraction(1, c)
                    if rem <= 0:
                        continue
                    # 1/d + 1/e = rem
                    # d and e > c, distinct
                    # 1/d < rem => d > 1/rem
                    # 1/d > rem/2 (since 1/e < 1/d) => d < 2/rem
                    d_min = max(c + 1, int(1 / rem) + 1)
                    d_max = min(10000000, int(2 / rem) + 2)
                    if d_min > d_max:
                        continue
                    for d in range(d_min, d_max + 1):
                        if time.time() > deadline:
                            return None
                        rem2 = rem - Fraction(1, d)
                        if rem2 <= 0:
                            continue
                        # 1/e = rem2 => e = 1/rem2
                        if rem2.denominator % rem2.numerator != 0:
                            continue
                        e = rem2.denominator // rem2.numerator
                        if e <= d or e < 5 or e > 10000000:
                            continue
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) == 1:
                            return [a, b, c, d, e]
        return None

    # Try smallest n first
    for n, search_fn in [(4, search_n4), (5, search_n5)]:
        result = search_fn()
        if result is not None:
            return result

    # Fallback to DP
    result = dp_baseline()
    if result is not None:
        return result

    # Last resort: hardcoded known valid solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 doesn't work as shown.
    # Let's use the DP result or a verified one.
    # Actually, let's compute a valid one using greedy with verification.
    # 1 = 1/5 + 4/5
    # 4/5 = 1/6 + 14/30 = 1/6 + 7/15
    # 7/15 = 1/7 + 4/105
    # 4/105 = 1/27 + 19/2835
    # This is getting long. Let's just return the DP solution.
    raise RuntimeError("no valid representation found")
