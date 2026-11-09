"""Combined approach: direct algebraic solving with systematic multi-term search."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    target = Fraction(5, 31)

    # Known solution: [8, 31, 248] with max 248
    best_solution = [8, 31, 248]
    best_max = 248

    # Verify known solution
    assert Fraction(1,8) + Fraction(1,31) + Fraction(1,248) == target

    def check_time():
        return time.time() > deadline

    # Direct algebraic: solve 1/b + 1/c = r (where r = p/q in lowest terms)
    # Returns list of (b, c) pairs with b < c, both positive integers
    def solve_2term(r):
        """Solve 1/b + 1/c = r for positive integers b < c.
        Formula: (p*b - q)(p*c - q) = q^2 where r = p/q
        """
        p = r.numerator
        q = r.denominator
        if p <= 0 or q <= 0:
            return []
        q_sq = q * q
        results = []
        # Find all divisors d of q_sq where d < q_sq/d (to ensure b < c)
        # and (d + q) % p == 0
        for d in range(1, int(q_sq**0.5) + 1):
            if q_sq % d == 0:
                d2 = q_sq // d
                # Case 1: d as smaller factor
                if (d + q) % p == 0:
                    b = (d + q) // p
                    c_num = d2 + q
                    if c_num % p == 0:
                        c = c_num // p
                        if b > 0 and c > 0 and b < c:
                            results.append((b, c))
                # Case 2: d2 as smaller factor (if d != d2)
                if d != d2 and (d2 + q) % p == 0:
                    b = (d2 + q) // p
                    c_num = d + q
                    if c_num % p == 0:
                        c = c_num // p
                        if b > 0 and c > 0 and b < c:
                            results.append((b, c))
        return results

    # Search for 3-term solutions: 1/a + 1/b + 1/c = target
    # For each a, solve 1/b + 1/c = target - 1/a
    def find_3term(max_c):
        """Find 3-term solution with c < max_c. Returns (max_den, [a,b,c]) or None."""
        best = None
        # a must satisfy: 1/a < target, so a > 31/5 = 6.2, a >= 7
        # Also 3/a >= target (since a<=b<=c), so a <= 3*31/5 = 18.6, a <= 18
        for a in range(7, 19):
            if check_time():
                break
            r = target - Fraction(1, a)
            if r <= 0:
                continue
            pairs = solve_2term(r)
            for b, c in pairs:
                if b <= a or c <= b:
                    continue
                if c >= max_c:
                    continue
                # All distinct
                if a == b or b == c or a == c:
                    continue
                max_den = c
                if best is None or max_den < best[0]:
                    best = (max_den, [a, b, c])
        return best

    # Search for 4-term solutions: 1/a + 1/b + 1/c + 1/d = target
    # For each (a,b), solve 1/c + 1/d = target - 1/a - 1/b
    def find_4term(max_d):
        """Find 4-term solution with d < max_d. Returns (max_den, [a,b,c,d]) or None."""
        best = None
        # a >= 7, a <= 18 (from 4/a >= target)
        for a in range(7, 19):
            if check_time():
                break
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            # b > a, and 3/b >= r1 (since b<=c<=d), so b <= 3/r1
            # Also 1/b < r1, so b > 1/r1
            b_min = a + 1
            b_max = min(max_d - 1, int(3 / r1) if r1 > 0 else max_d)
            for b in range(b_min, b_max + 1):
                if check_time():
                    break
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    break
                pairs = solve_2term(r2)
                for c, d in pairs:
                    if c <= b or d <= c:
                        continue
                    if d >= max_d:
                        continue
                    if a == b or b == c or c == d or a == c or a == d or b == d:
                        continue
                    max_den = d
                    if best is None or max_den < best[0]:
                        best = (max_den, [a, b, c, d])
        return best

    # Search for 5-term solutions
    def find_5term(max_e):
        """Find 5-term solution with e < max_e. Returns (max_den, [a,b,c,d,e]) or None."""
        best = None
        for a in range(7, 19):
            if check_time():
                break
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            b_min = a + 1
            b_max = min(max_e - 1, int(4 / r1) if r1 > 0 else max_e)
            for b in range(b_min, b_max + 1):
                if check_time():
                    break
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    break
                c_min = b + 1
                c_max = min(max_e - 1, int(3 / r2) if r2 > 0 else max_e)
                for c in range(c_min, c_max + 1):
                    if check_time():
                        break
                    r3 = r2 - Fraction(1, c)
                    if r3 <= 0:
                        break
                    pairs = solve_2term(r3)
                    for d, e in pairs:
                        if d <= c or e <= d:
                            continue
                        if e >= max_e:
                            continue
                        if len({a,b,c,d,e}) != 5:
                            continue
                        max_den = e
                        if best is None or max_den < best[0]:
                            best = (max_den, [a, b, c, d, e])
        return best

    # Systematic search: try to find solutions with max < 248
    # Check 3-term solutions with max < 248
    result_3 = find_3term(248)
    if result_3 is not None:
        if result_3[0] < best_max:
            best_max = result_3[0]
            best_solution = result_3[1]

    # Check 4-term solutions with max < best_max
    if best_max > 7:
        result_4 = find_4term(best_max)
        if result_4 is not None:
            if result_4[0] < best_max:
                best_max = result_4[0]
                best_solution = result_4[1]

    # Check 5-term solutions with max < best_max
    if best_max > 7:
        result_5 = find_5term(best_max)
        if result_5 is not None:
            if result_5[0] < best_max:
                best_max = result_5[0]
                best_solution = result_5[1]

    # If we still have time, try to find even better solutions
    # by searching with progressively smaller limits
    if not check_time() and best_max > 10:
        # Try to find 3-term with max < best_max
        for limit in range(best_max - 1, 6, -1):
            if check_time():
                break
            result_3 = find_3term(limit)
            if result_3 is not None:
                if result_3[0] < best_max:
                    best_max = result_3[0]
                    best_solution = result_3[1]
                break  # Found the smallest max for 3-term

    if not check_time() and best_max > 10:
        # Try to find 4-term with max < best_max
        for limit in range(best_max - 1, 6, -1):
            if check_time():
                break
            result_4 = find_4term(limit)
            if result_4 is not None:
                if result_4[0] < best_max:
                    best_max = result_4[0]
                    best_solution = result_4[1]
                break

    if not check_time() and best_max > 10:
        # Try to find 5-term with max < best_max
        for limit in range(best_max - 1, 6, -1):
            if check_time():
                break
            result_5 = find_5term(limit)
            if result_5 is not None:
                if result_5[0] < best_max:
                    best_max = result_5[0]
                    best_solution = result_5[1]
                break

    # Final verification
    s = sum(Fraction(1, d) for d in best_solution)
    assert s == target, f"Solution invalid: {s} != {target}"
    assert len(set(best_solution)) == len(best_solution), "Denominators not distinct"
    assert all(1 <= d <= 10000000 for d in best_solution), "Denominator out of range"
    assert 1 <= len(best_solution) <= 256, "Wrong number of terms"

    return best_solution

# EVOLVE-BLOCK-END
