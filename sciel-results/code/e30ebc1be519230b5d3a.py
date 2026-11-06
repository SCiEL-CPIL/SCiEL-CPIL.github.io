"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    # Verify
    s = Fraction(1, 8) + Fraction(1, 31) + Fraction(1, 248)
    assert s == target, "Known solution invalid"
    
    def is_unit_fraction(frac):
        """Check if frac is 1/k for some integer k."""
        if frac.numerator == 1:
            return frac.denominator
        return None

    def update_best(denoms):
        nonlocal best, best_max
        m = max(denoms)
        if m < best_max:
            # Verify distinctness and range
            if len(set(denoms)) == len(denoms) and all(1 <= d <= 10000000 for d in denoms):
                # Verify sum
                if sum(Fraction(1, d) for d in denoms) == target:
                    best_max = m
                    best = list(denoms)
                    return True
        return False

    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a < 3 * 31 / 5 = 18.6 => a <= 18
    # a > 1 / (5/31) = 6.2 => a >= 7
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # b > a
        # 1/b < rem_a => b > 1/rem_a
        # 2/b > rem_a => b < 2/rem_a
        b_min = a + 1
        b_max = int(2 / rem_a) + 1
        for b in range(b_min, min(b_max, best_max)):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c = is_unit_fraction(rem_b)
            if c is not None and c > b and c < best_max:
                if update_best([a, b, c]):
                    # If we found a very small one, we can potentially stop or tighten bounds
                    pass

    # Search for 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # a < 4 * 31 / 5 = 24.8 => a <= 24
    # a >= 7
    for a in range(7, 25):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = a + 1
        b_max = int(3 / rem_a) + 1
        for b in range(b_min, min(b_max, best_max)):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c_min = b + 1
            c_max = int(2 / rem_b) + 1
            for c in range(c_min, min(c_max, best_max)):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                d = is_unit_fraction(rem_c)
                if d is not None and d > c and d < best_max:
                    if update_best([a, b, c, d]):
                        pass

    # Search for 5-term solutions: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
    # a < 5 * 31 / 5 = 31 => a <= 30
    # a >= 7
    for a in range(7, 31):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = a + 1
        b_max = int(4 / rem_a) + 1
        for b in range(b_min, min(b_max, best_max)):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c_min = b + 1
            c_max = int(3 / rem_b) + 1
            for c in range(c_min, min(c_max, best_max)):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                d_min = c + 1
                d_max = int(2 / rem_c) + 1
                for d in range(d_min, min(d_max, best_max)):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        continue
                    e = is_unit_fraction(rem_d)
                    if e is not None and e > d and e < best_max:
                        if update_best([a, b, c, d, e]):
                            pass

    return best
# EVOLVE-BLOCK-END
