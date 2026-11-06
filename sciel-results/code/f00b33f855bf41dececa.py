"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    # Verify
    s = Fraction(1, 8) + Fraction(1, 31) + Fraction(1, 248)
    assert s == Fraction(5, 31), "Known solution invalid"
    
    target = Fraction(5, 31)
    
    def check_and_update(dens):
        nonlocal best, best_max
        mx = max(dens)
        if mx < best_max:
            best_max = mx
            best = list(dens)
    
    # Helper: given remaining fraction r = p/q and k terms left, all > prev,
    # find if r can be expressed as sum of k unit fractions with denominators > prev and < best_max
    # Returns list of denominators or None
    
    # Try 3-term solutions: 1/a + 1/b + 1/c = 5/31 with a < b < c
    for a in range(7, best_max):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # Need 1/b + 1/c = rem_a with b > a, c > b
        # b must satisfy 2/b > rem_a → b < 2/rem_a
        # and 1/b < rem_a → b > 1/rem_a
        b_lo = a + 1
        b_hi = min(best_max - 1, (2 * rem_a.denominator) // rem_a.numerator + 1)
        for b in range(b_lo, b_hi):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            # c = 1/rem_b must be integer > b
            if rem_b.numerator != 1:
                continue
            c = rem_b.denominator
            if c > b and c < best_max:
                check_and_update([a, b, c])
    
    # Try 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    for a in range(7, min(25, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_hi_4 = min(best_max - 1, (3 * rem_a.denominator) // rem_a.numerator + 1)
        for b in range(a + 1, b_hi_4):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c_hi_4 = min(best_max - 1, (2 * rem_b.denominator) // rem_b.numerator + 1)
            for c in range(b + 1, c_hi_4):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                if rem_c.numerator != 1:
                    continue
                d = rem_c.denominator
                if d > c and d < best_max:
                    check_and_update([a, b, c, d])
    
    # Try 5-term solutions: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
    for a in range(7, min(31, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_hi_5 = min(best_max - 1, (4 * rem_a.denominator) // rem_a.numerator + 1)
        for b in range(a + 1, b_hi_5):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c_hi_5 = min(best_max - 1, (3 * rem_b.denominator) // rem_b.numerator + 1)
            for c in range(b + 1, c_hi_5):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                d_hi_5 = min(best_max - 1, (2 * rem_c.denominator) // rem_c.numerator + 1)
                for d in range(c + 1, d_hi_5):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        continue
                    if rem_d.numerator != 1:
                        continue
                    e = rem_d.denominator
                    if e > d and e < best_max:
                        check_and_update([a, b, c, d, e])
    
    # Try 6-term solutions: 1/a + 1/b + 1/c + 1/d + 1/e + 1/f = 5/31
    for a in range(7, min(37, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_hi_6 = min(best_max - 1, (5 * rem_a.denominator) // rem_a.numerator + 1)
        for b in range(a + 1, b_hi_6):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c_hi_6 = min(best_max - 1, (4 * rem_b.denominator) // rem_b.numerator + 1)
            for c in range(b + 1, c_hi_6):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                d_hi_6 = min(best_max - 1, (3 * rem_c.denominator) // rem_c.numerator + 1)
                for d in range(c + 1, d_hi_6):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        continue
                    e_hi_6 = min(best_max - 1, (2 * rem_d.denominator) // rem_d.numerator + 1)
                    for e in range(d + 1, e_hi_6):
                        if time.time() > deadline:
                            break
                        rem_e = rem_d - Fraction(1, e)
                        if rem_e <= 0:
                            continue
                        if rem_e.numerator != 1:
                            continue
                        f = rem_e.denominator
                        if f > e and f < best_max:
                            check_and_update([a, b, c, d, e, f])
    
    return best
# EVOLVE-BLOCK-END
