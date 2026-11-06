"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    # Verify
    s = Fraction(1, 8) + Fraction(1, 31) + Fraction(1, 248)
    assert s == Fraction(5, 31), "Known solution invalid"
    
    # Search for solutions with max denominator < best_max
    # Try all combinations of 3 denominators in [2, best_max-1]
    # 1/a + 1/b + 1/c = 5/31 with a < b < c < best_max
    target = Fraction(5, 31)
    
    # For 3 terms: 1/a + 1/b + 1/c = 5/31
    # a must satisfy 1/a < 5/31, so a > 31/5 = 6.2, so a >= 7
    # Also 1/a + 1/(a+1) + 1/(a+2) > 5/31 for small a
    # Upper bound on a: 3/a > 5/31 → a < 93/5 = 18.6, so a <= 18
    # But we could have more terms too.
    
    # Try 3-term solutions
    for a in range(7, best_max):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # Need 1/b + 1/c = rem_a with b > a, c > b, c < best_max
        # b must satisfy 2/b > rem_a → b < 2/rem_a
        # and 1/b < rem_a → b > 1/rem_a
        b_max = min(best_max - 1, int(2 / rem_a.numerator * rem_a.denominator) + 1)
        b_min = a + 1
        # Also 1/b + 1/(b+1) >= rem_a for feasibility
        if b_min < 2:
            b_min = 2
        for b in range(b_min, b_max):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c = rem_b.denominator // rem_b.numerator
            if c <= b or c >= best_max:
                continue
            if rem_b == Fraction(1, c):
                if c > b and c < best_max:
                    cand = [a, b, c]
                    if c < best_max:
                        best_max = c
                        best = cand
    
    # Try 4-term solutions with max < best_max
    if best_max > 200:  # only if we haven't found something very good
        for a in range(7, min(15, best_max)):
            if time.time() > deadline:
                break
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            for b in range(a + 1, min(a + 20, best_max)):
                if time.time() > deadline:
                    break
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    continue
                for c in range(b + 1, min(b + 20, best_max)):
                    if time.time() > deadline:
                        break
                    rem_c = rem_b - Fraction(1, c)
                    if rem_c <= 0:
                        continue
                    # Need 1/d = rem_c
                    d = rem_c.denominator // rem_c.numerator
                    if d <= c or d >= best_max:
                        continue
                    if rem_c == Fraction(1, d):
                        if d > c and d < best_max:
                            best_max = d
                            best = [a, b, c, d]
    
    return best
# EVOLVE-BLOCK-END
