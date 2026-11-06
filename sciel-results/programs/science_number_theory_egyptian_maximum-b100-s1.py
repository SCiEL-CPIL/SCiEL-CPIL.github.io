"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    # Verify
    s = Fraction(1, 8) + Fraction(1, 31) + Fraction(1, 248)
    assert s == Fraction(5, 31), "Known solution invalid"
    
    target = Fraction(5, 31)
    
    # Try 3-term solutions: 1/a + 1/b + 1/c = 5/31 with a < b < c
    # a must satisfy 3/a > 5/31 → a < 93/5 = 18.6, so a <= 18
    # a must satisfy 1/a < 5/31 → a > 6.2, so a >= 7
    for a in range(7, best_max):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # Need 1/b + 1/c = rem_a with b > a, c > b
        # b must satisfy 2/b > rem_a → b < 2/rem_a
        # and 1/b < rem_a → b > 1/rem_a
        b_max = min(best_max - 1, int(2 / rem_a.numerator * rem_a.denominator) + 1)
        b_min = a + 1
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
                    best_max = c
                    best = [a, b, c]
    
    # Try 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # a: 4/a > 5/31 → a < 124/5 = 24.8, so a <= 24
    # a: 1/a < 5/31 → a >= 7
    for a in range(7, min(25, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # b: 3/b > rem_a → b < 3/rem_a
        b_max_4 = min(best_max - 1, int(3 / rem_a.numerator * rem_a.denominator) + 1)
        for b in range(a + 1, b_max_4):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            # c: 2/c > rem_b → c < 2/rem_b
            c_max_4 = min(best_max - 1, int(2 / rem_b.numerator * rem_b.denominator) + 1)
            for c in range(b + 1, c_max_4):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                d = rem_c.denominator // rem_c.numerator
                if d <= c or d >= best_max:
                    continue
                if rem_c == Fraction(1, d):
                    if d > c and d < best_max:
                        best_max = d
                        best = [a, b, c, d]
    
    # Try 5-term solutions: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
    # a: 5/a > 5/31 → a < 31, so a <= 30
    # a: 1/a < 5/31 → a >= 7
    for a in range(7, min(31, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # b: 4/b > rem_a → b < 4/rem_a
        b_max_5 = min(best_max - 1, int(4 / rem_a.numerator * rem_a.denominator) + 1)
        for b in range(a + 1, b_max_5):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            # c: 3/c > rem_b → c < 3/rem_b
            c_max_5 = min(best_max - 1, int(3 / rem_b.numerator * rem_b.denominator) + 1)
            for c in range(b + 1, c_max_5):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                # d: 2/d > rem_c → d < 2/rem_c
                d_max_5 = min(best_max - 1, int(2 / rem_c.numerator * rem_c.denominator) + 1)
                for d in range(c + 1, d_max_5):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        continue
                    e = rem_d.denominator // rem_d.numerator
                    if e <= d or e >= best_max:
                        continue
                    if rem_d == Fraction(1, e):
                        if e > d and e < best_max:
                            best_max = e
                            best = [a, b, c, d, e]
    
    return best
# EVOLVE-BLOCK-END
