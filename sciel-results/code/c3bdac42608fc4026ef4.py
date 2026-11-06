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
    assert s == Fraction(5, 31), "Known solution invalid"
    
    def check_and_update(denoms):
        nonlocal best, best_max
        m = max(denoms)
        if m < best_max:
            # Verify
            total = sum(Fraction(1, d) for d in denoms)
            if total == target and len(set(denoms)) == len(denoms) and all(1 <= d <= 10000000 for d in denoms):
                best_max = m
                best = list(denoms)
    
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
        b_max = min(best_max - 1, int(2 / rem_a.numerator * rem_a.denominator) + 1)
        b_min = a + 1
        for b in range(b_min, b_max):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            if rem_b.numerator == 1:
                c = rem_b.denominator
                if c > b and c < best_max:
                    check_and_update([a, b, c])
            else:
                # Check if rem_b is a unit fraction
                if rem_b.denominator % rem_b.numerator == 0:
                    c = rem_b.denominator // rem_b.numerator
                    if c > b and c < best_max and rem_b == Fraction(1, c):
                        check_and_update([a, b, c])
    
    # Try 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    for a in range(7, min(30, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_max_4 = min(best_max - 1, int(3 / rem_a.numerator * rem_a.denominator) + 1)
        for b in range(a + 1, b_max_4):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c_max_4 = min(best_max - 1, int(2 / rem_b.numerator * rem_b.denominator) + 1)
            for c in range(b + 1, c_max_4):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                if rem_c.numerator == 1:
                    d = rem_c.denominator
                    if d > c and d < best_max:
                        check_and_update([a, b, c, d])
                elif rem_c.denominator % rem_c.numerator == 0:
                    d = rem_c.denominator // rem_c.numerator
                    if d > c and d < best_max and rem_c == Fraction(1, d):
                        check_and_update([a, b, c, d])
    
    # Try 5-term solutions
    for a in range(7, min(31, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_max_5 = min(best_max - 1, int(4 / rem_a.numerator * rem_a.denominator) + 1)
        for b in range(a + 1, b_max_5):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c_max_5 = min(best_max - 1, int(3 / rem_b.numerator * rem_b.denominator) + 1)
            for c in range(b + 1, c_max_5):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                d_max_5 = min(best_max - 1, int(2 / rem_c.numerator * rem_c.denominator) + 1)
                for d in range(c + 1, d_max_5):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        continue
                    if rem_d.numerator == 1:
                        e = rem_d.denominator
                        if e > d and e < best_max:
                            check_and_update([a, b, c, d, e])
                    elif rem_d.denominator % rem_d.numerator == 0:
                        e = rem_d.denominator // rem_d.numerator
                        if e > d and e < best_max and rem_d == Fraction(1, e):
                            check_and_update([a, b, c, d, e])
    
    # Try 6-term solutions with tighter bounds
    for a in range(7, min(31, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_max_6 = min(best_max - 1, int(5 / rem_a.numerator * rem_a.denominator) + 1)
        for b in range(a + 1, b_max_6):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c_max_6 = min(best_max - 1, int(4 / rem_b.numerator * rem_b.denominator) + 1)
            for c in range(b + 1, c_max_6):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                d_max_6 = min(best_max - 1, int(3 / rem_c.numerator * rem_c.denominator) + 1)
                for d in range(c + 1, d_max_6):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        continue
                    e_max_6 = min(best_max - 1, int(2 / rem_d.numerator * rem_d.denominator) + 1)
                    for e in range(d + 1, e_max_6):
                        if time.time() > deadline:
                            break
                        rem_e = rem_d - Fraction(1, e)
                        if rem_e <= 0:
                            continue
                        if rem_e.numerator == 1:
                            f = rem_e.denominator
                            if f > e and f < best_max:
                                check_and_update([a, b, c, d, e, f])
                        elif rem_e.denominator % rem_e.numerator == 0:
                            f = rem_e.denominator // rem_e.numerator
                            if f > e and f < best_max and rem_e == Fraction(1, f):
                                check_and_update([a, b, c, d, e, f])
    
    return best
# EVOLVE-BLOCK-END
