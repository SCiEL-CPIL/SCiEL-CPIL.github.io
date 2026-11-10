"""Egyptian fraction of 5/31 minimizing the largest denominator."""
import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85

    target = Fraction(5, 31)
    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    def check_time():
        return time.time() > deadline

    # Recursive search with tight bounds
    def search(rem, min_a, remaining, current):
        nonlocal best_max, best_result
        if check_time():
            return
        if remaining == 1:
            a_frac = 1 / rem
            if a_frac.denominator == 1:
                a = a_frac.numerator
                if a >= min_a and a < best_max and a <= 10000000:
                    denoms = current + [a]
                    update_best(denoms)
            return
        # a must satisfy: a >= min_a, a > 1/rem, a <= remaining/rem
        max_a_frac = remaining / rem
        max_a = max_a_frac.numerator // max_a_frac.denominator
        max_a = min(max_a, best_max - 1)
        if max_a < min_a:
            return
        for a in range(min_a, max_a + 1):
            if check_time():
                return
            if a >= best_max:
                break
            new_rem = rem - Fraction(1, a)
            if new_rem <= 0:
                continue
            search(new_rem, a + 1, remaining - 1, current + [a])

    # Try increasing number of terms
    for k in range(3, 8):
        if check_time():
            break
        # a1 range: a1 > 1/target = 31/5 = 6.2, so a1 >= 7
        # a1 <= k/target = 31k/5
        max_a1 = (31 * k) // 5
        for a in range(7, min(max_a1, best_max - 1) + 1):
            if check_time():
                break
            if a >= best_max:
                break
            rem = target - Fraction(1, a)
            if rem <= 0:
                continue
            search(rem, a + 1, k - 1, [a])

    if best_result is not None:
        return best_result
    return [7, 8, 9, 10000000]
