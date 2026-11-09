import time
from fractions import Fraction

def construct():
    start_time = time.time()
    target = Fraction(5, 31)
    deadline = start_time + 85.0

    best_solution = None
    best_max = float('inf')

    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31, a < b < c
    # a >= 7 (since 1/6 > 5/31)
    # For each a, b, c is determined by c = 1 / (5/31 - 1/a - 1/b)
    # c must be integer and c > b

    for a in range(7, 200):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            break

        # b > a, and 1/b < rem_a => b > 1/rem_a
        # Also, c = 1/(rem_a - 1/b) > b => rem_a - 1/b < 1/b => rem_a < 2/b => b < 2/rem_a
        if rem_a.numerator > 0:
            b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator)
            b_max = (2 * rem_a.denominator - 1) // rem_a.numerator
        else:
            continue

        if b_max < b_min:
            continue

        # Limit b range to keep search tractable
        b_max = min(b_max, b_min + 1000)

        for b in range(b_min, b_max + 1):
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            if rem_b.numerator == 0:
                continue

            # c = 1/rem_b = rem_b.denominator / rem_b.numerator
            c_num = rem_b.denominator
            c_den = rem_b.numerator

            if c_num % c_den != 0:
                continue

            c = c_num // c_den

            if c <= b:
                continue
            if c > 10000000:
                continue

            max_den = c
            if max_den < best_max:
                best_max = max_den
                best_solution = [a, b, c]

    if best_solution is not None:
        return best_solution

    # Fallback
    return [8, 31, 248]
