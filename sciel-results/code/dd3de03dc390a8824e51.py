"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
# EVOLVE-BLOCK-START
import math
import time

def _solve_one(n):
    """Find distinct x,y,z with 1/x+1/y+1/z=4/n, x<y<z, all <= 10^12."""
    x_lo = n // 4 + 1
    x_hi = (3 * n) // 4

    for x in range(x_lo, x_hi + 1):
        a = 4 * x - n
        b = n * x
        if a <= 0:
            continue
        g = math.gcd(a, b)
        a_r = a // g
        b_r = b // g
        b_r_sq = b_r * b_r

        # y = k, d = a_r*k - b_r, need d|b_r^2, z=(b_r+b_r^2//d)/a_r integer > k
        # d > 0 => k > b_r/a_r; z > k => d < b_r => k < 2*b_r/a_r
        k_lo = b_r // a_r + 1
        k_hi = (2 * b_r) // a_r

        for k in range(k_lo, k_hi + 1):
            d = a_r * k - b_r
            if d <= 0:
                continue
            if d >= b_r:
                break
            if b_r_sq % d != 0:
                continue
            num_z = b_r + b_r_sq // d
            if num_z % a_r != 0:
                continue
            z = num_z // a_r
            if z <= k or k <= x:
                continue
            if z > 10**12 or k > 10**12:
                continue
            if n * k * z + n * x * z + n * x * k == 4 * x * k * z:
                return (x, k, z)
    return None

def construct():
    deadline = time.time() + 85
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        sol = _solve_one(n)
        answers.append(list(sol) if sol is not None else None)
    return answers
# EVOLVE-BLOCK-END
