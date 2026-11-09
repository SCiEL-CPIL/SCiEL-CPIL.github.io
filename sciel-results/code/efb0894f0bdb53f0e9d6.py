"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
# EVOLVE-BLOCK-START
import time
import math

def _factorize(n):
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return factors

def _divisors_from_factors(factors):
    divs = [1]
    for p, e in factors:
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def construct():
    deadline = time.time() + 85
    answers = []

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4

        for x in range(x_min, x_max + 1):
            if time.time() > deadline or found:
                break
            a = 4 * x - n
            b = n * x
            if a <= 0 or b <= 0:
                continue
            m_min = (b + 1 + a - 1) // a
            m_max = (2 * b - 1) // a
            if m_min > m_max:
                continue
            if m_min <= x:
                m_min = x + 1
            if m_min > m_max:
                continue

            m_range = m_max - m_min + 1
            b2 = b * b

            if m_range < 500:
                for m in range(m_min, m_max + 1):
                    if time.time() > deadline or found:
                        break
                    d = a * m - b
                    if d <= 0 or d >= b:
                        continue
                    if b2 % d != 0:
                        continue
                    d2 = b2 // d
                    num_y = d + b
                    if num_y % a != 0:
                        continue
                    y = num_y // a
                    if y <= x:
                        continue
                    num_z = d2 + b
                    if num_z % a != 0:
                        continue
                    z = num_z // a
                    if z <= y:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    if z > 10**12 or y > 10**12 or x > 10**12:
                        continue
                    lhs_num = n * y * z + n * x * z + n * x * y
                    rhs_num = 4 * x * y * z
                    if lhs_num == rhs_num:
                        found = [x, y, z]
                        break
            else:
                bfactors = _factorize(b)
                b2_factors = [(p, 2 * e) for p, e in bfactors]
                divs = _divisors_from_factors(b2_factors)
                for d in divs:
                    if time.time() > deadline or found:
                        break
                    if d <= 0 or d >= b:
                        continue
                    num_m = d + b
                    if num_m % a != 0:
                        continue
                    m = num_m // a
                    if m < m_min or m > m_max or m <= x:
                        continue
                    d2 = b2 // d
                    num_y = d + b
                    if num_y % a != 0:
                        continue
                    y = num_y // a
                    if y <= x:
                        continue
                    num_z = d2 + b
                    if num_z % a != 0:
                        continue
                    z = num_z // a
                    if z <= y:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    if z > 10**12 or y > 10**12 or x > 10**12:
                        continue
                    lhs_num = n * y * z + n * x * z + n * x * y
                    rhs_num = 4 * x * y * z
                    if lhs_num == rhs_num:
                        found = [x, y, z]
                        break
        answers.append(found)
    return answers
# EVOLVE-BLOCK-END
