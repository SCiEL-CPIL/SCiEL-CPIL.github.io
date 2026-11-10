"""
Model-and-Optimize approach for Egyptian fraction of 5/31:
Goal: Minimize the LARGEST denominator in a representation with 1..256 distinct terms.

Strategy:
1. Quick 3-term search for initial upper bound.
2. Exhaustive 4-term search with integer arithmetic.
3. 5-term search with integer arithmetic, time-bounded.
4. 6-term search if time permits, with tighter bounds.

All arithmetic uses integer (num, den) pairs for speed.
"""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Target: 5/31, stored as (num, den)
    T_num, T_den = 5, 31

    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    # --- 2-term check: 1/a + 1/b = 5/31, a < b ---
    # 1/a + 1/b = (a+b)/(ab) = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0
    # (5a - 31)(5b - 31) = 31^2 = 961
    # 961 = 1 * 961 = 31 * 31
    # 5a - 31 = 1 => a = 32/5 (not integer)
    # 5a - 31 = 31 => a = 62/5 (not integer)
    # 5a - 31 = 961 => a = 992/5 (not integer)
    # No 2-term solution.

    # --- 3-term search: 1/a + 1/b + 1/c = 5/31, a < b < c ---
    for a in range(7, 19):
        if time.time() > deadline:
            break
        # rem1 = 5/31 - 1/a = (5a-31)/(31a)
        r1_num = 5 * a - 31
        r1_den = 31 * a
        if r1_num <= 0:
            continue
        # 1/b + 1/c = r1_num/r1_den
        # b > a, 2/b >= r1_num/r1_den => b <= 2*r1_den/r1_num
        max_b = (2 * r1_den) // r1_num
        min_b = a + 1
        for b in range(min_b, max_b + 1):
            # rem2 = r1_num/r1_den - 1/b = (r1_num*b - r1_den)/(r1_den*b)
            r2_num = r1_num * b - r1_den
            r2_den = r1_den * b
            if r2_num <= 0:
                continue
            # 1/c = r2_num/r2_den => c = r2_den/r2_num
            if r2_den % r2_num != 0:
                continue
            c = r2_den // r2_num
            if c <= b:
                continue
            update_best([a, b, c])

    # --- 4-term search: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d < best_max ---
    for a in range(7, 19):
        if time.time() > deadline:
            break
        r1_num = 5 * a - 31
        r1_den = 31 * a
        if r1_num <= 0:
            continue

        max_b = (3 * r1_den) // r1_num
        min_b = a + 1

        for b in range(min_b, max_b + 1):
            r2_num = r1_num * b - r1_den
            r2_den = r1_den * b
            if r2_num <= 0:
                continue

            # c range: c > b, 2/c >= r2_num/r2_den => c <= 2*r2_den/r2_num
            max_c = min((2 * r2_den) // r2_num, best_max - 1)
            min_c = b + 1

            if max_c < min_c:
                continue

            for c in range(min_c, max_c + 1):
                r3_num = r2_num * c - r2_den
                r3_den = r2_den * c
                if r3_num <= 0:
                    continue
                # 1/d = r3_num/r3_den => d = r3_den/r3_num
                if r3_den % r3_num != 0:
                    continue
                d = r3_den // r3_num
                if d <= c:
                    continue
                if d >= best_max:
                    continue
                update_best([a, b, c, d])

    # --- 5-term search: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31 ---
    if best_max < float('inf'):
        for a in range(7, 19):
            if time.time() > deadline:
                break
            r1_num = 5 * a - 31
            r1_den = 31 * a
            if r1_num <= 0:
                continue

            max_b = (4 * r1_den) // r1_num
            min_b = a + 1

            for b in range(min_b, max_b + 1):
                if time.time() > deadline:
                    break
                r2_num = r1_num * b - r1_den
                r2_den = r1_den * b
                if r2_num <= 0:
                    continue

                max_c = min((3 * r2_den) // r2_num, best_max - 1)
                min_c = b + 1

                if max_c < min_c:
                    continue

                for c in range(min_c, max_c + 1):
                    if time.time() > deadline:
                        break
                    r3_num = r2_num * c - r2_den
                    r3_den = r2_den * c
                    if r3_num <= 0:
                        continue

                    max_d = min((2 * r3_den) // r3_num, best_max - 1)
                    min_d = c + 1

                    if max_d < min_d:
                        continue

                    for d in range(min_d, max_d + 1):
                        if time.time() > deadline:
                            break
                        r4_num = r3_num * d - r3_den
                        r4_den = r3_den * d
                        if r4_num <= 0:
                            continue
                        # 1/e = r4_num/r4_den => e = r4_den/r4_num
                        if r4_den % r4_num != 0:
                            continue
                        e = r4_den // r4_num
                        if e <= d:
                            continue
                        if e >= best_max:
                            continue
                        update_best([a, b, c, d, e])

    # --- 6-term search: 1/a + 1/b + 1/c + 1/d + 1/e + 1/f = 5/31 ---
    if best_max < float('inf') and time.time() < deadline - 10:
        for a in range(7, 19):
            if time.time() > deadline:
                break
            r1_num = 5 * a - 31
            r1_den = 31 * a
            if r1_num <= 0:
                continue

            max_b = (5 * r1_den) // r1_num
            min_b = a + 1

            for b in range(min_b, max_b + 1):
                if time.time() > deadline:
                    break
                r2_num = r1_num * b - r1_den
                r2_den = r1_den * b
                if r2_num <= 0:
                    continue

                max_c = min((4 * r2_den) // r2_num, best_max - 1)
                min_c = b + 1

                if max_c < min_c:
                    continue

                for c in range(min_c, max_c + 1):
                    if time.time() > deadline:
                        break
                    r3_num = r2_num * c - r2_den
                    r3_den = r2_den * c
                    if r3_num <= 0:
                        continue

                    max_d = min((3 * r3_den) // r3_num, best_max - 1)
                    min_d = c + 1

                    if max_d < min_d:
                        continue

                    for d in range(min_d, max_d + 1):
                        if time.time() > deadline:
                            break
                        r4_num = r3_num * d - r3_den
                        r4_den = r3_den * d
                        if r4_num <= 0:
                            continue

                        max_e = min((2 * r4_den) // r4_num, best_max - 1)
                        min_e = d + 1

                        if max_e < min_e:
                            continue

                        for e in range(min_e, max_e + 1):
                            if time.time() > deadline:
                                break
                            r5_num = r4_num * e - r4_den
                            r5_den = r4_den * e
                            if r5_num <= 0:
                                continue
                            # 1/f = r5_num/r5_den => f = r5_den/r5_num
                            if r5_den % r5_num != 0:
                                continue
                            f = r5_den // r5_num
                            if f <= e:
                                continue
                            if f >= best_max:
                                continue
                            update_best([a, b, c, d, e, f])

    if best_result is not None:
        return best_result

    # Final fallback
    return [7, 8, 9, 10000000]
