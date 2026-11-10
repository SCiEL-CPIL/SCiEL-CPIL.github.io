"""
Optimized search for Egyptian fraction of 5/31 with minimal largest denominator.
Uses systematic M-scan with float pre-filtering for speed.
"""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 87.0
    T_num, T_den = 5, 31
    target = 5.0 / 31.0

    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    # Helper: check if we can complete with k more terms, all > prev, all <= M
    # Returns True if feasible
    def can_complete(num, den, prev, k, M):
        """Check if num/den can be written as sum of k distinct unit fractions
        all > prev and <= M. Quick feasibility check."""
        if k <= 0:
            return num == 0
        if num <= 0:
            return False
        # Max possible sum: k/M
        # Min possible sum: sum of smallest k terms > prev: (prev+1) to (prev+k)
        min_sum = 0.0
        for i in range(1, k + 1):
            min_sum += 1.0 / (prev + i)
        max_sum = k * 1.0 / M
        val = float(num) / float(den)
        if val > max_sum + 1e-15:
            return False
        if val < min_sum - 1e-15:
            return False
        return True

    # Systematic search over M (max denominator), from small to large
    # For each M, try to find a valid decomposition
    # Start with M from 10 upward
    
    # First, quick 3-term search for initial bound
    for a in range(7, 20):
        if time.time() > deadline:
            break
        r1_num = 5 * a - 31
        r1_den = 31 * a
        if r1_num <= 0:
            continue
        max_b = (2 * r1_den) // r1_num
        min_b = a + 1
        for b in range(min_b, max_b + 1):
            r2_num = r1_num * b - r1_den
            r2_den = r1_den * b
            if r2_num <= 0:
                continue
            if r2_den % r2_num != 0:
                continue
            c = r2_den // r2_num
            if c <= b:
                continue
            update_best([a, b, c])

    if best_max == float('inf'):
        # Fallback greedy
        return [7, 8, 9, 10000000]

    # Now search for better solutions with bounded M
    # Try 4-term, 5-term, 6-term with M < best_max
    # Use float pre-filtering for speed

    # 4-term search with M bound
    M_bound = best_max - 1
    for a in range(7, 20):
        if time.time() > deadline:
            break
        r1_num = 5 * a - 31
        r1_den = 31 * a
        if r1_num <= 0:
            continue
        max_b = min((3 * r1_den) // r1_num, M_bound)
        min_b = a + 1
        for b in range(min_b, max_b + 1):
            r2_num = r1_num * b - r1_den
            r2_den = r1_den * b
            if r2_num <= 0:
                continue
            max_c = min((2 * r2_den) // r2_num, M_bound)
            min_c = b + 1
            if max_c < min_c:
                continue
            for c in range(min_c, max_c + 1):
                r3_num = r2_num * c - r2_den
                r3_den = r2_den * c
                if r3_num <= 0:
                    continue
                if r3_den % r3_num != 0:
                    continue
                d = r3_den // r3_num
                if d <= c or d > M_bound:
                    continue
                update_best([a, b, c, d])

    if time.time() > deadline - 5:
        return best_result if best_result else [7, 8, 9, 10000000]

    # 5-term search
    M_bound = best_max - 1
    for a in range(7, 20):
        if time.time() > deadline:
            break
        r1_num = 5 * a - 31
        r1_den = 31 * a
        if r1_num <= 0:
            continue
        max_b = min((4 * r1_den) // r1_num, M_bound)
        min_b = a + 1
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            r2_num = r1_num * b - r1_den
            r2_den = r1_den * b
            if r2_num <= 0:
                continue
            max_c = min((3 * r2_den) // r2_num, M_bound)
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
                max_d = min((2 * r3_den) // r3_num, M_bound)
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
                    if r4_den % r4_num != 0:
                        continue
                    e = r4_den // r4_num
                    if e <= d or e > M_bound:
                        continue
                    update_best([a, b, c, d, e])

    if time.time() > deadline - 5:
        return best_result if best_result else [7, 8, 9, 10000000]

    # 6-term search
    M_bound = best_max - 1
    for a in range(7, 20):
        if time.time() > deadline:
            break
        r1_num = 5 * a - 31
        r1_den = 31 * a
        if r1_num <= 0:
            continue
        max_b = min((5 * r1_den) // r1_num, M_bound)
        min_b = a + 1
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            r2_num = r1_num * b - r1_den
            r2_den = r1_den * b
            if r2_num <= 0:
                continue
            max_c = min((4 * r2_den) // r2_num, M_bound)
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
                max_d = min((3 * r3_den) // r3_num, M_bound)
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
                    max_e = min((2 * r4_den) // r4_num, M_bound)
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
                        if r5_den % r5_num != 0:
                            continue
                        f = r5_den // r5_num
                        if f <= e or f > M_bound:
                            continue
                        update_best([a, b, c, d, e, f])

    if best_result is not None:
        return best_result
    return [7, 8, 9, 10000000]
