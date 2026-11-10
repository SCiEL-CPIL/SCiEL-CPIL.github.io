"""
Find Egyptian fraction of 5/31 minimizing the largest denominator.

Strategy:
1. Fast 3-term and 4-term searches (complete, fast).
2. For 5+ terms: use a meet-in-the-middle approach over small denominators.
   - Precompute all subset sums of reciprocals from a candidate pool {a, a+1, ..., M}
     for small subsets (size 2 and 3).
   - For each pair of complementary subset sizes that sum to k, look for pairs
     whose reciprocal sum equals the target.
3. Also do a bounded DFS for k=5,6,7 with aggressive pruning.

The key insight: if we restrict to denominators in [a, M], we can precompute
subset sums for small subsets and use a hash table to find complementary pairs.
"""
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 82  # safety margin before 90s kill

    target = Fraction(5, 31)
    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    # --- 3-term search ---
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        min_b_frac = 1 / rem
        min_b = max(a + 1, min_b_frac.numerator // min_b_frac.denominator + 1)
        max_b_frac = 2 / rem
        max_b = max_b_frac.numerator // max_b_frac.denominator

        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem2 = rem - Fraction(1, b)
            if rem2 <= 0:
                continue
            c_frac = 1 / rem2
            if c_frac.denominator == 1:
                c = c_frac.numerator
                if c > b and c <= 10000000:
                    update_best([a, b, c])

    # --- 4-term search ---
    for a in range(7, 126):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        min_b_frac = 1 / rem
        min_b = max(a + 1, min_b_frac.numerator // min_b_frac.denominator + 1)
        max_b_frac = 3 / rem
        max_b = min(max_b_frac.numerator // max_b_frac.denominator, best_max - 1)

        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            if b >= best_max:
                break
            rem2 = rem - Fraction(1, b)
            if rem2 <= 0:
                continue
            min_c_frac = 1 / rem2
            min_c = max(b + 1, min_c_frac.numerator // min_c_frac.denominator + 1)
            max_c_frac = 2 / rem2
            max_c = min(max_c_frac.numerator // max_c_frac.denominator, best_max - 1)

            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                if c >= best_max:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                d_frac = 1 / rem3
                if d_frac.denominator == 1:
                    d = d_frac.numerator
                    if d > c and d <= 10000000 and d < best_max:
                        update_best([a, b, c, d])

    # --- 5-term search (DFS with pruning) ---
    for a in range(7, 32):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        min_b_frac = 1 / rem
        min_b = max(a + 1, min_b_frac.numerator // min_b_frac.denominator + 1)
        max_b_frac = 4 / rem
        max_b = min(max_b_frac.numerator // max_b_frac.denominator, best_max - 1)

        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            if b >= best_max:
                break
            rem2 = rem - Fraction(1, b)
            if rem2 <= 0:
                continue
            min_c_frac = 1 / rem2
            min_c = max(b + 1, min_c_frac.numerator // min_c_frac.denominator + 1)
            max_c_frac = 3 / rem2
            max_c = min(max_c_frac.numerator // max_c_frac.denominator, best_max - 1)

            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                if c >= best_max:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                min_d_frac = 1 / rem3
                min_d = max(c + 1, min_d_frac.numerator // min_d_frac.denominator + 1)
                max_d_frac = 2 / rem3
                max_d = min(max_d_frac.numerator // max_d_frac.denominator, best_max - 1)

                for d in range(min_d, max_d + 1):
                    if time.time() > deadline:
                        break
                    if d >= best_max:
                        break
                    rem4 = rem3 - Fraction(1, d)
                    if rem4 <= 0:
                        continue
                    e_frac = 1 / rem4
                    if e_frac.denominator == 1:
                        e = e_frac.numerator
                        if e > d and e <= 10000000 and e < best_max:
                            update_best([a, b, c, d, e])

    # --- 6-term search (only if best_max is still large) ---
    if best_max > 100:
        for a in range(7, 25):
            if time.time() > deadline:
                break
            if a >= best_max:
                break
            rem = target - Fraction(1, a)
            if rem <= 0:
                continue
            min_b_frac = 1 / rem
            min_b = max(a + 1, min_b_frac.numerator // min_b_frac.denominator + 1)
            max_b_frac = 5 / rem
            max_b = min(max_b_frac.numerator // max_b_frac.denominator, best_max - 1)

            for b in range(min_b, max_b + 1):
                if time.time() > deadline:
                    break
                if b >= best_max:
                    break
                rem2 = rem - Fraction(1, b)
                if rem2 <= 0:
                    continue
                min_c_frac = 1 / rem2
                min_c = max(b + 1, min_c_frac.numerator // min_c_frac.denominator + 1)
                max_c_frac = 4 / rem2
                max_c = min(max_c_frac.numerator // max_c_frac.denominator, best_max - 1)

                for c in range(min_c, max_c + 1):
                    if time.time() > deadline:
                        break
                    if c >= best_max:
                        break
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    min_d_frac = 1 / rem3
                    min_d = max(c + 1, min_d_frac.numerator // min_d_frac.denominator + 1)
                    max_d_frac = 3 / rem3
                    max_d = min(max_d_frac.numerator // max_d_frac.denominator, best_max - 1)

                    for d in range(min_d, max_d + 1):
                        if time.time() > deadline:
                            break
                        if d >= best_max:
                            break
                        rem4 = rem3 - Fraction(1, d)
                        if rem4 <= 0:
                            continue
                        min_e_frac = 1 / rem4
                        min_e = max(d + 1, min_e_frac.numerator // min_e_frac.denominator + 1)
                        max_e_frac = 2 / rem4
                        max_e = min(max_e_frac.numerator // max_e_frac.denominator, best_max - 1)

                        for e in range(min_e, max_e + 1):
                            if time.time() > deadline:
                                break
                            if e >= best_max:
                                break
                            rem5 = rem4 - Fraction(1, e)
                            if rem5 <= 0:
                                continue
                            f_frac = 1 / rem5
                            if f_frac.denominator == 1:
                                f = f_frac.numerator
                                if f > e and f <= 10000000 and f < best_max:
                                    update_best([a, b, c, d, e, f])

    # --- Meet-in-the-middle for 5-term: split into 2+3 subsets from pool [a, M] ---
    # Try various upper bounds M and first-denominator a
    for M in range(50, min(best_max if best_max < float('inf') else 300, 300) + 1, 2):
        if time.time() > deadline:
            break
        if M < 20:
            continue
        # Pool of denominators
        pool = list(range(7, M + 1))
        if len(pool) < 5:
            continue

        # Precompute all 2-subset sums and 3-subset sums
        # For 2-subsets: store in dict sum -> list of (i,j) indices
        sum2_dict = {}
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                s = Fraction(1, pool[i]) + Fraction(1, pool[j])
                if s < target:
                    key = s
                    if key not in sum2_dict:
                        sum2_dict[key] = []
                    sum2_dict[key].append((i, j))

        # For 3-subsets: store in dict sum -> list of (i,j,k) indices
        sum3_dict = {}
        n = len(pool)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    s = Fraction(1, pool[i]) + Fraction(1, pool[j]) + Fraction(1, pool[k])
                    if s < target:
                        key = s
                        if key not in sum3_dict:
                            sum3_dict[key] = []
                        sum3_dict[key].append((i, j, k))

        # Now look for 2-subset + 3-subset = target
        # For each 2-subset sum s2, check if target - s2 is in sum3_dict
        for s2, idx2_list in sum2_dict.items():
            if time.time() > deadline:
                break
            rem3 = target - s2
            if rem3 in sum3_dict:
                for idx2 in idx2_list:
                    for idx3 in sum3_dict[rem3]:
                        # Check that all indices are distinct
                        all_idx = set(idx2) | set(idx3)
                        if len(all_idx) == 5:
                            denoms = [pool[x] for x in sorted(all_idx)]
                            if max(denoms) < best_max:
                                # Verify
                                total = sum(Fraction(1, d) for d in denoms)
                                if total == target:
                                    update_best(denoms)

        # Also try 3+2 (same thing, already covered)
        # And 1+4: single element + 4-subset
        # Precompute 4-subset sums (expensive, only for small pools)
        if n <= 25 and time.time() < deadline - 5:
            sum4_dict = {}
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        for l in range(k + 1, n):
                            s = (Fraction(1, pool[i]) + Fraction(1, pool[j]) +
                                 Fraction(1, pool[k]) + Fraction(1, pool[l]))
                            if s < target:
                                key = s
                                if key not in sum4_dict:
                                    sum4_dict[key] = []
                                sum4_dict[key].append((i, j, k, l))

            for i in range(n):
                if time.time() > deadline:
                    break
                rem4 = target - Fraction(1, pool[i])
                if rem4 in sum4_dict:
                    for idx4 in sum4_dict[rem4]:
                        all_idx = set([i]) | set(idx4)
                        if len(all_idx) == 5:
                            denoms = [pool[x] for x in sorted(all_idx)]
                            if max(denoms) < best_max:
                                total = sum(Fraction(1, d) for d in denoms)
                                if total == target:
                                    update_best(denoms)

    # --- 7-term search via MITM: 3+4 split ---
    if best_max > 80 and time.time() < deadline - 10:
        for M in range(40, min(best_max if best_max < float('inf') else 200, 200) + 1, 3):
            if time.time() > deadline - 5:
                break
            pool = list(range(7, M + 1))
            n = len(pool)
            if n < 7:
                continue

            # 3-subset sums
            sum3 = {}
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        s = Fraction(1, pool[i]) + Fraction(1, pool[j]) + Fraction(1, pool[k])
                        if s < target:
                            if s not in sum3:
                                sum3[s] = []
                            sum3[s].append((i, j, k))

            # 4-subset sums
            sum4 = {}
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        for l in range(k + 1, n):
                            s = (Fraction(1, pool[i]) + Fraction(1, pool[j]) +
                                 Fraction(1, pool[k]) + Fraction(1, pool[l]))
                            if s < target:
                                if s not in sum4:
                                    sum4[s] = []
                                sum4[s].append((i, j, k, l))

            for s3, idx3_list in sum3.items():
                if time.time() > deadline - 3:
                    break
                rem4 = target - s3
                if rem4 in sum4:
                    for idx3 in idx3_list:
                        for idx4 in sum4[rem4]:
                            all_idx = set(idx3) | set(idx4)
                            if len(all_idx) == 7:
                                denoms = [pool[x] for x in sorted(all_idx)]
                                if max(denoms) < best_max:
                                    total = sum(Fraction(1, d) for d in denoms)
                                    if total == target:
                                        update_best(denoms)

    if best_result is not None:
        return best_result

    # Fallback
    return [7, 8, 9, 10000000]
