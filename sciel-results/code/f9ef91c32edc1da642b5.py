import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Precompute gcd table for values up to 256
    max_sum = 256
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1):
            gcd_table[i][j] = math.gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    def fast_pairwise_coprime(sums_set):
        distinct = list(sums_set)
        n = len(distinct)
        for i in range(n):
            vi = distinct[i]
            for j in range(i + 1, n):
                vj = distinct[j]
                if gcd_table[vi][vj] != 1:
                    return False
        return True

    # STRUCTURAL CHANGE: Algebraic Diophantine formulation.
    # The 3x3 sum matrix S[i][j] = a_i + b_j satisfies:
    #   S[i][j] - S[i][k] = b_j - b_k  (row differences are column-independent)
    #   S[i][j] - S[k][j] = a_i - a_k  (column differences are row-independent)
    #
    # Thus the entire matrix is determined by 5 free entries:
    #   s00, s01, s02 (first row), s10, s20 (first column, rows 1,2)
    #   s11 = s10 + (s01 - s00),  s12 = s10 + (s02 - s00)
    #   s21 = s20 + (s01 - s00),  s22 = s20 + (s02 - s00)
    #
    # All 9 sums must be odd (pairwise coprime => at most one even),
    # so one set is all-odd, the other all-even.
    #
    # Given the 5 free values, b0 is determined by bounds:
    #   a_i = s_i0 - b0,  b_j = b0 + (s0j - s00)
    # We search over the 5 free odd values (pairwise coprime) and solve for b0.

    # Phase 1: Algebraic search with tightly bounded free variables
    # Sums must be in [2, 256] and odd, so in [3, 255]
    # Keep free values small to ensure derived values stay in range
    max_free = 50
    odd_candidates = [x for x in range(3, max_free + 1, 2)]

    # Precompute coprime pairs among candidates for fast filtering
    n_cand = len(odd_candidates)
    cop = [[False] * n_cand for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if gcd_table[odd_candidates[i]][odd_candidates[j]] == 1:
                cop[i][j] = cop[j][i] = True

    # Index map: value -> index in odd_candidates
    val_to_idx = {v: i for i, v in enumerate(odd_candidates)}

    for s00 in odd_candidates:
        if time.time() > deadline:
            break
        i0 = val_to_idx[s00]
        for s01 in odd_candidates:
            if s01 <= s00:
                continue
            i1 = val_to_idx[s01]
            if not cop[i0][i1]:
                continue
            d1 = s01 - s00
            for s02 in odd_candidates:
                if s02 <= s01:
                    continue
                i2 = val_to_idx[s02]
                if not cop[i0][i2] or not cop[i1][i2]:
                    continue
                d2 = s02 - s00
                for s10 in odd_candidates:
                    if s10 == s00 or s10 == s01 or s10 == s02:
                        continue
                    i3 = val_to_idx[s10]
                    if not cop[i0][i3] or not cop[i1][i3] or not cop[i2][i3]:
                        continue
                    s11 = s10 + d1
                    s12 = s10 + d2
                    if s11 > 256 or s12 > 256 or s11 < 2 or s12 < 2:
                        continue
                    if s11 % 2 == 0 or s12 % 2 == 0:
                        continue
                    if s11 == s00 or s11 == s01 or s11 == s02 or s11 == s10:
                        continue
                    if s12 == s00 or s12 == s01 or s12 == s02 or s12 == s10 or s12 == s11:
                        continue
                    # Check coprimality of s11, s12 against all existing
                    if (gcd_table[s00][s11] != 1 or gcd_table[s01][s11] != 1 or
                        gcd_table[s02][s11] != 1 or gcd_table[s10][s11] != 1):
                        continue
                    if (gcd_table[s00][s12] != 1 or gcd_table[s01][s12] != 1 or
                        gcd_table[s02][s12] != 1 or gcd_table[s10][s12] != 1 or
                        gcd_table[s11][s12] != 1):
                        continue
                    for s20 in odd_candidates:
                        if time.time() > deadline:
                            break
                        if s20 == s00 or s20 == s01 or s20 == s02 or s20 == s10 or s20 == s11 or s20 == s12:
                            continue
                        i4 = val_to_idx[s20]
                        if (not cop[i0][i4] or not cop[i1][i4] or not cop[i2][i4] or
                            not cop[i3][i4]):
                            continue
                        s21 = s20 + d1
                        s22 = s20 + d2
                        if s21 > 256 or s22 > 256 or s21 < 2 or s22 < 2:
                            continue
                        if s21 % 2 == 0 or s22 % 2 == 0:
                            continue
                        if (s21 == s00 or s21 == s01 or s21 == s02 or s21 == s10 or
                            s21 == s11 or s21 == s12 or s21 == s20):
                            continue
                        if (s22 == s00 or s22 == s01 or s22 == s02 or s22 == s10 or
                            s22 == s11 or s22 == s12 or s22 == s20 or s22 == s21):
                            continue
                        if (gcd_table[s00][s21] != 1 or gcd_table[s01][s21] != 1 or
                            gcd_table[s02][s21] != 1 or gcd_table[s10][s21] != 1 or
                            gcd_table[s11][s21] != 1 or gcd_table[s12][s21] != 1 or
                            gcd_table[s20][s21] != 1):
                            continue
                        if (gcd_table[s00][s22] != 1 or gcd_table[s01][s22] != 1 or
                            gcd_table[s02][s22] != 1 or gcd_table[s10][s22] != 1 or
                            gcd_table[s11][s22] != 1 or gcd_table[s12][s22] != 1 or
                            gcd_table[s20][s22] != 1 or gcd_table[s21][s22] != 1):
                            continue

                        # All 9 sums are distinct and pairwise coprime.
                        # Solve for b0: a_i = s_i0 - b0, b_j = b0 + d_j
                        # b0 must satisfy: 1 <= s_i0 - b0 <= 128 for i=0,1,2
                        #                   1 <= b0 <= 128
                        #                   1 <= b0 + d1 <= 128
                        #                   1 <= b0 + d2 <= 128
                        b0_lo = max(1, s00 - 128, s10 - 128, s20 - 128)
                        b0_hi = min(128, s00 - 1, s10 - 1, s20 - 1, 128 - d1, 128 - d2)

                        for b0 in range(b0_lo, b0_hi + 1):
                            a0 = s00 - b0
                            a1 = s10 - b0
                            a2 = s20 - b0
                            b1 = b0 + d1
                            b2 = b0 + d2

                            if len({a0, a1, a2}) != 3 or len({b0, b1, b2}) != 3:
                                continue

                            A = [a0, a1, a2]
                            B = [b0, b1, b2]
                            return [A, B]

    # Phase 2: Brute-force fallback (guaranteed to find a solution)
    odd_vals = [x for x in range(1, 129) if x % 2 == 1]
    even_vals = [x for x in range(2, 129) if x % 2 == 0]
    odds_small = [x for x in odd_vals if x <= 40]
    evens_small = [x for x in even_vals if x <= 40]

    A_triples = list(combinations(odds_small, 3))
    B_triples = list(combinations(evens_small, 3))

    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    best_count = 0

    for A in A_triples:
        if time.time() > deadline:
            break
        a1, a2, a3 = A
        for B in B_triples:
            if time.time() > deadline:
                break
            b1, b2, b3 = B
            s = (a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3)
            sums_set = set(s)
            count = len(sums_set)
            if count <= best_count:
                continue
            if fast_pairwise_coprime(sums_set):
                best_count = count
                best_A = list(A)
                best_B = list(B)
                if best_count == 9:
                    return [best_A, best_B]

    return [best_A, best_B]
