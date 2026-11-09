import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s safety margin out of 90s

    def is_pairwise_coprime(nums):
        nums = list(nums)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True

    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) == 0:
            return -1
        if not is_pairwise_coprime(sums):
            return -1
        return len(sums)

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = evaluate(best_A, best_B)  # 5

    # Precompute coprime compatibility for sums in [2, 256]
    # cop[s][t] = True if gcd(s, t) == 1
    cop = [[True] * 257 for _ in range(257)]
    for s in range(2, 257):
        for t in range(s + 1, 257):
            cop[s][t] = cop[t][s] = (math.gcd(s, t) == 1)

    # Backtracking CSP over 9 grid variables S[i][j] = A[i] + B[j]
    # Rank-1 constraint: S[i][j] - S[i][k] = S[i'][j] - S[i'][k] for all i,i',j,k
    # Equivalently: S[i][j] + S[i'][k] = S[i][k] + S[i'][j]
    # We assign row by row. Row 0 is free (defines A offsets relative to B).
    # For row i > 0, given row 0, the constraints are:
    #   S[i][j] - S[0][j] = S[i][k] - S[0][k] for all j,k
    # This means S[i][j] = S[i][0] + (S[0][j] - S[0][0])
    # So once S[i][0] is chosen, the entire row is determined!
    #
    # Variables: S[0][0], S[0][1], S[0][2], S[1][0], S[2][0]
    # Derived: S[i][j] = S[i][0] + (S[0][j] - S[0][0]) for i=1,2 and j=1,2
    #
    # Bounds: 1 <= A[i] <= 128, 1 <= B[j] <= 128
    # A[i] = S[i][0] - B[0], B[j] = S[0][j] - A[0]
    # Let B[0] = b0, A[0] = a0
    # S[0][j] = a0 + B[j], S[i][0] = A[i] + b0
    # S[i][j] = A[i] + B[j] = S[i][0] + S[0][j] - S[0][0]
    #
    # Constraints:
    # 1 <= S[0][0] - b0 <= 128  => S[0][0] - 128 <= b0 <= S[0][0] - 1
    # 1 <= S[i][0] - b0 <= 128  => S[i][0] - 128 <= b0 <= S[i][0] - 1
    # 1 <= S[0][j] - a0 <= 128  => S[0][j] - 128 <= a0 <= S[0][j] - 1
    # 1 <= b0 <= 128
    # 1 <= a0 <= 128
    #
    # So b0 in [max(1, S[0][0]-128, S[1][0]-128, S[2][0]-128), min(128, S[0][0]-1, S[1][0]-1, S[2][0]-1)]
    # And a0 in [max(1, S[0][0]-128, S[0][1]-128, S[0][2]-128), min(128, S[0][0]-1, S[0][1]-1, S[0][2]-1)]
    #
    # Also all 9 S values must be in [2, 256] (since min sum is 1+1=2, max is 128+128=256)
    # And all 9 must be distinct (for score 9) and pairwise coprime.
    #
    # Strategy: enumerate S[0][0], S[0][1], S[0][2] (row 0), then S[1][0], S[2][0] (first column)
    # Derive the rest, check all constraints.

    # For efficiency, precompute which values work in row 0
    # S[0][0], S[0][1], S[0][2] must be pairwise coprime and in [2, 256]
    # Also they must be distinct.

    # Let's enumerate row 0 first, then first column entries.
    # Row 0: s00, s01, s02 in [2, 256], pairwise coprime, distinct
    # Col 0: s10, s20 in [2, 256]
    # Derived: s11 = s10 + s01 - s00, s12 = s10 + s02 - s00
    #          s21 = s20 + s01 - s00, s22 = s20 + s02 - s00
    # All 9 must be in [2, 256], distinct, pairwise coprime.
    # Then check if valid b0, a0 exist.

    # Precompute coprime triples for row 0
    # This is still large, so let's use backtracking with pruning.

    def solve_csp():
        """Backtracking search for 9 distinct pairwise-coprime sums in a rank-1 grid."""
        found = [False]
        best = [None, None]

        # Enumerate s00, s01, s02 (row 0)
        # s00 in [2, 256], s01 in [2, 256], s02 in [2, 256]
        # Must be pairwise coprime and distinct.
        # For each valid row 0, enumerate s10, s20.

        for s00 in range(2, 257):
            if time.time() > deadline - 3:
                return None, None
            if found[0]:
                return best[0], best[1]
            for s01 in range(2, 257):
                if time.time() > deadline - 3:
                    return None, None
                if s01 == s00:
                    continue
                if not cop[s00][s01]:
                    continue
                for s02 in range(2, 257):
                    if time.time() > deadline - 3:
                        return None, None
                    if s02 == s00 or s02 == s01:
                        continue
                    if not cop[s00][s02] or not cop[s01][s02]:
                        continue

                    # Row 0 is valid. Now enumerate s10, s20.
                    d1 = s01 - s00  # S[0][1] - S[0][0]
                    d2 = s02 - s00  # S[0][2] - S[0][0]

                    for s10 in range(2, 257):
                        if time.time() > deadline - 3:
                            return None, None
                        if s10 == s00:
                            continue
                        if not cop[s00][s10]:
                            continue

                        s11 = s10 + d1
                        s12 = s10 + d2

                        # Check s11, s12 bounds and coprimality with existing
                        if s11 < 2 or s11 > 256 or s12 < 2 or s12 > 256:
                            continue
                        if s11 == s00 or s11 == s01 or s11 == s02 or s11 == s10:
                            continue
                        if s12 == s00 or s12 == s01 or s12 == s02 or s12 == s10 or s12 == s11:
                            continue
                        if not cop[s00][s11] or not cop[s01][s11] or not cop[s02][s11] or not cop[s10][s11]:
                            continue
                        if not cop[s00][s12] or not cop[s01][s12] or not cop[s02][s12] or not cop[s10][s12] or not cop[s11][s12]:
                            continue

                        for s20 in range(2, 257):
                            if time.time() > deadline - 3:
                                return None, None
                            if s20 == s00 or s20 == s10:
                                continue
                            if not cop[s00][s20] or not cop[s10][s20]:
                                continue

                            s21 = s20 + d1
                            s22 = s20 + d2

                            if s21 < 2 or s21 > 256 or s22 < 2 or s22 > 256:
                                continue

                            # Check all distinctness
                            existing = {s00, s01, s02, s10, s11, s12}
                            if s20 in existing or s21 in existing or s22 in existing:
                                continue
                            if s21 == s22:
                                continue

                            # Check coprimality of s20, s21, s22 with all existing
                            all_prev = [s00, s01, s02, s10, s11, s12]
                            ok = True
                            for v in [s20, s21, s22]:
                                for pv in all_prev:
                                    if not cop[v][pv]:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                continue
                            if not cop[s20][s21] or not cop[s20][s22] or not cop[s21][s22]:
                                continue

                            # All 9 values are distinct and pairwise coprime.
                            # Check if valid b0, a0 exist.
                            # b0 in [max(1, s00-128, s10-128, s20-128), min(128, s00-1, s10-1, s20-1)]
                            b0_min = max(1, s00 - 128, s10 - 128, s20 - 128)
                            b0_max = min(128, s00 - 1, s10 - 1, s20 - 1)

                            if b0_min > b0_max:
                                continue

                            # a0 in [max(1, s00-128, s01-128, s02-128), min(128, s00-1, s01-1, s02-1)]
                            a0_min = max(1, s00 - 128, s01 - 128, s02 - 128)
                            a0_max = min(128, s00 - 1, s01 - 1, s02 - 1)

                            if a0_min > a0_max:
                                continue

                            # Pick b0 = b0_min, a0 = a0_min
                            b0 = b0_min
                            a0 = a0_min

                            A = [a0, s10 - b0, s20 - b0]
                            B = [b0, s01 - a0, s02 - a0]

                            # Final verification
                            if len(set(A)) != 3 or len(set(B)) != 3:
                                continue
                            if any(v < 1 or v > 128 for v in A + B):
                                continue

                            sums = set()
                            for a in A:
                                for b in B:
                                    sums.add(a + b)

                            if len(sums) == 9 and is_pairwise_coprime(sums):
                                found[0] = True
                                best[0] = A
                                best[1] = B
                                return A, B

        return None, None

    result = solve_csp()
    if result[0] is not None:
        return [list(result[0]), list(result[1])]

    # Fallback: if CSP didn't find score 9, use the original prime-grid approach quickly
    # then local search
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(256)
    prime_set = set(primes)

    # Quick prime-grid search
    for d1 in range(1, 128):
        if time.time() > deadline - 5:
            break
        for d2 in range(d1 + 1, 128):
            if time.time() > deadline - 5:
                break
            starts = [p for p in range(2, 256 - d2 + 1)
                      if p in prime_set and (p + d1) in prime_set and (p + d2) in prime_set]
            if len(starts) < 3:
                continue
            n_s = len(starts)
            for i in range(n_s):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, n_s):
                    if time.time() > deadline - 3:
                        break
                    for k in range(j + 1, n_s):
                        if time.time() > deadline - 3:
                            break
                        P00, P10, P20 = starts[i], starts[j], starts[k]
                        P = [[P00, P00+d1, P00+d2], [P10, P10+d1, P10+d2], [P20, P20+d1, P20+d2]]
                        all_p = [P[i][j] for i in range(3) for j in range(3)]
                        if len(set(all_p)) != 9:
                            continue
                        x_min = max(1, P00-128, P10-128, P20-128)
                        x_max = min(128-d2, P00-1, P10-1, P20-1)
                        if x_min > x_max:
                            continue
                        x = x_min
                        A = [P00-x, P10-x, P20-x]
                        B = [x, x+d1, x+d2]
                        if len(set(A)) != 3 or len(set(B)) != 3:
                            continue
                        if any(v < 1 or v > 128 for v in A+B):
                            continue
                        sums = set(a+b for a in A for b in B)
                        if len(sums) == 9 and is_pairwise_coprime(sums):
                            return [A, B]

    # Local search fallback
    import random
    def random_set():
        return sorted(random.sample(range(1, 129), 3))

    for _ in range(30):
        if time.time() > deadline - 2:
            break
        A = random_set()
        B = random_set()
        curr = evaluate(A, B)
        for _ in range(3000):
            if time.time() > deadline - 2:
                break
            if curr >= best_score:
                if curr > best_score:
                    best_score = curr
                    best_A, best_B = A[:], B[:]
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old = A[idx]
                A[idx] = random.randint(1, 128)
                while len(set(A)) < 3:
                    A[idx] = random.randint(1, 128)
                A.sort()
                ns = evaluate(A, B)
                if ns < curr:
                    A[idx] = old
                    A.sort()
                else:
                    curr = ns
            else:
                idx = random.randint(0, 2)
                old = B[idx]
                B[idx] = random.randint(1, 128)
                while len(set(B)) < 3:
                    B[idx] = random.randint(1, 128)
                B.sort()
                ns = evaluate(A, B)
                if ns < curr:
                    B[idx] = old
                    B.sort()
                else:
                    curr = ns

    return [list(best_A), list(best_B)]
