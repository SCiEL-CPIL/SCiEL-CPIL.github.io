import math
from itertools import combinations

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1]
    best_score = calc_score(best)
    
    # 1. Search Geometric Progressions explicitly
    # U = {0, 1, B, B^2, ..., B^(N-1)}
    for B in range(2, 50):
        for N in range(2, 10):
            U = [0]
            val = 1
            for i in range(N - 1):
                U.append(val)
                val *= B
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best = U

    # 2. Search Full Binary Sets (all subsets of digits)
    for B in range(2, 20):
        for n in range(2, 8):
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            U = all_vals
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best = U

            # 3. Search Subsets of Full Binary Sets (limited)
            if n <= 4:
                # Only search small subsets to avoid combinatorial explosion
                for size in range(2, min(5, len(all_vals) + 1)):
                    for combo in combinations(all_vals, size):
                        U_sub = [0] + list(combo)
                        U_sub = list(set(U_sub))
                        if len(U_sub) != size + 1:
                            continue
                        s_sub = calc_score(U_sub)
                        if s_sub > best_score:
                            best_score = s_sub
                            best = U_sub

    return best
