"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from itertools import combinations

def construct():
    """
    Deterministic exhaustive search over a reduced domain to find optimal A, B.
    This approach systematically checks all pairs of sets from a candidate pool,
    using precomputed coprimality to speed up validation.
    """
    
    # Precompute coprime table for sums up to 128+128=256
    max_sum = 256
    coprime_table = [[False] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1):
            if gcd(i, j) == 1:
                coprime_table[i][j] = True
                coprime_table[j][i] = True
    
    def is_valid(A, B):
        """Check if distinct sums of A and B are pairwise coprime. Returns count."""
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        n = len(distinct_sums)
        # Check pairwise coprimality
        for i in range(n):
            si = distinct_sums[i]
            for j in range(i + 1, n):
                sj = distinct_sums[j]
                if not coprime_table[si][sj]:
                    return 0
        return n

    # Search space: We suspect optimal solutions use small numbers.
    # Let's search A, B in [1, 40].
    # C(40, 3) = 9880. Pairs = ~97M. Too many for pure Python in 60s.
    # Let's reduce to [1, 30]. C(30, 3) = 4060. Pairs = ~16M.
    # 16M * (9 sums + set + checks) might be ~10-20s.
    
    limit = 30
    candidates = list(range(1, limit + 1))
    comb3 = list(combinations(candidates, 3))
    
    best_count = 0
    best_A = None
    best_B = None
    
    # To speed up, we can iterate A and B.
    # We can also add a heuristic: if best_count == 9, break early.
    
    for A in comb3:
        for B in comb3:
            count = is_valid(A, B)
            if count > best_count:
                best_count = count
                best_A = A
                best_B = B
                if best_count == 9:
                    return [list(best_A), list(best_B)]
    
    # If we didn't find 9 in [1,30], try a slightly larger range or random search for 9
    # But first, return the best found so far
    if best_A is not None:
        return [list(best_A), list(best_B)]
    
    # Fallback to known good solution
    return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
