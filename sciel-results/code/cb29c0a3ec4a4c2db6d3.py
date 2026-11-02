import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank 18 as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

    # Build tensor T of shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find rank-15 decomposition.
    # Use deterministic exhaustive search over support patterns with small supports.
    
    # Generate all "elementary" rank-1 tensors: each has support size 1 in each of U, V, W
    # plus some "merged" patterns where supports have size 2.
    
    # For a pattern with supports (S_u, S_v, S_w), the rank-1 tensor is:
    #   sum_{a in S_u, b in S_v, c in S_w} e_a e_b e_c  (outer product of indicator vectors)
    # But we only want it to contribute to the matrix multiplication tensor positions
    # where (a, b, c) corresponds to a valid (i, j, k) triple with consistent indices.
    
    # A position (a, b, c) is "active" in the MM tensor iff:
    #   a = i*m + j, b = j*p + k, c = k*n + i for some i, j, k
    # i.e., the indices are consistent.
    
    def is_active(a, b, c):
        i_u, j_u = divmod(a, m)
        j_v, k_v = divmod(b, p)
        k_w, i_w = divmod(c, n)
        return j_u == j_v and k_v == k_w and i_u == i_w
    
    def active_positions(supp_u, supp_v, supp_w):
        """Return list of (a, b, c) triples in the support product that are active."""
        result = []
        for a in supp_u:
            for b in supp_v:
                for c in supp_w:
                    if is_active(a, b, c):
                        result.append((a, b, c))
        return result
    
    # Build all 18 active positions
    all_active = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                all_active.add((a, b, c))
    
    # Generate candidate patterns:
    # Type 1: single positions (18 of them) - each covers exactly one active position
    candidates = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                candidates.append(((a,), (b,), (c,)))
    
    # Type 2: pairs of positions that share at least one index dimension
    # For efficiency, only include patterns that cover exactly 2 active positions
    # and where the non-active entries in the support product are zero or can be cancelled.
    # Actually, for a pattern to be "clean", the support product should contain ONLY active positions.
    # Let's generate all small support patterns (size <= 2 in each dimension) and filter.
    
    all_indices_u = list(range(n * m))
    all_indices_v = list(range(m * p))
    all_indices_w = list(range(p * n))
    
    # Generate patterns with supports of size 1 or 2
    seen_patterns = set()
    for su in range(1, 3):
        for sv in range(1, 3):
            for sw in range(1, 3):
                if su * sv * sw > 4:
                    continue
                for supp_u in combinations(all_indices_u, su):
                    for supp_v in combinations(all_indices_v, sv):
                        for supp_w in combinations(all_indices_w, sw):
                            active = active_positions(supp_u, supp_v, supp_w)
                            # Pattern is valid if it covers 1 or 2 active positions
                            # and the total support product size equals the number of active positions
                            # (i.e., no "waste" non-active entries)
                            total = len(supp_u) * len(supp_v) * len(supp_w)
                            if len(active) in (1, 2) and total == len(active):
                                pattern = (supp_u, supp_v, supp_w)
                                key = frozenset(active)
                                if key not in seen_patterns:
                                    seen_patterns.add(key)
                                    candidates.append(pattern)
    
    # Now search for a set of <= 15 patterns with coefficients +/-1, +/-0.5
    # that exactly cover all 18 active positions.
    # This is a small combinatorial problem. We'll use a simple backtracking search.
    
    target = frozenset(all_active)
    
    # Sort candidates by number of active positions covered (prefer larger coverage)
    def pattern_coverage(pat):
        return len(active_positions(pat[0], pat[1], pat[2]))
    
    candidates.sort(key=pattern_coverage, reverse=True)
    
    # For each candidate, precompute which active positions it covers
    cand_active = []
    for pat in candidates:
        act = frozenset(active_positions(pat[0], pat[1], pat[2]))
        cand_active.append(act)
    
    # Search for combination of <= 15 patterns
    best_rank = rank_fb
    best_solution = None
    
    # Use recursive backtracking
    import sys
    sys.setrecursionlimit(10000)
    
    def backtrack(remaining_target, start_idx, current_pats, current_coefs):
        nonlocal best_rank, best_solution
        if len(current_pats) > 15:
            return
        if not remaining_target:
            if len(current_pats) < best_rank:
                best_rank = len(current_pats)
                best_solution = (current_pats[:], current_coefs[:])
            return
        if start_idx >= len(candidates):
            return
        
        # Try each candidate from start_idx onward
        for idx in range(start_idx, len(candidates)):
            pat_active = cand_active[idx]
            # Check if this pattern's active positions are a subset of remaining
            if pat_active.issubset(remaining_target):
                new_remaining = remaining_target - pat_active
                current_pats.append(candidates[idx])
                # Try coefficients: 1, -1, 0.5, -0.5
                for coef in [1, -1, 0.5, -0.5]:
                    current_coefs.append(coef)
                    backtrack(new_remaining, idx + 1, current_pats, current_coefs)
                    current_coefs.pop()
                current_pats.pop()
    
    backtrack(target, 0, [], [])
    
    if best_solution is not None:
        pats, coefs = best_solution
        rank = len(pats)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for r, (pat, coef) in enumerate(zip(pats, coefs)):
            supp_u, supp_v, supp_w = pat
            for a in supp_u:
                U[a][r] = coef
            for b in supp_v:
                V[b][r] = coef
            for c in supp_w:
                W[c][r] = coef
        return {"U": U, "V": V, "W": W}
    
    return fallback
# EVOLVE-BLOCK-END

