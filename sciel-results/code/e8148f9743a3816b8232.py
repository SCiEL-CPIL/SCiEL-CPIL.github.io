"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        """
        terms = []
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def winograd_3x3x3_terms():
        """
        Return Winograd's 23-term decomposition for 3x3x3 matrix multiplication.
        This uses the Winograd scheme F(3,3,3) with rank 23.
        Each term is (U_pattern, V_pattern, W_pattern).
        """
        # Winograd's algorithm for 3x3x3 uses specific patterns
        # Based on the Winograd method with m=n=p=3
        # The key idea is to use tensor rank 23 instead of 27
        
        terms = []
        
        # We'll use a known rank-23 decomposition
        # This is based on the work by Winograd and others
        # The patterns are derived from the structure of the multiplication
        
        # For simplicity and correctness, we'll use a systematic approach
        # that guarantees rank <= 23
        
        # Actually, let me use a simpler but still effective approach:
        # Use block decomposition within the 3x3x3 block
        
        # Split 3x3x3 into:
        # - A 2x2x2 block (rank 7 via Strassen)
        # - Remaining parts handled by schoolbook
        
        # But this gives 7 + 2*3 + 3*2 + 3*2 - overlaps... complex
        
        # Instead, let's use a direct rank-23 construction
        # Based on the fact that 3x3x3 has rank at most 23
        
        # Here's a valid rank-23 decomposition:
        # We use the structure from the literature
        
        # For now, use a conservative approach: 
        # Split into one 2x2x2 (Strassen, rank 7) + 20 schoolbook terms
        # Total: 7 + 20 = 27, which is not better
        
        # Better: use multiple 2x2x2 blocks
        # 3x3x3 can be split into:
        # - positions (0,0,0), (0,0,1), (0,1,0), (0,1,1) -> 2x2x2 block
        # - remaining 19 terms
        # Total: 7 + 19 = 26, still not great
        
        # Let me use a different approach: 
        # Use the fact that we can do better with careful construction
        
        # For this implementation, I'll use a known result:
        # 3x3x3 matrix multiplication has rank 23
        # Here's a construction:
        
        # The 23 terms are indexed by (a,b,c) where a,b,c in {0,1,2}
        # with some structure
        
        # Actually, let me just use schoolbook for 3x3x3 for now
        # and focus on getting the 2x2x2 blocks right
        # We can improve this later
        
        return None  # Not implemented yet, use schoolbook for 3x3x3

    def build_block_decomposition(i_groups, j_groups, k_groups):
        """
        Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise.
        """
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W}

    def count_rank(i_groups, j_groups, k_groups):
        """Count the total rank for a given partition."""
        total = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total += 7
                    else:
                        total += si * sj * sk
        return total

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Generate all possible partitions
    # A partition of [0,1,...,d-1] is a set of non-empty subsets that cover all elements
    def generate_partitions(d):
        """Generate all partitions of [0, 1, ..., d-1]."""
        if d == 0:
            yield []
        elif d == 1:
            yield [[0]]
        else:
            for part in generate_partitions(d - 1):
                # Add d-1 to each existing block
                for i in range(len(part)):
                    new_part = [list(block) for block in part]
                    new_part[i].append(d - 1)
                    yield new_part
                # Add d-1 as a new block
                new_part = [list(block) for block in part]
                new_part.append([d - 1])
                yield new_part

    # Generate partitions for each dimension
    i_partitions = list(generate_partitions(n))
    j_partitions = list(generate_partitions(m))
    k_partitions = list(generate_partitions(p))

    # Filter partitions to keep only reasonable sizes (avoid too many small blocks)
    # For n=3: partitions are [[0,1,2]], [[0],[1,2]], [[0,1],[2]], [[0],[1],[2]]
    # For m=5: many more, but we'll filter
    # For p=6: even more
    
    # Limit the number of partitions to consider
    max_partitions_per_dim = 50
    
    i_partitions = i_partitions[:max_partitions_per_dim]
    j_partitions = j_partitions[:max_partitions_per_dim]
    k_partitions = k_partitions[:max_partitions_per_dim]

    # Generate candidate partitions and sort by estimated rank
    candidates = []
    for ig in i_partitions:
        for jg in j_partitions:
            for kg in k_partitions:
                r = count_rank(ig, jg, kg)
                candidates.append((r, ig, jg, kg))
    
    # Sort by rank (ascending)
    candidates.sort(key=lambda x: x[0])
    
    # Try candidates in order of rank
    for r, ig, jg, kg in candidates:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        result = build_block_decomposition(ig, jg, kg)
        if verify(result["U"], result["V"], result["W"], r):
            best_result = result
            best_rank = r
            if best_rank <= 40:  # Good enough, stop searching
                break

    return best_result
# EVOLVE-BLOCK-END
