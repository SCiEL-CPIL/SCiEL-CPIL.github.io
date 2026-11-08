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

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

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

    def find_best_partition():
        """
        Find the best partition of indices into groups.
        For each dimension, we partition the indices into contiguous groups.
        Then for each combination of groups (one from each dimension),
        we use Strassen if all sizes are 2, else schoolbook.
        """
        # For dimension of size s, generate all ways to partition into contiguous groups
        # where each group has size 1 or 2 (since Strassen only works for 2x2x2)
        # Actually, we can have larger groups but they'll use schoolbook
        # Let's be smarter: only consider partitions where groups are size 1 or 2
        # because larger groups just use schoolbook anyway
        
        def get_partitions(s):
            """Get all ways to partition [0..s-1] into contiguous groups of size 1 or 2."""
            results = []
            def backtrack(remaining, current):
                if not remaining:
                    results.append(tuple(current))
                    return
                if remaining[0] == 1:
                    backtrack(remaining[1:], current + [1])
                if len(remaining) >= 2:
                    backtrack(remaining[2:], current + [2])
            backtrack(list(range(s)), [])
            # Convert to actual index groups
            final = []
            for sizes in results:
                groups = []
                start = 0
                for sz in sizes:
                    groups.append(list(range(start, start + sz)))
                    start += sz
                final.append(tuple(tuple(g) for g in groups))
            return final
        
        i_partitions = get_partitions(n)  # 3
        j_partitions = get_partitions(m)  # 5
        k_partitions = get_partitions(p)  # 6
        
        best_rank = n * m * p  # schoolbook
        best_config = None
        
        for ip in i_partitions:
            for jp in j_partitions:
                for kp in k_partitions:
                    total_rank = 0
                    for ig in ip:
                        for jg in jp:
                            for kg in kp:
                                si, sj, sk = len(ig), len(jg), len(kg)
                                if si == 2 and sj == 2 and sk == 2:
                                    total_rank += 7  # Strassen
                                else:
                                    total_rank += si * sj * sk  # schoolbook
                    
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best_config = (ip, jp, kp)
        
        return best_rank, best_config

    def build_decomposition_from_config(config):
        """Build decomposition from a given partition configuration."""
        ip, jp, kp = config
        strassen_terms = strassen_2x2x2_terms()
        
        # Calculate total rank
        total_rank = 0
        for ig in ip:
            for jg in jp:
                for kg in kp:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total_rank += 7
                    else:
                        total_rank += si * sj * sk
        
        # Allocate matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        for ig in ip:
            for jg in jp:
                for kg in kp:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        # Use Strassen
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
                        # Schoolbook for this sub-block
                        for li in range(si):
                            for lj in range(sj):
                                for lk in range(sk):
                                    gi, gj, gk = ig[li], jg[lj], kg[lk]
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    # Try to find the best partition
    best_rank, best_config = find_best_partition()
    
    if best_config is not None:
        result = build_decomposition_from_config(best_config)
        rank = result.pop("rank")
        
        # Verify
        if verify(result["U"], result["V"], result["W"], rank):
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
