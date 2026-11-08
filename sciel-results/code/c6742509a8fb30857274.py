"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

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

    def get_strassen_terms():
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11) -> C01 += 1, C11 += 1
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01) -> C11 += 1
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11) -> C00 += 1
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def try_rank_reduction():
        """
        Try to find a rank reduction by cleverly combining schoolbook terms.
        For 3x5x6, try to find patterns that allow rank reduction.
        """
        if time.time() > deadline:
            return None
        
        strassen_terms = get_strassen_terms()
        
        # Try multiple partition strategies and pick the best
        partitions = [
            ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
            ([[0], [1, 2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
            ([[0, 1], [2]], [[0], [1, 2], [3, 4]], [[0, 1], [2, 3], [4, 5]]),
            ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0], [1, 2], [3, 4], [5]]),
            ([[0, 1], [2]], [[0], [1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
            ([[0], [1], [2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
            ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0, 1], [2], [3], [4, 5]]),
            ([[0, 1], [2]], [[0, 1], [2], [3], [4]], [[0, 1], [2, 3], [4, 5]]),
            ([[0], [1, 2]], [[0, 1], [2, 3], [4]], [[0], [1], [2, 3], [4, 5]]),
            ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4], [5]]),
        ]
        
        best_result = None
        best_rank = float('inf')
        
        for i_groups, j_groups, k_groups in partitions:
            if time.time() > deadline:
                break
            
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
            
            if total_rank >= best_rank:
                continue
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            col = 0
            for ig, jg, kg, r in block_info:
                si, sj, sk = len(ig), len(jg), len(kg)
                if si == 2 and sj == 2 and sk == 2:
                    # Use Strassen: accumulate into the same columns
                    # Each Strassen term gets its own column
                    for (u_pat, v_pat, w_pat) in strassen_terms:
                        for (li, lj), val_u in u_pat.items():
                            gi, gj = ig[li], jg[lj]
                            U[gi * m + gj][col] += val_u
                        for (lj, lk), val_v in v_pat.items():
                            gj, gk = jg[lj], kg[lk]
                            V[gj * p + gk][col] += val_v
                        for (lk, li), val_w in w_pat.items():
                            gk, gi = kg[lk], ig[li]
                            W[gk * n + gi][col] += val_w
                        col += 1
                else:
                    for li in range(si):
                        for lj in range(sj):
                            for lk in range(sk):
                                gi, gj, gk = ig[li], jg[lj], kg[lk]
                                U[gi * m + gj][col] += 1
                                V[gj * p + gk][col] += 1
                                W[gk * n + gi][col] += 1
                                col += 1
            
            if col != total_rank:
                continue
                
            result = {"U": U, "V": V, "W": W}
            
            if verify(U, V, W, total_rank):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = result
        
        return best_result

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try improved decomposition
    improved = try_rank_reduction()
    if improved is not None:
        improved_rank = len(improved["U"][0])
        if improved_rank < best_rank:
            best_result = improved
            best_rank = improved_rank

    return best_result
# EVOLVE-BLOCK-END
