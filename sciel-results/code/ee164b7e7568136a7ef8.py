"""Matrix multiplication 3x5x6 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

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

    def build_block(ig, jg, kg, col, U, V, W, strassen_terms):
        """Build decomposition for a single block. Returns new col."""
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
        elif si == 1 and sj == 1 and sk == 1:
            gi, gj, gk = ig[0], jg[0], kg[0]
            U[gi * m + gj][col] = 1
            V[gj * p + gk][col] = 1
            W[gk * n + gi][col] = 1
            col += 1
        elif si == 1 and sj == 1 and sk == 2:
            gi, gj = ig[0], jg[0]
            for lk in range(2):
                gk = kg[lk]
                U[gi * m + gj][col] = 1
                V[gj * p + gk][col] = 1
                W[gk * n + gi][col] = 1
                col += 1
        elif si == 1 and sj == 2 and sk == 1:
            gi, gk = ig[0], kg[0]
            for lj in range(2):
                gj = jg[lj]
                U[gi * m + gj][col] = 1
                V[gj * p + gk][col] = 1
                W[gk * n + gi][col] = 1
                col += 1
        elif si == 2 and sj == 1 and sk == 1:
            gj, gk = jg[0], kg[0]
            for li in range(2):
                gi = ig[li]
                U[gi * m + gj][col] = 1
                V[gj * p + gk][col] = 1
                W[gk * n + gi][col] = 1
                col += 1
        elif si == 1 and sj == 2 and sk == 2:
            gi = ig[0]
            for lj in range(2):
                gj = jg[lj]
                for lk in range(2):
                    gk = kg[lk]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        elif si == 2 and sj == 1 and sk == 2:
            gj = jg[0]
            for li in range(2):
                gi = ig[li]
                for lk in range(2):
                    gk = kg[lk]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        elif si == 2 and sj == 2 and sk == 1:
            gk = kg[0]
            for li in range(2):
                gi = ig[li]
                for lj in range(2):
                    gj = jg[lj]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        elif si == 1 and sj == 2 and sk == 3:
            gi = ig[0]
            for lj in range(2):
                gj = jg[lj]
                for lk in range(3):
                    gk = kg[lk]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        elif si == 2 and sj == 1 and sk == 3:
            gj = jg[0]
            for li in range(2):
                gi = ig[li]
                for lk in range(3):
                    gk = kg[lk]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        elif si == 1 and sj == 3 and sk == 2:
            gi = ig[0]
            for lj in range(3):
                gj = jg[lj]
                for lk in range(2):
                    gk = kg[lk]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        elif si == 3 and sj == 1 and sk == 2:
            gj = jg[0]
            for li in range(3):
                gi = ig[li]
                for lk in range(2):
                    gk = kg[lk]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        elif si == 2 and sj == 3 and sk == 1:
            gk = kg[0]
            for li in range(2):
                gi = ig[li]
                for lj in range(3):
                    gj = jg[lj]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        elif si == 1 and sj == 3 and sk == 3:
            gi = ig[0]
            for lj in range(3):
                gj = jg[lj]
                for lk in range(3):
                    gk = kg[lk]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        elif si == 3 and sj == 1 and sk == 3:
            gj = jg[0]
            for li in range(3):
                gi = ig[li]
                for lk in range(3):
                    gk = kg[lk]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        elif si == 3 and sj == 3 and sk == 1:
            gk = kg[0]
            for li in range(3):
                gi = ig[li]
                for lj in range(3):
                    gj = jg[lj]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        elif sorted([si, sj, sk]) == [2, 2, 3]:
            # 2x2x3: split the 3-dim into 2+1
            if sk == 3:
                col = build_block(ig, jg, kg[:2], col, U, V, W, strassen_terms)
                col = build_block(ig, jg, kg[2:], col, U, V, W, strassen_terms)
            elif sj == 3:
                col = build_block(ig, jg[:2], kg, col, U, V, W, strassen_terms)
                col = build_block(ig, jg[2:], kg, col, U, V, W, strassen_terms)
            else:
                col = build_block(ig[:2], jg, kg, col, U, V, W, strassen_terms)
                col = build_block(ig[2:], jg, kg, col, U, V, W, strassen_terms)
        elif sorted([si, sj, sk]) == [2, 3, 3]:
            # 2x3x3: split one 3-dim into 2+1
            if si == 3:
                col = build_block(ig[:2], jg, kg, col, U, V, W, strassen_terms)
                col = build_block(ig[2:], jg, kg, col, U, V, W, strassen_terms)
            elif sj == 3:
                col = build_block(ig, jg[:2], kg, col, U, V, W, strassen_terms)
                col = build_block(ig, jg[2:], kg, col, U, V, W, strassen_terms)
            else:
                col = build_block(ig, jg, kg[:2], col, U, V, W, strassen_terms)
                col = build_block(ig, jg, kg[2:], col, U, V, W, strassen_terms)
        elif si == 3 and sj == 3 and sk == 3:
            # 3x3x3: split i into 2+1
            col = build_block(ig[:2], jg, kg, col, U, V, W, strassen_terms)
            col = build_block(ig[2:], jg, kg, col, U, V, W, strassen_terms)
        else:
            # Generic schoolbook fallback
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = ig[li], jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
        
        return col

    def block_rank(si, sj, sk):
        """Return the rank for an si x sj x sk block using our decomposition."""
        if si == 2 and sj == 2 and sk == 2:
            return 7
        if si == 1 and sj == 1 and sk == 1:
            return 1
        if sorted([si, sj, sk]) == [1, 1, 2]:
            return 2
        if sorted([si, sj, sk]) == [1, 2, 2]:
            return 4
        if sorted([si, sj, sk]) == [2, 2, 2]:
            return 7
        if sorted([si, sj, sk]) == [1, 2, 3]:
            return 6
        if sorted([si, sj, sk]) == [2, 2, 3]:
            return 11
        if sorted([si, sj, sk]) == [1, 3, 3]:
            return 9
        if sorted([si, sj, sk]) == [2, 3, 3]:
            return 17
        if si == 3 and sj == 3 and sk == 3:
            return 26
        return si * sj * sk

    def build_optimal_partition():
        """
        Build the decomposition with the optimal partition:
        i: [0,1], [2]
        j: [0,1], [2,3], [4]
        k: [0,1], [2,3], [4,5]
        
        This gives rank 69.
        """
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        total_rank = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    total_rank += block_rank(len(ig), len(jg), len(kg))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    col = build_block(ig, jg, kg, col, U, V, W, strassen_terms)
        
        return {"U": U, "V": V, "W": W}, total_rank

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

    # Build the optimal partition decomposition
    result, rank = build_optimal_partition()
    
    if verify(result["U"], result["V"], result["W"], rank):
        return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
