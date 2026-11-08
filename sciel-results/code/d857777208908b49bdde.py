"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

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

    def laderman_3x3x3_terms():
        """
        Return the 23 Laderman terms for 3x3x3 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        
        Based on Laderman's 1976 algorithm.
        """
        terms = []
        
        # We'll use a known rank-23 decomposition.
        # The 3x3x3 tensor has 27 entries. We need 23 rank-1 tensors.
        
        # Let me use a systematic construction.
        # For a 3x3x3 matrix multiplication, the tensor T[i,j,k] = 1 if (i,j,k) corresponds to C[i,k] += A[i,j]*B[j,k]
        
        # Using the standard Laderman construction:
        # M1 = (A00 + A11 + A22)(B00 + B11 + B22)
        u1 = {(0,0): 1, (1,1): 1, (2,2): 1}
        v1 = {(0,0): 1, (1,1): 1, (2,2): 1}
        w1 = {(0,0): 1, (1,1): 1, (2,2): 1}
        terms.append((u1, v1, w1))
        
        # M2 = (A00 + A11 - A22)(B00 + B11 - B22)
        u2 = {(0,0): 1, (1,1): 1, (2,2): -1}
        v2 = {(0,0): 1, (1,1): 1, (2,2): -1}
        w2 = {(0,0): 1, (1,1): 1, (2,2): -1}
        terms.append((u2, v2, w2))
        
        # M3 = (A00 - A11 + A22)(B00 - B11 + B22)
        u3 = {(0,0): 1, (1,1): -1, (2,2): 1}
        v3 = {(0,0): 1, (1,1): -1, (2,2): 1}
        w3 = {(0,0): 1, (1,1): -1, (2,2): 1}
        terms.append((u3, v3, w3))
        
        # M4 = (A01 + A12 + A20)(B01 + B12 + B20)
        u4 = {(0,1): 1, (1,2): 1, (2,0): 1}
        v4 = {(0,1): 1, (1,2): 1, (2,0): 1}
        w4 = {(0,1): 1, (1,2): 1, (2,0): 1}
        terms.append((u4, v4, w4))
        
        # M5 = (A01 + A12 - A20)(B01 + B12 - B20)
        u5 = {(0,1): 1, (1,2): 1, (2,0): -1}
        v5 = {(0,1): 1, (1,2): 1, (2,0): -1}
        w5 = {(0,1): 1, (1,2): 1, (2,0): -1}
        terms.append((u5, v5, w5))
        
        # M6 = (A01 - A12 + A20)(B01 - B12 + B20)
        u6 = {(0,1): 1, (1,2): -1, (2,0): 1}
        v6 = {(0,1): 1, (1,2): -1, (2,0): 1}
        w6 = {(0,1): 1, (1,2): -1, (2,0): 1}
        terms.append((u6, v6, w6))
        
        # M7 = (A02 + A10 + A21)(B02 + B10 + B21)
        u7 = {(0,2): 1, (1,0): 1, (2,1): 1}
        v7 = {(0,2): 1, (1,0): 1, (2,1): 1}
        w7 = {(0,2): 1, (1,0): 1, (2,1): 1}
        terms.append((u7, v7, w7))
        
        # M8 = (A02 + A10 - A21)(B02 + B10 - B21)
        u8 = {(0,2): 1, (1,0): 1, (2,1): -1}
        v8 = {(0,2): 1, (1,0): 1, (2,1): -1}
        w8 = {(0,2): 1, (1,0): 1, (2,1): -1}
        terms.append((u8, v8, w8))
        
        # M9 = (A02 - A10 + A21)(B02 - B10 + B21)
        u9 = {(0,2): 1, (1,0): -1, (2,1): 1}
        v9 = {(0,2): 1, (1,0): -1, (2,1): 1}
        w9 = {(0,2): 1, (1,0): -1, (2,1): 1}
        terms.append((u9, v9, w9))
        
        # M10 = A00*B10
        u10 = {(0,0): 1}
        v10 = {(1,0): 1}
        w10 = {(0,0): 1, (1,1): 1, (2,2): 1}
        terms.append((u10, v10, w10))
        
        # M11 = A11*B21
        u11 = {(1,1): 1}
        v11 = {(2,1): 1}
        w11 = {(0,0): 1, (1,1): 1, (2,2): 1}
        terms.append((u11, v11, w11))
        
        # M12 = A22*B02
        u12 = {(2,2): 1}
        v12 = {(0,2): 1}
        w12 = {(0,0): 1, (1,1): 1, (2,2): 1}
        terms.append((u12, v12, w12))
        
        # M13 = A01*B11
        u13 = {(0,1): 1}
        v13 = {(1,1): 1}
        w13 = {(0,1): 1, (1,2): 1, (2,0): 1}
        terms.append((u13, v13, w13))
        
        # M14 = A12*B22
        u14 = {(1,2): 1}
        v14 = {(2,2): 1}
        w14 = {(0,1): 1, (1,2): 1, (2,0): 1}
        terms.append((u14, v14, w14))
        
        # M15 = A20*B00
        u15 = {(2,0): 1}
        v15 = {(0,0): 1}
        w15 = {(0,1): 1, (1,2): 1, (2,0): 1}
        terms.append((u15, v15, w15))
        
        # M16 = A02*B20
        u16 = {(0,2): 1}
        v16 = {(2,0): 1}
        w16 = {(0,2): 1, (1,0): 1, (2,1): 1}
        terms.append((u16, v16, w16))
        
        # M17 = A10*B01
        u17 = {(1,0): 1}
        v17 = {(0,1): 1}
        w17 = {(0,2): 1, (1,0): 1, (2,1): 1}
        terms.append((u17, v17, w17))
        
        # M18 = A21*B12
        u18 = {(2,1): 1}
        v18 = {(1,2): 1}
        w18 = {(0,2): 1, (1,0): 1, (2,1): 1}
        terms.append((u18, v18, w18))
        
        # M19 = A00*B21
        u19 = {(0,0): 1}
        v19 = {(2,1): 1}
        w19 = {(0,0): 1, (1,1): 1, (2,2): 1}
        terms.append((u19, v19, w19))
        
        # M20 = A11*B02
        u20 = {(1,1): 1}
        v20 = {(0,2): 1}
        w20 = {(0,0): 1, (1,1): 1, (2,2): 1}
        terms.append((u20, v20, w20))
        
        # M21 = A22*B10
        u21 = {(2,2): 1}
        v21 = {(1,0): 1}
        w21 = {(0,0): 1, (1,1): 1, (2,2): 1}
        terms.append((u21, v21, w21))
        
        # M22 = A01*B22
        u22 = {(0,1): 1}
        v22 = {(2,2): 1}
        w22 = {(0,1): 1, (1,2): 1, (2,0): 1}
        terms.append((u22, v22, w22))
        
        # M23 = A12*B00
        u23 = {(1,2): 1}
        v23 = {(0,0): 1}
        w23 = {(0,1): 1, (1,2): 1, (2,0): 1}
        terms.append((u23, v23, w23))
        
        return terms

    def build_block_decomposition(i_groups, j_groups, k_groups, use_laderman=True):
        """
        Build a decomposition using Strassen/Laderman for 2x2x2/3x3x3 blocks
        and schoolbook elsewhere.
        """
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    elif si == 3 and sj == 3 and sk == 3 and use_laderman:
                        r = 23  # Laderman
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        laderman_terms = laderman_3x3x3_terms()
        
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
            elif si == 3 and sj == 3 and sk == 3 and use_laderman:
                for (u_pat, v_pat, w_pat) in laderman_terms:
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
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def try_decomposition():
        """Try various block partitions and return the best valid decomposition."""
        if time.time() > deadline:
            return None
        
        best_result = None
        best_rank = float('inf')
        
        # Partition 1: Strassen only (current approach)
        i_groups1 = [[0, 1], [2]]
        j_groups1 = [[0, 1], [2, 3], [4]]
        k_groups1 = [[0, 1], [2, 3], [4, 5]]
        
        result1 = build_block_decomposition(i_groups1, j_groups1, k_groups1, use_laderman=False)
        rank1 = result1.pop("rank")
        if verify(result1["U"], result1["V"], result1["W"], rank1):
            if rank1 < best_rank:
                best_rank = rank1
                best_result = result1
        
        # Partition 2: Try to use Laderman on a 3x3x3 block
        # i=[0,1,2], j=[0,1,2], k=[0,1,2] -> 3x3x3
        # Remaining: j=[3,4], k=[3,4,5] etc.
        i_groups2 = [[0, 1, 2]]
        j_groups2 = [[0, 1, 2], [3, 4]]
        k_groups2 = [[0, 1, 2], [3, 4, 5]]
        
        if time.time() < deadline - 5:
            result2 = build_block_decomposition(i_groups2, j_groups2, k_groups2, use_laderman=True)
            rank2 = result2.pop("rank")
            if verify(result2["U"], result2["V"], result2["W"], rank2):
                if rank2 < best_rank:
                    best_rank = rank2
                    best_result = result2
        
        # Partition 3: Another Laderman partition
        # i=[0,1,2], j=[0,1,2,3,4], k=[0,1,2,3,4,5] - but this doesn't help since j and k are larger than 3
        # Let's try: i=[0,1,2], j=[0,1,2], k=[0,1,2] for Laderman, rest schoolbook
        # This is the same as partition 2
        
        # Partition 4: Try a different 3x3x3 placement
        i_groups4 = [[0, 1, 2]]
        j_groups4 = [[0, 1, 2], [3, 4]]
        k_groups4 = [[3, 4, 5], [0, 1, 2]]
        
        if time.time() < deadline - 5:
            result4 = build_block_decomposition(i_groups4, j_groups4, k_groups4, use_laderman=True)
            rank4 = result4.pop("rank")
            if verify(result4["U"], result4["V"], result4["W"], rank4):
                if rank4 < best_rank:
                    best_rank = rank4
                    best_result = result4
        
        return best_result

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try improved decomposition
    improved = try_decomposition()
    if improved is not None:
        improved_rank = len(improved["U"][0])
        if improved_rank < best_rank:
            best_result = improved
            best_rank = improved_rank

    return best_result
# EVOLVE-BLOCK-END
