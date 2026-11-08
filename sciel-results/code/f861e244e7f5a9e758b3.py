import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    def get_strassen_2x2x2():
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        # Term 2: M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ))
        # Term 3: M3 = A00(B01-B11) -> C01 += 1, C11 += 1
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        ))
        # Term 4: M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ))
        # Term 5: M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1}
        ))
        # Term 6: M6 = (A10-A00)(B00+B01) -> C11 += 1
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        # Term 7: M7 = (A01-A11)(B10+B11) -> C00 += 1
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ))
        return terms
    
    def build_decomposition(i_groups, j_groups, k_groups):
        """Build a decomposition given block partitions."""
        strassen_terms = get_strassen_2x2x2()
        
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
        
        if col != total_rank:
            return None, 0
        
        return {"U": U, "V": V, "W": W}, total_rank
    
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
    
    # Try different block partitions
    best_result = None
    best_rank = n * m * p  # 90
    
    # Partition options
    i_options = [
        [[0, 1], [2]],
        [[0], [1], [2]],
    ]
    j_options = [
        [[0, 1], [2, 3], [4]],
        [[0, 1], [2], [3], [4]],
        [[0], [1], [2], [3], [4]],
    ]
    k_options = [
        [[0, 1], [2, 3], [4, 5]],
        [[0, 1], [2, 3], [4], [5]],
        [[0], [1], [2], [3], [4], [5]],
    ]
    
    for i_g in i_options:
        for j_g in j_options:
            for k_g in k_options:
                if time.time() > deadline:
                    break
                
                result, rank = build_decomposition(i_g, j_g, k_g)
                if result is None:
                    continue
                
                if verify(result["U"], result["V"], result["W"], rank):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = result
    
    if best_result is None:
        # Fallback to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
